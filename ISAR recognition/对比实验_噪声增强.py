import matplotlib
matplotlib.use('Agg')

import os
import glob
import random
import copy
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_ROOT = "./序列数据集3-包含seq2和seq4"
RUN_NAME = "对比实验_噪声增强"
NUM_CLASSES = 8
SEQ_LEN = 6
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

CENTER_LOSS_WEIGHT = 0.05
TRIPLET_LOSS_WEIGHT =  0.2
MARGIN = 1.5
NUM_REPEATS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.7)
NUM_WORKERS = 0
TEST_SNR_LIST = [-20,-15,-10,-5,0,5,10,15,20]

# Noise-augmentation settings
TRAIN_NOISE_PROB = 0.70
TRAIN_SNR_BINS = (
    (-20,-10),
    (-10, 0),
    (0, 10),
    (10, 20),
)
TRAIN_SNR_WEIGHTS = (0.40, 0.30, 0.20, 0.10)

# Noise transforms
class AddGaussianNoise:
    def __init__(self, snr_db):
        self.snr_db = snr_db

    @staticmethod
    def _stable_seed(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.blake2b(array.tobytes(), digest_size=8).digest()
        return int.from_bytes(digest, byteorder='little', signed=False) & 0x7FFFFFFFFFFFFFFF

    def __call__(self, tensor):
        signal_std = tensor.std()
        if signal_std < 1e-6:
            return tensor

        noise_std = signal_std * (10 ** (-self.snr_db / 20))

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self._stable_seed(tensor))
        noise = torch.randn(
            tensor.shape,
            generator=generator,
            dtype=tensor.dtype,
            device='cpu'
        ).to(tensor.device) * noise_std

        noisy = tensor + noise
        return torch.clamp(noisy, 0.0, 1.0)

def sample_weighted_train_snr(snr_bins=TRAIN_SNR_BINS, snr_weights=TRAIN_SNR_WEIGHTS):
    if len(snr_bins) == 0:
        raise ValueError("snr_bins 不能为空")
    if len(snr_bins) != len(snr_weights):
        raise ValueError("snr_bins 与 snr_weights 长度必须一致")
    if any(float(w) < 0.0 for w in snr_weights):
        raise ValueError("snr_weights 不能包含负数")
    if sum(float(w) for w in snr_weights) <= 0.0:
        raise ValueError("snr_weights 权重和必须大于 0")

    clean_bins = []
    for low, high in snr_bins:
        low = float(low)
        high = float(high)
        if low > high:
            raise ValueError(f"非法 SNR 区间: ({low}, {high})")
        clean_bins.append((low, high))

    bin_idx = random.choices(
        population=range(len(clean_bins)),
        weights=[float(w) for w in snr_weights],
        k=1
    )[0]
    low, high = clean_bins[bin_idx]
    return random.uniform(low, high)

class RandomGaussianNoiseTrain:
    def __init__(
        self,
        probability=TRAIN_NOISE_PROB,
        snr_bins=TRAIN_SNR_BINS,
        snr_weights=TRAIN_SNR_WEIGHTS,
    ):
        self.probability = float(probability)
        self.snr_bins = tuple((float(a), float(b)) for a, b in snr_bins)
        self.snr_weights = tuple(float(w) for w in snr_weights)

        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("probability 必须位于 [0, 1]")

        if len(self.snr_bins) == 0 or len(self.snr_bins) != len(self.snr_weights):
            raise ValueError("snr_bins 与 snr_weights 必须非空且长度一致")
        if any(a > b for a, b in self.snr_bins):
            raise ValueError("snr_bins 中存在下限大于上限的区间")
        if any(w < 0.0 for w in self.snr_weights) or sum(self.snr_weights) <= 0.0:
            raise ValueError("snr_weights 必须为非负数且权重和大于 0")

    @staticmethod
    def add_noise_at_snr(tensor, snr_db):
        signal_std = tensor.std()
        if signal_std < 1e-6:
            return tensor

        noise_std = signal_std * (10 ** (-float(snr_db) / 20.0))
        noise = torch.randn_like(tensor) * noise_std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def sample_snr(self):
        return sample_weighted_train_snr(self.snr_bins, self.snr_weights)

    def __call__(self, tensor):
        if random.random() >= self.probability:
            return tensor

        snr_db = self.sample_snr()
        return self.add_noise_at_snr(tensor, snr_db)

class SequenceConsistentTransform:
    def __init__(
        self,
        train=False,
        noise_snr=None,
        train_noise=False,
        train_noise_prob=TRAIN_NOISE_PROB,
        train_snr_bins=TRAIN_SNR_BINS,
        train_snr_weights=TRAIN_SNR_WEIGHTS,
    ):
        self.train = train

        self.noise = AddGaussianNoise(noise_snr) if noise_snr is not None else None

        self.train_noise = bool(train_noise)
        self.train_noise_prob = float(train_noise_prob)
        self.train_snr_bins = tuple((float(a), float(b)) for a, b in train_snr_bins)
        self.train_snr_weights = tuple(float(w) for w in train_snr_weights)

        if not (0.0 <= self.train_noise_prob <= 1.0):
            raise ValueError("train_noise_prob 必须位于 [0, 1]")
        if len(self.train_snr_bins) == 0 or len(self.train_snr_bins) != len(self.train_snr_weights):
            raise ValueError("train_snr_bins 与 train_snr_weights 必须非空且长度一致")
        if any(a > b for a, b in self.train_snr_bins):
            raise ValueError("train_snr_bins 中存在下限大于上限的区间")
        if any(w < 0.0 for w in self.train_snr_weights) or sum(self.train_snr_weights) <= 0.0:
            raise ValueError("train_snr_weights 必须为非负数且权重和大于 0")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.brightness = [0.8, 1.2]
        self.contrast = [0.8, 1.2]
        self.saturation = [0.8, 1.2]
        self.hue = None

    def __call__(self, images):
        images = [TF.resize(img, [IMG_SIZE, IMG_SIZE]) for img in images]

        if self.train:

            do_flip = torch.rand(1).item() < 0.5

            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                transforms.ColorJitter.get_params(
                    self.brightness,
                    self.contrast,
                    self.saturation,
                    self.hue
                )

            augmented = []
            for img in images:
                if do_flip:
                    img = TF.hflip(img)

                for fn_id in fn_idx:
                    fn_id = int(fn_id)
                    if fn_id == 0 and brightness_factor is not None:
                        img = TF.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = TF.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = TF.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = TF.adjust_hue(img, hue_factor)

                augmented.append(img)
            images = augmented

        use_train_noise = (
            self.train
            and self.train_noise
            and random.random() < self.train_noise_prob
        )

        train_snr_db = None
        if use_train_noise:

            train_snr_db = sample_weighted_train_snr(
                self.train_snr_bins,
                self.train_snr_weights
            )

        tensors = []
        for img in images:
            tensor = TF.to_tensor(img)

            if train_snr_db is not None:

                tensor = RandomGaussianNoiseTrain.add_noise_at_snr(
                    tensor,
                    train_snr_db
                )

            elif self.noise is not None:

                tensor = self.noise(tensor)

            tensor = self.normalize(tensor)
            tensors.append(tensor)

        return torch.stack(tensors, dim=0)

# Dataset definitions
class AllSingleFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not os.path.isdir(root_dir):
            raise ValueError(f"根目录不存在: {root_dir}")
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_path):
                for fname in files:
                    if fname.endswith('.png'):
                        img_path = os.path.join(root, fname)
                        if os.path.getsize(img_path) > 100:
                            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class AllISARSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_erase=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_erase = use_erase
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not os.path.isdir(root_dir):
            raise ValueError(f"根目录不存在: {root_dir}")
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)

            for root, dirs, files in os.walk(class_path):

                png_files = [f for f in files if f.endswith('.png')]
                if len(png_files) == SEQ_LEN:

                    valid = True
                    img_paths = []
                    for fname in sorted(png_files):
                        fpath = os.path.join(root, fname)
                        if os.path.getsize(fpath) < 100:
                            valid = False
                            break
                        img_paths.append(fpath)
                    if valid:
                        self.samples.append((img_paths, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        images = [Image.open(path).convert('RGB') for path in img_paths]

        if self.transform:

            seq_tensor = self.transform(images)
        else:
            seq_tensor = torch.stack([TF.to_tensor(img) for img in images], dim=0)

        if self.use_erase and random.random() < 0.5:
            seq_tensor = self.random_erase_sequence(seq_tensor)
        return seq_tensor, label

    @staticmethod
    def random_erase_sequence(seq_tensor, max_area=0.3, num_patches=2):
        _, _, H, W = seq_tensor.shape
        for _ in range(num_patches):
            area_ratio = random.uniform(0.02, max_area)
            er_h = int(H * np.sqrt(area_ratio))
            er_w = int(W * np.sqrt(area_ratio))
            x1 = random.randint(0, W - er_w) if W > er_w else 0
            y1 = random.randint(0, H - er_h) if H > er_h else 0
            seq_tensor[:, :, y1:y1+er_h, x1:x1+er_w] = 0.0
        return seq_tensor

class SingleFrameSubset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class SequenceSubset(Dataset):
    def __init__(self, samples, transform, use_erase=False):
        self.samples = samples
        self.transform = transform
        self.use_erase = use_erase
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        images = [Image.open(path).convert('RGB') for path in img_paths]

        if self.transform:

            seq_tensor = self.transform(images)
        else:
            seq_tensor = torch.stack([TF.to_tensor(img) for img in images], dim=0)

        if self.use_erase and random.random() < 0.5:
            seq_tensor = self.random_erase_sequence(seq_tensor)
        return seq_tensor, label

    @staticmethod
    def random_erase_sequence(seq_tensor, max_area=0.3, num_patches=2):
        _, _, H, W = seq_tensor.shape
        for _ in range(num_patches):
            area_ratio = random.uniform(0.02, max_area)
            er_h = int(H * np.sqrt(area_ratio))
            er_w = int(W * np.sqrt(area_ratio))
            x1 = random.randint(0, W - er_w) if W > er_w else 0
            y1 = random.randint(0, H - er_h) if H > er_h else 0
            seq_tensor[:, :, y1:y1+er_h, x1:x1+er_w] = 0.0
        return seq_tensor

# Baseline models
class SingleFrameClassifier(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=NUM_CLASSES):
        super().__init__()
        if backbone == 'resnet18':
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 512
        elif backbone == 'efficientnet-b0':
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        feat = self.backbone(x)
        if len(feat.shape) == 4:
            feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class DecisionFusionClassifier(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=NUM_CLASSES, fusion_mode='average'):
        super().__init__()
        self.fusion_mode = fusion_mode
        if backbone == 'resnet18':
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 512
        elif backbone == 'efficientnet-b0':
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feat = self.backbone(x_reshaped)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        logits = logits.view(B, T, -1)
        return logits.mean(dim=1)

class MVCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feat = self.backbone(x_reshaped)
        feat = feat.view(feat.size(0), -1)
        feat = feat.view(B, T, -1)
        feat_pooled, _ = torch.max(feat, dim=1)
        return self.fc(feat_pooled)

class MIIR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.attn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feat = self.backbone(x_reshaped)
        feat = feat.view(feat.size(0), -1)
        feat = feat.view(B, T, -1)
        attn_weights = F.softmax(self.attn(feat).squeeze(-1), dim=1)
        feat_weighted = torch.bmm(attn_weights.unsqueeze(1), feat).squeeze(1)
        return self.fc(feat_weighted)

class MVFRnet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feat = self.backbone(x_reshaped)
        feat = feat.view(feat.size(0), -1)
        feat = feat.view(B, T, -1)
        pair_features = []
        for i in range(T):
            for j in range(i + 1, T):
                pair_diff = torch.abs(feat[:, i, :] - feat[:, j, :])
                pair_features.append(pair_diff)
        relation_feat = torch.stack(pair_features, dim=1).mean(dim=1)
        feat_mean = feat.mean(dim=1)
        return self.fc(feat_mean + relation_feat * 0.1)

class C3D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# STFusionNet modules
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()

        self.branch1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, in_channels//2, 3, padding=2, dilation=2, bias=False)
        self.branch3 = nn.Conv2d(in_channels, in_channels//2, 3, padding=3, dilation=3, bias=False)
        self.fuse = nn.Conv2d(in_channels//2 * 3, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        multi = torch.cat([b1, b2, b3], dim=1)
        multi = self.fuse(multi)
        multi = self.bn(multi)
        multi = self.relu(multi)
        att_map = self.spatial_att(multi)
        return x * att_map + x

class ISAR_FeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super().__init__()
        if pretrained:
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            base_model = resnet18(weights=None)

        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.attention = MultiScaleAttention(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        feat = self.pool(feat).flatten(1)
        return feat

class TemporalEvolutionModule(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        self.bi_gru = nn.GRU(feature_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim*2,
                                                    num_heads=num_heads,
                                                    batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim*2)

    def forward(self, x):
        gru_out, _ = self.bi_gru(x)
        attn_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)
        out = self.norm(gru_out + attn_out)
        return out

class HierarchicalAttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.att_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1, bias=False)
        )

    def forward(self, x):
        scores = self.att_fc(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weighted, weights

class STFusionNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=SEQ_LEN, feature_dim=512,
                 hidden_dim=256, num_heads=8):
        super().__init__()
        self.feature_extractor = ISAR_FeatureExtractor()
        self.temporal_module = TemporalEvolutionModule(feature_dim, hidden_dim, num_heads)
        self.pooling = HierarchicalAttentionPooling(hidden_dim*2)
        self.classifier = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x_reshaped = x.view(B*T, C, H, W)
        spatial_feat = self.feature_extractor(x_reshaped)
        spatial_feat = spatial_feat.view(B, T, -1)

        temporal_feat = self.temporal_module(spatial_feat)

        fused_feat, att_weights = self.pooling(temporal_feat)

        logits = self.classifier(fused_feat)
        return logits, fused_feat, att_weights

# Loss functions
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)
        return loss

class JointLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, center_weight=0.005,
                 triplet_weight=0.01, margin=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.center_loss_fn = CenterLoss(num_classes, feat_dim)
        self.margin = margin
        self.center_weight = center_weight
        self.triplet_weight = triplet_weight

    def forward(self, logits, features, labels):
        loss_ce = self.ce_loss(logits, labels)
        loss_center = self.center_loss_fn(features, labels)
        loss_triplet = self._batch_random_triplet_loss(features, labels)

        total = loss_ce + self.center_weight * loss_center + self.triplet_weight * loss_triplet
        return total, loss_ce.item(), loss_center.item(), loss_triplet.item()

    def _batch_random_triplet_loss(self, features, labels):
        batch_size = features.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=features.device)

        triplets = []
        for i in range(batch_size):
            label = labels[i]
            pos_mask = (labels == label) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = (labels != label)

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,)).item()]
            neg_idx = neg_indices[torch.randint(len(neg_indices), (1,)).item()]

            anchor   = features[i].unsqueeze(0)
            positive = features[pos_idx].unsqueeze(0)
            negative = features[neg_idx].unsqueeze(0)

            loss = F.triplet_margin_loss(anchor, positive, negative,
                                         margin=self.margin, p=2, reduction='mean')
            triplets.append(loss)

        if len(triplets) == 0:
            return torch.tensor(0.0, device=features.device)
        return torch.stack(triplets).mean()

# Training and evaluation
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_total = 0.0
    running_ce = 0.0
    running_center = 0.0
    running_triplet = 0.0
    all_preds, all_labels = [], []

    for seqs, labels in dataloader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, features, _ = model(seqs)
        total_loss, ce, ct, tr = criterion(logits, features, labels)

        total_loss.backward()
        optimizer.step()

        batch_size = seqs.size(0)
        running_total += total_loss.item() * batch_size
        running_ce += ce * batch_size
        running_center += ct * batch_size
        running_triplet += tr * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n_samples = len(dataloader.dataset)
    avg_total = running_total / n_samples
    avg_ce = running_ce / n_samples
    avg_center = running_center / n_samples
    avg_triplet = running_triplet / n_samples
    epoch_acc = accuracy_score(all_labels, all_preds)

    loss_stats = {
        'total': avg_total,
        'ce': avg_ce,
        'center': avg_center,
        'triplet': avg_triplet,
    }
    return loss_stats, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits, features, _ = model(seqs)
            total_loss, _, _, _ = criterion(logits, features, labels)
            running_loss += total_loss.item() * seqs.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def train_single_frame(model, train_loader, val_loader, num_epochs=EPOCHS, save_path=None):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"  Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_acc': best_val_acc
                }, save_path)
                print(f"    Best model saved to {save_path} (val_acc={best_val_acc:.4f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def train_sequence_model(model, train_loader, val_loader, num_epochs=EPOCHS, save_path=None):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(DEVICE)
                logits = model(seqs)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"  Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_acc': best_val_acc
                }, save_path)
                print(f"    Best model saved to {save_path} (val_acc={best_val_acc:.4f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def train_stfusion_model(model, train_loader, val_loader, num_epochs=EPOCHS, save_path=None):
    model = model.to(DEVICE)

    criterion = JointLoss(
        NUM_CLASSES,
        feat_dim=512,
        center_weight=CENTER_LOSS_WEIGHT,
        triplet_weight=TRIPLET_LOSS_WEIGHT,
        margin=MARGIN
    ).to(DEVICE)

    optimizer = optim.Adam([
        {
            'params': model.parameters(),
            'lr': LEARNING_RATE,
        },
        {
            'params': criterion.center_loss_fn.parameters(),
            'lr': LEARNING_RATE,
        },
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_model_state = None

    loss_history = {
        'total': [],
        'ce': [],
        'center': [],
        'triplet': [],
        'weighted_center': [],
        'weighted_triplet': [],
        'center_to_ce_percent': [],
        'triplet_to_ce_percent': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    print(
        f"    JointLoss weights: Center={CENTER_LOSS_WEIGHT:g}, "
        f"Triplet={TRIPLET_LOSS_WEIGHT:g}"
    )
    print(
        "    Optimizer: model parameters + CenterLoss.centers "
        f"(center parameters={sum(p.numel() for p in criterion.center_loss_fn.parameters())})"
    )

    for epoch in range(num_epochs):
        train_stats, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)

        train_total = train_stats['total']
        train_ce = train_stats['ce']
        train_center = train_stats['center']
        train_triplet = train_stats['triplet']

        weighted_center = CENTER_LOSS_WEIGHT * train_center
        weighted_triplet = TRIPLET_LOSS_WEIGHT * train_triplet

        ce_safe = max(train_ce, 1e-12)
        center_ratio = weighted_center / ce_safe * 100.0
        triplet_ratio = weighted_triplet / ce_safe * 100.0

        loss_history['total'].append(train_total)
        loss_history['ce'].append(train_ce)
        loss_history['center'].append(train_center)
        loss_history['triplet'].append(train_triplet)
        loss_history['weighted_center'].append(weighted_center)
        loss_history['weighted_triplet'].append(weighted_triplet)
        loss_history['center_to_ce_percent'].append(center_ratio)
        loss_history['triplet_to_ce_percent'].append(triplet_ratio)
        loss_history['train_acc'].append(train_acc)
        loss_history['val_loss'].append(val_loss)
        loss_history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1}/{num_epochs}")
        print(f"    Train Acc            : {train_acc:.4f}")
        print(f"    Total Loss           : {train_total:.6f}")
        print(f"    CE Loss              : {train_ce:.6f}")
        print(
            f"    Center Loss          : {train_center:.6f}  "
            f"(weighted={weighted_center:.6f}, {center_ratio:.2f}% of CE)"
        )
        print(
            f"    Triplet Loss         : {train_triplet:.6f}  "
            f"(weighted={weighted_triplet:.6f}, {triplet_ratio:.2f}% of CE)"
        )
        print(f"    Val Loss             : {val_loss:.6f}")
        print(f"    Val Acc              : {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if save_path is not None:

                torch.save(best_model_state, save_path)
                print(f"    Best STFusionNet saved to {save_path} (val_acc={best_val_acc:.4f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if save_path is not None:
        history_path = os.path.splitext(save_path)[0] + '_loss_history.npz'
        np.savez(
            history_path,
            **{key: np.asarray(value, dtype=np.float64)
               for key, value in loss_history.items()}
        )
        print(f"    STFusionNet loss history saved to {history_path}")

    return model

def evaluate_model(model, test_loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            output = model(data)

            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, average='weighted', zero_division=0)
    rec  = recall_score(labels_all, preds_all, average='weighted', zero_division=0)
    f1   = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def stratified_split_by_class(samples, class_to_idx, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    class_samples = {idx: [] for idx in class_to_idx.values()}
    for sample in samples:
        label = sample[1]
        class_samples[label].append(sample)

    train_list, val_list, test_list = [], [], []
    for label, items in class_samples.items():
        random.shuffle(items)
        total = len(items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_list.extend(items[:train_end])
        val_list.extend(items[train_end:val_end])
        test_list.extend(items[val_end:])
    return train_list, val_list, test_list

# Experiment runner
def run_all_experiments():
    base_results_dir = "./results_噪声增强"
    run_dir = os.path.join(base_results_dir, RUN_NAME)
    if os.path.exists(run_dir):
        print(f"警告：目录 '{run_dir}' 已存在，可能会覆盖之前的文件。")
    os.makedirs(run_dir, exist_ok=True)

    global MODEL_SAVE_DIR
    MODEL_SAVE_DIR = os.path.join(run_dir, "best_models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(
        f"训练噪声增强: noisy probability={TRAIN_NOISE_PROB:.2f}, "
        f"clean probability={1.0 - TRAIN_NOISE_PROB:.2f}"
    )
    print("  条件于 noisy 样本的 SNR 加权分布：")
    for (low, high), weight in zip(TRAIN_SNR_BINS, TRAIN_SNR_WEIGHTS):
        effective = TRAIN_NOISE_PROB * weight
        print(
            f"    {low:.0f}~{high:.0f} dB: noisy内部 {weight*100:.1f}% "
            f"/ 全部训练样本期望 {effective*100:.1f}%"
        )
    print(
        "  单帧模型：每张图片独立决定是否加噪并采样SNR；"
        "序列模型：整段序列共享SNR、各帧独立Gaussian realization。"
    )

    experiments = [
        ('B4_STFusionNet_Ours', 'sequence', lambda: STFusionNet()),
        ('B1_ResNet18_Single', 'single', lambda: SingleFrameClassifier('resnet18')),
        ('B1_EfficientNet_Single', 'single', lambda: SingleFrameClassifier('efficientnet-b0')),
        ('B2_ResNet18_Average', 'sequence', lambda: DecisionFusionClassifier('resnet18', fusion_mode='average')),
        ('B2_EfficientNet_Average', 'sequence', lambda: DecisionFusionClassifier('efficientnet-b0', fusion_mode='average')),
        ('B3_MVCNN', 'sequence', lambda: MVCNN()),
        ('B3_MIIR', 'sequence', lambda: MIIR()),
        ('B3_MVFRnet', 'sequence', lambda: MVFRnet()),
        ('B4_STFusionNet_Ours', 'sequence', lambda: STFusionNet()),
    ]

    metrics_clean = {name: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for name, _, _ in experiments}
    metrics_noise = {snr: {name: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for name, _, _ in experiments} for snr in TEST_SNR_LIST}

    print("正在加载全部序列样本...")
    full_dataset_seq = AllISARSequenceDataset(DATA_ROOT, transform=None, use_erase=False)
    all_seq_samples = full_dataset_seq.samples
    class_to_idx = full_dataset_seq.class_to_idx
    print(f"序列样本总数: {len(all_seq_samples)}")

    for run_idx in range(NUM_REPEATS):
        print(f"\n{'='*60}")
        print(f"重复实验 {run_idx+1}/{NUM_REPEATS}")
        print(f"{'='*60}")

        seed = 46 + run_idx
        print(f"  划分数据集 (seed={seed})...")

        train_seq, val_seq, test_seq = stratified_split_by_class(
            all_seq_samples, class_to_idx,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=seed
        )
        print(f"  训练序列: {len(train_seq)}, 验证序列: {len(val_seq)}, 测试序列: {len(test_seq)}")

        def expand_sequences(seq_list):
            img_samples = []
            for img_paths, label in seq_list:
                for img_path in img_paths:
                    img_samples.append((img_path, label))
            return img_samples

        train_s = expand_sequences(train_seq)
        val_s = expand_sequences(val_seq)
        test_s = expand_sequences(test_seq)
        print(f"  单帧训练图片: {len(train_s)}, 验证图片: {len(val_s)}, 测试图片: {len(test_s)}")

        def get_single_transforms(noise_snr=None):
            train_t = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),

                RandomGaussianNoiseTrain(
                    probability=TRAIN_NOISE_PROB,
                    snr_bins=TRAIN_SNR_BINS,
                    snr_weights=TRAIN_SNR_WEIGHTS,
                ),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_t = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if noise_snr is not None:
                test_t = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    AddGaussianNoise(noise_snr),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                test_t = val_t
            return train_t, val_t, test_t

        def get_seq_transforms(noise_snr=None):
            train_t = SequenceConsistentTransform(
                train=True,
                noise_snr=None,
                train_noise=True,
                train_noise_prob=TRAIN_NOISE_PROB,
                train_snr_bins=TRAIN_SNR_BINS,
                train_snr_weights=TRAIN_SNR_WEIGHTS,
            )
            val_t = SequenceConsistentTransform(
                train=False,
                noise_snr=None,
                train_noise=False,
            )
            if noise_snr is not None:
                test_t = SequenceConsistentTransform(
                    train=False,
                    noise_snr=noise_snr,
                    train_noise=False,
                )
            else:
                test_t = val_t
            return train_t, val_t, test_t

        train_t_s, val_t_s, _ = get_single_transforms(None)
        train_dataset_single = SingleFrameSubset(train_s, train_t_s)
        val_dataset_single = SingleFrameSubset(val_s, val_t_s)
        train_loader_single = DataLoader(train_dataset_single, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_single = DataLoader(val_dataset_single, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        train_t_seq, val_t_seq, _ = get_seq_transforms(None)
        train_dataset_seq = SequenceSubset(train_seq, train_t_seq, use_erase=True)
        val_dataset_seq = SequenceSubset(val_seq, val_t_seq, use_erase=False)
        train_loader_seq = DataLoader(train_dataset_seq, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_seq = DataLoader(val_dataset_seq, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        _, _, test_t_clean_single = get_single_transforms(None)
        _, _, test_t_clean_seq = get_seq_transforms(None)
        test_dataset_clean_single = SingleFrameSubset(test_s, test_t_clean_single)
        test_dataset_clean_seq = SequenceSubset(test_seq, test_t_clean_seq, use_erase=False)
        test_loader_clean_single = DataLoader(test_dataset_clean_single, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader_clean_seq = DataLoader(test_dataset_clean_seq, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        test_loaders_noisy_single = {}
        test_loaders_noisy_seq = {}
        for snr in TEST_SNR_LIST:
            _, _, test_t_noisy_single = get_single_transforms(snr)
            _, _, test_t_noisy_seq = get_seq_transforms(snr)
            test_dataset_noisy_single = SingleFrameSubset(test_s, test_t_noisy_single)
            test_dataset_noisy_seq = SequenceSubset(test_seq, test_t_noisy_seq, use_erase=False)
            test_loaders_noisy_single[snr] = DataLoader(test_dataset_noisy_single, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            test_loaders_noisy_seq[snr] = DataLoader(test_dataset_noisy_seq, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        for name, data_type, model_fn in experiments:
            print(f"  训练 {name}...")
            safe_name = name.replace('/', '_')
            save_path = os.path.join(MODEL_SAVE_DIR, f"run{run_idx}_{safe_name}.pth")
            try:
                if data_type == 'single':
                    model = train_single_frame(model_fn(), train_loader_single, val_loader_single,
                                               num_epochs=EPOCHS, save_path=save_path)

                    acc, prec, rec, f1 = evaluate_model(model, test_loader_clean_single)
                    metrics_clean[name]['acc'].append(acc)
                    metrics_clean[name]['prec'].append(prec)
                    metrics_clean[name]['rec'].append(rec)
                    metrics_clean[name]['f1'].append(f1)
                    print(f"    干净: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                    for snr in TEST_SNR_LIST:
                        acc, prec, rec, f1 = evaluate_model(model, test_loaders_noisy_single[snr])
                        metrics_noise[snr][name]['acc'].append(acc)
                        metrics_noise[snr][name]['prec'].append(prec)
                        metrics_noise[snr][name]['rec'].append(rec)
                        metrics_noise[snr][name]['f1'].append(f1)
                        print(f"    SNR={snr}dB: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                else:
                    model = model_fn()
                    if isinstance(model, STFusionNet):

                        model = train_stfusion_model(
                            model, train_loader_seq, val_loader_seq,
                            num_epochs=EPOCHS, save_path=save_path
                        )
                    else:

                        model = train_sequence_model(
                            model, train_loader_seq, val_loader_seq,
                            num_epochs=EPOCHS, save_path=save_path
                        )
                    acc, prec, rec, f1 = evaluate_model(model, test_loader_clean_seq)
                    metrics_clean[name]['acc'].append(acc)
                    metrics_clean[name]['prec'].append(prec)
                    metrics_clean[name]['rec'].append(rec)
                    metrics_clean[name]['f1'].append(f1)
                    print(f"    干净: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                    for snr in TEST_SNR_LIST:
                        acc, prec, rec, f1 = evaluate_model(model, test_loaders_noisy_seq[snr])
                        metrics_noise[snr][name]['acc'].append(acc)
                        metrics_noise[snr][name]['prec'].append(prec)
                        metrics_noise[snr][name]['rec'].append(rec)
                        metrics_noise[snr][name]['f1'].append(f1)
                        print(f"    SNR={snr}dB: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            except Exception as e:
                print(f"  失败: {e}")
                metrics_clean[name]['acc'].append(0.0)
                metrics_clean[name]['prec'].append(0.0)
                metrics_clean[name]['rec'].append(0.0)
                metrics_clean[name]['f1'].append(0.0)
                for snr in TEST_SNR_LIST:
                    metrics_noise[snr][name]['acc'].append(0.0)
                    metrics_noise[snr][name]['prec'].append(0.0)
                    metrics_noise[snr][name]['rec'].append(0.0)
                    metrics_noise[snr][name]['f1'].append(0.0)

    model_names = [name for name, _, _ in experiments]
    print("\n" + "=" * 100)
    print("最终结果汇总（平均值 ± 标准差）")
    print("=" * 100)

    print("\n--- 干净测试 (无噪声) ---")
    print(f"{'方法':<30} {'准确率':<18} {'精确率':<18} {'召回率':<18} {'F1分数':<18}")
    for name in model_names:
        acc_mean = np.mean(metrics_clean[name]['acc']); acc_std = np.std(metrics_clean[name]['acc'])
        prec_mean = np.mean(metrics_clean[name]['prec']); prec_std = np.std(metrics_clean[name]['prec'])
        rec_mean = np.mean(metrics_clean[name]['rec']); rec_std = np.std(metrics_clean[name]['rec'])
        f1_mean = np.mean(metrics_clean[name]['f1']); f1_std = np.std(metrics_clean[name]['f1'])
        display_name = name.replace('B1_', 'Baseline1_').replace('B2_', 'Baseline2_') \
                           .replace('B3_', 'Baseline3_').replace('B4_', 'Baseline4_') \
                           .replace('B5_', 'Baseline5_')
        print(f"{display_name:<30} {acc_mean:.4f}±{acc_std:.4f}   {prec_mean:.4f}±{prec_std:.4f}   {rec_mean:.4f}±{rec_std:.4f}   {f1_mean:.4f}±{f1_std:.4f}")

    for snr in TEST_SNR_LIST:
        print(f"\n--- 噪声测试 (SNR={snr}dB) ---")
        print(f"{'方法':<30} {'准确率':<18} {'精确率':<18} {'召回率':<18} {'F1分数':<18}")
        for name in model_names:
            acc_mean = np.mean(metrics_noise[snr][name]['acc']); acc_std = np.std(metrics_noise[snr][name]['acc'])
            prec_mean = np.mean(metrics_noise[snr][name]['prec']); prec_std = np.std(metrics_noise[snr][name]['prec'])
            rec_mean = np.mean(metrics_noise[snr][name]['rec']); rec_std = np.std(metrics_noise[snr][name]['rec'])
            f1_mean = np.mean(metrics_noise[snr][name]['f1']); f1_std = np.std(metrics_noise[snr][name]['f1'])
            display_name = name.replace('B1_', 'Baseline1_').replace('B2_', 'Baseline2_') \
                               .replace('B3_', 'Baseline3_').replace('B4_', 'Baseline4_') \
                               .replace('B5_', 'Baseline5_')
            print(f"{display_name:<30} {acc_mean:.4f}±{acc_std:.4f}   {prec_mean:.4f}±{prec_std:.4f}   {rec_mean:.4f}±{rec_std:.4f}   {f1_mean:.4f}±{f1_std:.4f}")

    clean_save = {}
    for name in model_names:
        clean_save[f'{name}_acc'] = np.array(metrics_clean[name]['acc'])
        clean_save[f'{name}_prec'] = np.array(metrics_clean[name]['prec'])
        clean_save[f'{name}_rec'] = np.array(metrics_clean[name]['rec'])
        clean_save[f'{name}_f1'] = np.array(metrics_clean[name]['f1'])
    np.savez(os.path.join(run_dir, '对比实验结果_clean_allmetrics_noiseaug_weighted.npz'), **clean_save)

    for snr in TEST_SNR_LIST:
        snr_save = {}
        for name in model_names:
            snr_save[f'{name}_acc'] = np.array(metrics_noise[snr][name]['acc'])
            snr_save[f'{name}_prec'] = np.array(metrics_noise[snr][name]['prec'])
            snr_save[f'{name}_rec'] = np.array(metrics_noise[snr][name]['rec'])
            snr_save[f'{name}_f1'] = np.array(metrics_noise[snr][name]['f1'])
        np.savez(os.path.join(run_dir, f'对比实验结果_noise{snr}dB_allmetrics_noiseaug_weighted.npz'), **snr_save)

    with open(os.path.join(run_dir, '对比结果_详细指标_noiseaug_weighted22.txt'), 'w') as f:
        f.write("Clean vs Noise (multiple SNRs) - All Metrics (Mean ± Std)\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Method':<25}")
        for phase in ['Clean'] + [f'Noise_{s}dB' for s in TEST_SNR_LIST]:
            f.write(f" {phase:^12}")
        f.write("\n")
        for metric in ['acc', 'prec', 'rec', 'f1']:
            f.write(f"\n--- {metric.upper()} ---\n")
            f.write(f"{'Method':<25}")
            for _ in ['Clean'] + [f'Noise_{s}dB' for s in TEST_SNR_LIST]:
                f.write(f" {'Mean±Std':^12}")
            f.write("\n")
            for name in model_names:
                clean_mean = np.mean(metrics_clean[name][metric]); clean_std = np.std(metrics_clean[name][metric])
                line = f"{name:<25} {clean_mean:.4f}±{clean_std:.4f}"
                for snr in TEST_SNR_LIST:
                    noise_mean = np.mean(metrics_noise[snr][name][metric]); noise_std = np.std(metrics_noise[snr][name][metric])
                    line += f"    {noise_mean:.4f}±{noise_std:.4f}"
                f.write(line + "\n")

    print("\n所有详细结果已保存至：")
    print(f"  - 本次运行目录: {run_dir}")
    print(f"  - 最佳模型目录: {MODEL_SAVE_DIR}")
    print("  - 对比实验结果_clean_allmetrics_noiseaug_weighted.npz")
    for snr in TEST_SNR_LIST:
        print(f"  - 对比实验结果_noise{snr}dB_allmetrics_noiseaug_weighted.npz")
    print("  - 对比结果_详细指标_noiseaug_weighted.txt")
    print(f"  - 最佳模型已保存在 '{MODEL_SAVE_DIR}' 目录下")

if __name__ == "__main__":
    run_all_experiments()
