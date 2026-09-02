**SAR Imaging and Recognition**
This repository contains the source code for ISAR imaging, motion compensation, and aircraft target recognition.

**ISAR imaging**
The ISAR imaging folder contains code for ISAR image generation based on t3D geometric aircraft models.

The main functions include:
ISAR echo generation based on 3D geometric target models;
High-resolution range profile generation;
Range centroid compensation;
Doppler centroid compensation;
Fine compensation based on minimum entropy method;
2D ISAR image formation.

These codes are mainly used to generate continuous multi-view ISAR image sequences for aircraft targets under different observation conditions.

**ISAR recognition**
The ISAR recognition folder contains code for classification and recognition of aircraft targets using ISAR images and ISAR image sequences.

The main functions include:
Classification based on single frame spatial features;
Classification based on decision‑level average fusion；
Classification based on sequence modeling and multi-view aggregation approaches；
Classification based on STFusionNet；
Noise-Augmented Training.

The proposed STFusionNet combines spatial feature enhancement, temporal evolution modeling, and adaptive sequence-level feature aggregation 
for aircraft target recognition from continuous multi-view ISAR image sequences.

**Requirements**：
The main implementation is based on Python, PyTorch, MATLAB, and commonly used scientific computing libraries.

**Citation**：
If this code is useful for your research, please consider citing the corresponding paper.
