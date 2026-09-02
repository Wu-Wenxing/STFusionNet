import numpy as np
import trimesh

# 计算RCS。考虑到后面要计算的俯仰角、方位角不总是结构化的，可以将俯仰角和方位角改成相同数量的参数表示。
def generator(
        vertices,
        faces,
        frequencies,
        zeniths,
        azimuths,
        最大反射次数
):
    # np.seterr(divide='ignore')
    k = 2 * np.pi / .3 * frequencies
    jk = 1j * k
    eps = 1e-6
    mesh_trimesh = trimesh.Trimesh(vertices, faces)
    normals = mesh_trimesh.face_normals
    # plotter=pv.Plotter()
    # faces_pv=np.column_stack([np.full(faces.shape[0],faces.shape[1]),faces])
    # mesh=pv.PolyData(vertices,faces_pv)
    # plotter.add_mesh(mesh,show_edges=True,opacity=.1)
    # plotter.add_point_labels(vertices,np.arange(vertices.shape[0]))
    # centers=mesh.cell_centers().points
    # plotter.add_point_labels(centers,np.arange(centers.shape[0]),text_color='r')
    面元序号=np.arange(faces.shape[0])
    边方向标记=np.ones(faces.shape[0],dtype=int)
    边01=np.column_stack([faces[:,[0,1]],面元序号,边方向标记])
    边12=np.column_stack([faces[:,[1,2]],面元序号,边方向标记])
    边20=np.column_stack([faces[:,[2,0]],面元序号,边方向标记])
    边=np.vstack([边01,边12,边20])
    需要换序=边[:,0]>边[:,1]
    边[np.ix_(需要换序,[0,1])]=边[np.ix_(需要换序,[1,0])]
    边[需要换序,-1]=-边[需要换序,-1]
    排序=np.lexsort((边[:,1],边[:,0]))
    边=边[排序]
    重复=(边[:-1,0]==边[1:,0])&(边[:-1,1]==边[1:,1])
    重复=np.hstack([重复,False])
    重复下一个=np.roll(重复,1)
    边=np.column_stack([边[重复],边[np.ix_(重复下一个,[2,3])]])
    顶点=vertices[边[:,[0,1]]]
    边缘向量=顶点[:,1]-顶点[:,0]
    边缘中心=(顶点[:,1]+顶点[:,0])/2
    # plotter.add_points(边缘中心,color='r')
    边缘长度=np.linalg.norm(边缘向量,axis=-1)
    边缘方向=边缘向量/边缘长度[:,np.newaxis]
    # plotter.add_points(边缘中心,color='r')
    # plotter.add_arrows(边缘中心,边缘方向,mag=.1,color='g')
    # plotter.add_arrows(centers,normals,mag=.1,color='r')
    法向0=normals[边[:,2]]*边[:,3][:,np.newaxis]
    法向1=normals[边[:,4]]*边[:,5][:,np.newaxis]
    法向叉积点边缘方向=np.sum(np.cross(法向0,法向1)*边缘方向,axis=-1)
    # 凸边缘=法向叉积点边缘方向<-eps
    # 凹边缘=法向叉积点边缘方向>eps
    有效边缘=np.abs(法向叉积点边缘方向)>eps*1000
    边=边[有效边缘]
    边缘向量=边缘向量[有效边缘]
    边缘中心=边缘中心[有效边缘]
    边缘长度=边缘长度[有效边缘]
    边缘方向=边缘方向[有效边缘]
    法向0=法向0[有效边缘]
    法向1=法向1[有效边缘]
    法向叉积点边缘方向=法向叉积点边缘方向[有效边缘]

    边另一面=边.copy()
    边缘向量另一面=边缘向量.copy()
    边缘中心另一面=边缘中心.copy()
    边缘长度另一面=边缘长度.copy()
    边缘方向另一面=边缘方向.copy()
    法向0另一面=法向0.copy()
    法向1另一面=法向1.copy()
    法向叉积点边缘方向另一面=法向叉积点边缘方向.copy()

    边另一面[:,[0,1]]=边另一面[:,[1,0]]
    边另一面[:,[3,5]]=-边另一面[:,[3,5]]
    边缘向量另一面=-边缘向量另一面
    边缘方向另一面=-边缘方向另一面
    法向0另一面=-法向0另一面
    法向1另一面=-法向1另一面
    法向叉积点边缘方向另一面=-法向叉积点边缘方向另一面

    边=np.vstack([边,边另一面])
    边缘向量=np.vstack([边缘向量,边缘向量另一面])
    边缘中心=np.vstack([边缘中心,边缘中心另一面])
    边缘长度=np.hstack([边缘长度,边缘长度另一面])
    边缘方向=np.vstack([边缘方向,边缘方向另一面])
    法向0=np.vstack([法向0,法向0另一面])
    法向1=np.vstack([法向1,法向1另一面])
    法向叉积点边缘方向=np.hstack([法向叉积点边缘方向,法向叉积点边缘方向另一面])

    法向0点法向1=np.sum(法向0*法向1,axis=-1)
    边缘劈角=np.arctan2(法向叉积点边缘方向,法向0点法向1)%(2*np.pi)
    边缘n=边缘劈角/np.pi
    # 边缘X=-1/边缘n/np.tan(np.pi/(2*边缘n))
    # 用来后面计算射线与边缘的夹角
    劈面方向=np.cross(边缘方向,法向0)
    # plotter.add_axes()
    # plotter.add_point_labels(边缘中心,边,text_color='b')
    # plotter.show(jupyter_backend='trame')

    亮面元标记=np.zeros(faces.shape[0],dtype=bool)
    ni标记=np.zeros(faces.shape[0])
    for zenith下标, zenith in enumerate(zeniths):
        coszenith = np.cos(zenith * np.pi / 180)
        sinzenith = np.sin(zenith * np.pi / 180)
        for azimuth下标, azimuth in enumerate(azimuths):
            cosazimuth = np.cos(azimuth * np.pi / 180)
            sinazimuth = np.sin(azimuth * np.pi / 180)
            # print(f'当前入射角{zenith=}°,{azimuth=}°,')
            rcshh = np.zeros_like(frequencies, dtype=complex)
            rcsvv = np.zeros_like(frequencies, dtype=complex)
            rcs_po_hh各次反射贡献=np.zeros((最大反射次数,frequencies.shape[0]),dtype=complex)
            rcs_po_vv各次反射贡献=np.zeros((最大反射次数,frequencies.shape[0]),dtype=complex)
            er = np.array([sinzenith * cosazimuth, sinzenith * sinazimuth, coszenith])
            ezenith = np.array([coszenith * cosazimuth, coszenith * sinazimuth, -sinzenith])
            eazimuth = np.array([-sinazimuth, cosazimuth, 0])
            hh入射e = eazimuth
            hh入射h = ezenith
            vv入射e = ezenith
            vv入射h = -eazimuth
            hh散射e = eazimuth
            hh散射h = -ezenith
            vv散射e = ezenith
            vv散射h = eazimuth
            散射方向 = er
            r = vertices[faces]
            rc=np.mean(r,axis=1)
            ezenith投影 = rc @ ezenith
            zenith方向 = ezenith投影[:, np.newaxis] * ezenith
            eazimuth投影 = rc @ eazimuth
            azimuth方向 = eazimuth投影[:, np.newaxis] * eazimuth
            # 确保射线都在以原点为中心的模型包围球之外，否则有可能有射线起点从模型中间开始。
            模型半径=np.max(np.linalg.norm(vertices,axis=-1))
            射线起点 = zenith方向 + azimuth方向 + 模型半径*1.01 * er
            入射方向 = np.repeat(-er[np.newaxis, :], 射线起点.shape[0], axis=0)
            locations, index_ray, index_tri = mesh_trimesh.ray.intersects_location(射线起点, 入射方向, multiple_hits=False)
            # 去除平行入射的射线
            n = normals[index_tri]
            ni = -n@er
            照到的面元 = (index_ray == index_tri) & (np.abs(ni) > eps * 100)
            # print('1次反射射线数量', 照到的面元.sum())
            index_tri = index_tri[照到的面元]
            index_ray = index_ray[照到的面元]
            n = n[照到的面元]
            ni = ni[照到的面元]
            入射方向 = 入射方向[index_ray]
            r = r[index_tri]
            rc = rc[index_tri]
            波程因子 = np.ones((frequencies.shape[0],index_tri.shape[0]), dtype=complex)
            # 下面计算边缘的绕射，只计算第一次照射的
            亮面元标记[:]=False
            亮面元标记[index_tri]=True
            ni标记[:]=0
            ni标记[index_tri]=ni
            # print(f'{亮面元标记=}')
            # print(f'{[边[:,2]]=}')
            # print(f'{[边[:,3]]=}')
            # print(f'{ni=}')
            # print(f'{亮面元标记[边[:,2]]=}')
            # print(f'{ni标记=}')
            照到劈面0=亮面元标记[边[:,2]]|(ni标记[边[:,2]]*边[:,3]<-eps*100)
            照到劈面1=亮面元标记[边[:,4]]|(ni标记[边[:,4]]*边[:,5]>eps*100)
            照到劈面=照到劈面0|照到劈面1
            当前边缘向量=边缘向量[照到劈面]
            当前边缘中心=边缘中心[照到劈面]
            当前边缘长度=边缘长度[照到劈面]
            当前边缘方向=边缘方向[照到劈面]
            当前法向0=法向0[照到劈面]
            当前边缘劈角=边缘劈角[照到劈面]
            当前边缘n=边缘n[照到劈面]
            当前劈面方向=劈面方向[照到劈面]
            sin=当前劈面方向@散射方向
            cos=当前法向0@散射方向
            入射角0=np.pi-np.arctan2(sin,cos)
            入射角1=当前边缘劈角-入射角0
            # 两个入射角都要大于0才有效，目前只能处理外劈角
            有效照射=(入射角1>eps*100)&(当前边缘n>1)
            入射角0=入射角0[有效照射]
            入射角1=入射角1[有效照射]
            当前边缘向量=当前边缘向量[有效照射]
            当前边缘中心=当前边缘中心[有效照射]
            当前边缘长度=当前边缘长度[有效照射]
            当前边缘方向=当前边缘方向[有效照射]
            当前边缘n=当前边缘n[有效照射]

            # 计算绕射，用这个w比较方便
            w = 2*散射方向
            # print(f'{当前边缘向量=}')
            # print(f'{w=}')
            入射方向劈边分量=当前边缘方向@散射方向
            hh入射e劈边分量=当前边缘方向@hh入射e
            hh入射h劈边分量=当前边缘方向@hh入射h
            vv入射e劈边分量=当前边缘方向@vv入射e
            vv入射h劈边分量=当前边缘方向@vv入射h
            sinc=np.sinc(k[:,np.newaxis]*(当前边缘向量@w)/2/np.pi)
            exp=np.exp(jk[:,np.newaxis]*(当前边缘中心@w))
            共同因子=exp*当前边缘长度*sinc/(1-入射方向劈边分量**2)/np.sqrt(np.pi)

            cosbeta=入射方向劈边分量
            sinbeta=np.sqrt(1-cosbeta**2)
            cotbeta=cosbeta/sinbeta

            phi=入射角0
            cosphi=np.cos(phi)
            sinphi=np.sin(phi)
            cosalpha=cosphi-2*cotbeta**2
            # 可能是复数，所以要确保是复数
            alpha=np.arccos(cosalpha+0j)
            sinalpha=np.sin(alpha)
            X0=-1/(2*当前边缘n)/np.tan((np.pi-(alpha-phi))/(2*当前边缘n))
            Y0=-1/(2*当前边缘n)/np.tan((np.pi-(alpha+phi))/(2*当前边缘n))
            X0po=-1/2/np.tan((np.pi-(alpha-phi))/2)*np.heaviside(np.pi-phi,0)
            Y0po=-1/2/np.tan((np.pi-(alpha+phi))/2)*np.heaviside(np.pi-phi,0)
            X0f=X0-X0po
            入射阴影=np.abs(np.pi-(alpha-phi))<1e-4
            X0f[入射阴影]=0
            Y0f=Y0-Y0po
            反射阴影=np.abs(np.pi-(alpha+phi))<1e-4
            Y0f[反射阴影]=0
            De0=X0f-Y0f
            Dm0=sinphi/sinalpha*(X0f+Y0f)
            sinbetaDem0=-2*(cosphi-cotbeta**2)*cosbeta/sinalpha*(X0f+Y0f)

            phi=入射角1
            cosphi=np.cos(phi)
            sinphi=np.sin(phi)
            cosalpha=cosphi-2*cotbeta**2
            # 可能是复数，所以要确保是复数
            alpha=np.arccos(cosalpha+0j)
            sinalpha=np.sin(alpha)
            X1=-1/(2*当前边缘n)/np.tan((np.pi-(alpha-phi))/(2*当前边缘n))
            Y1=-1/(2*当前边缘n)/np.tan((np.pi-(alpha+phi))/(2*当前边缘n))
            X1po=-1/2/np.tan((np.pi-(alpha-phi))/2)*np.heaviside(np.pi-phi,0)
            Y1po=-1/2/np.tan((np.pi-(alpha+phi))/2)*np.heaviside(np.pi-phi,0)
            X1f=X1-X1po
            入射阴影=np.abs(np.pi-(alpha-phi))<1e-4
            X1f[入射阴影]=0
            Y1f=Y1-Y1po
            反射阴影=np.abs(np.pi-(alpha+phi))<1e-4
            Y1f[反射阴影]=0
            De1=X1f-Y1f
            Dm1=sinphi/sinalpha*(X1f+Y1f)
            sinbetaDem1=-2*(cosphi-cotbeta**2)*cosbeta/sinalpha*(X1f+Y1f)

            De=De0+De1
            Dm=Dm0+Dm1
            sinbetaDem=sinbetaDem0-sinbetaDem1

            # hh贡献=hh入射e劈边分量**2*De-hh入射h劈边分量**2*Dm+hh入射e劈边分量*hh入射h劈边分量*sinbetaDem
            # vv贡献=vv入射e劈边分量**2*De-vv入射h劈边分量**2*Dm+vv入射e劈边分量*vv入射h劈边分量*sinbetaDem
            hh贡献Dem=hh入射e劈边分量*hh入射h劈边分量*sinbetaDem
            vv贡献Dem=vv入射e劈边分量*vv入射h劈边分量*sinbetaDem
            hh贡献Dm=-hh入射h劈边分量**2*Dm
            vv贡献Dm=-vv入射h劈边分量**2*Dm
            hh贡献De=hh入射e劈边分量**2*De
            vv贡献De=vv入射e劈边分量**2*De
            hh贡献=hh贡献De+hh贡献Dm+hh贡献Dem
            vv贡献=vv贡献De+vv贡献Dm+vv贡献Dem
            rcs_ptd_hh=np.sum(共同因子*hh贡献,axis=-1)
            rcs_ptd_vv=np.sum(共同因子*vv贡献,axis=-1)
            rcshh += rcs_ptd_hh
            rcsvv += rcs_ptd_vv
            for 反射次数 in range(最大反射次数):
                w = 散射方向 - 入射方向
                需要反向 = ni > 0
                n[需要反向] = -n[需要反向]
                ni[需要反向] = -ni[需要反向]
                # print(需要反向.shape)
                r[np.ix_(需要反向,[1,2])]=r[np.ix_(需要反向,[2,1])]
                # 这些可以优化，用3维数组，紧凑代码
                l=np.roll(r,-1,axis=1)-r
                rcm = (np.roll(r,-1,axis=1)+r)/2 - rc[:,np.newaxis,:]
                nw = np.cross(n, w)
                nw2 = np.sum(nw * nw, axis=-1)
                w_rc = np.sum(w * rc, axis=-1)
                hh反射e = -hh入射e + 2 * np.sum(n * hh入射e, axis=-1, keepdims=True) * n
                hh反射h = hh入射h - 2 * np.sum(n * hh入射h, axis=-1, keepdims=True) * n
                vv反射e = -vv入射e + 2 * np.sum(n * vv入射e, axis=-1, keepdims=True) * n
                vv反射h = vv入射h - 2 * np.sum(n * vv入射h, axis=-1, keepdims=True) * n
                hh_n点es叉hi = np.sum(n * np.cross(hh散射e, hh入射h), axis=-1)
                vv_n点es叉hi = np.sum(n * np.cross(vv散射e, vv入射h), axis=-1)
                相位因子 = np.exp(jk[:,np.newaxis] * w_rc)
                公共因子hh = jk[:,np.newaxis] * hh_n点es叉hi * 波程因子 * 相位因子 / np.sqrt(np.pi)
                公共因子vv = jk[:,np.newaxis] * vv_n点es叉hi * 波程因子 * 相位因子 / np.sqrt(np.pi)
                积分 = np.zeros((frequencies.shape[0], n.shape[0]), dtype=complex)
                垂直照射 = nw2 < eps
                倾斜照射 = ~垂直照射
                if np.any(垂直照射):
                    面积 = np.linalg.norm(np.cross(l[垂直照射,0], l[垂直照射,1]), axis=-1) / 2
                    积分[:, 垂直照射] = np.repeat(面积[np.newaxis, :], frequencies.shape[0], axis=0)
                if np.any(倾斜照射):
                    l = l[倾斜照射]
                    rcm = rcm[倾斜照射]
                    w = w[倾斜照射]
                    nw = nw[倾斜照射]
                    nw2 = nw2[倾斜照射]
                    w_l = np.sum(w[:,np.newaxis,:] * l, axis=-1)
                    w_rcm = np.sum(w[:,np.newaxis,:] * rcm, axis=-1)
                    临m = np.exp(jk[:,np.newaxis,np.newaxis] * w_rcm) * np.sinc(k[:,np.newaxis,np.newaxis] * w_l / 2 / np.pi)
                    临 = np.sum(临m[:, :, :, np.newaxis] * l,axis=-2)
                    积分[:, 倾斜照射] = np.sum(nw * 临, axis=-1) / nw2 / jk[:,np.newaxis]
                rcs_po_hh = np.sum(公共因子hh * 积分, axis=-1)
                rcs_po_vv = np.sum(公共因子vv * 积分, axis=-1)
                rcs_po_hh各次反射贡献[反射次数,:]=rcs_po_hh
                rcs_po_vv各次反射贡献[反射次数,:]=rcs_po_vv
                rcshh += rcs_po_hh
                rcsvv += rcs_po_vv
                # print(f'各频率{反射次数 + 1}次反射RCS贡献[dBsm],{frequencies}GHz,vv:', 20 * np.log10(np.abs(rcs_po_vv)))
                # 如果到了最后一次，则不进行后面的射线判断了
                if 反射次数 == 最大反射次数 - 1:
                    break
                w = -2 * ni[:, np.newaxis] * n
                入射方向 = 入射方向 + w
                # hh入射e = -hh入射e + 2 * np.sum(n * hh入射e, axis=-1, keepdims=True) * n
                # hh入射h = hh入射h - 2 * np.sum(n * hh入射h, axis=-1, keepdims=True) * n
                # vv入射e = -vv入射e + 2 * np.sum(n * vv入射e, axis=-1, keepdims=True) * n
                # vv入射h = vv入射h - 2 * np.sum(n * vv入射h, axis=-1, keepdims=True) * n
                # 起始点稍微前移一点，防止射线照射到出发端点
                # 另一个办法，新建一个把这些相交的面去掉的trimesh，就不需要用下面的方法了。下面的方法不够精巧，而且可能会有问题。
                射线起点 = rc + 入射方向 * 0.001
                locations, index_ray, index_tri = mesh_trimesh.ray.intersects_location(射线起点, 入射方向, multiple_hits=False)
                # 去除平行入射的射线
                n = normals[index_tri]
                ni = np.sum(n * 入射方向[index_ray], axis=-1)
                照到的面元 = np.abs(ni) > eps * 100
                # print(f'{反射次数 + 2}次反射射线数量', 照到的面元.sum())
                if 照到的面元.shape[0] == 0:
                    break
                index_ray = index_ray[照到的面元]
                n = n[照到的面元]
                ni = ni[照到的面元]
                入射方向 = 入射方向[index_ray]
                hh入射e = hh反射e[index_ray]
                hh入射h = hh反射h[index_ray]
                vv入射e = vv反射e[index_ray]
                vv入射h = vv反射h[index_ray]
                # hh入射e = hh入射e[index_ray]
                # hh入射h = hh入射h[index_ray]
                # vv入射e = vv入射e[index_ray]
                # vv入射h = vv入射h[index_ray]
                波程因子 = 波程因子[:, index_ray]
                # 面元方向交换，使外法向向外
                r = r[np.ix_(index_ray,[0,2,1])]
                w_rc = np.sum(w[index_ray] * rc[index_ray], axis=-1)
                波程因子 *= np.exp(jk[:,np.newaxis] * w_rc)
                rc = locations[照到的面元]
                r += np.sum(n[:,np.newaxis,:] * (rc[:,np.newaxis,:] - r), axis=-1, keepdims=True) / ni[:, np.newaxis, np.newaxis] * 入射方向[:,np.newaxis,:]
            # return zenith下标,azimuth下标,zenith, azimuth, rcshh, rcsvv,rcs_po_hh各次反射贡献,rcs_po_vv各次反射贡献,rcs_ptd_hh,rcs_ptd_vv
            yield zenith下标,azimuth下标,zenith, azimuth, rcshh, rcsvv,rcs_po_hh各次反射贡献,rcs_po_vv各次反射贡献,rcs_ptd_hh,rcs_ptd_vv