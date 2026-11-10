"""Parte 3 (método base): bounding box orientada do Open3D.

Serve só de referência para comparar com o ajuste por otimização.

Para cada conjunto de pontos (um por caixa, vindo da segmentação):
    1. Remove outliers (estatístico + maior componente ligada), para tratar o método de forma justa;
    2. Calcula a bounding box orientada com get_minimal_oriented_bounding_box() (menor volume)
       ou get_oriented_bounding_box() (PCA do invólucro convexo);
    3. Converte para a convenção do trabalho: o eixo z da caixa tem de ser perpendicular à mesa.
       Escolhemos, dos 3 eixos da bounding box, o mais alinhado com o z da mesa como 'altura';
       os outros dois dão o yaw e as dimensões no plano;
    4. Como sabemos que a caixa assenta na mesa, a base é posta em z=0 e a altura vai até ao topo
       da bounding box (sem isto, a caixa perdia os ~8 mm cortados junto à mesa na segmentação).
"""

import numpy as np
import open3d as o3d

from .fitting import largest_component, normalize_box


def bbox_box(points, minimal=True):
    """Bounding box orientada de um conjunto de pontos (referencial da mesa) -> [x, y, yaw, l, w, h]."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pts = np.asarray(pc.points)
    pts = pts[largest_component(pts)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    obb = pc.get_minimal_oriented_bounding_box() if minimal else pc.get_oriented_bounding_box()
    R, extent, center = np.asarray(obb.R), np.asarray(obb.extent), np.asarray(obb.center)

    # Eixo da bounding box mais alinhado com o z da mesa -> altura
    k = int(np.argmax(np.abs(R[2, :])))
    others = [i for i in range(3) if i != k]
    axis_x = R[:, others[0]]
    yaw = np.arctan2(axis_x[1], axis_x[0])
    length, width = extent[others[0]], extent[others[1]]
    top = center[2] + extent[k] / 2          # topo da caixa
    return normalize_box(np.array([center[0], center[1], yaw, length, width, top]))


def bbox_method(seg_points, box_labels, num_boxes, minimal=True):
    """Aplica a bounding box a cada conjunto de pontos (box_labels[i] = caixa do ponto i)."""
    boxes = []
    for j in range(num_boxes):
        pts = seg_points[box_labels == j]
        if len(pts) < 10:
            continue
        boxes.append(bbox_box(pts, minimal))
    return np.array(boxes)
