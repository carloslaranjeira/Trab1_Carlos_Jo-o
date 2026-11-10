"""Comparação com o ground truth (só para avaliação, nunca para estimar).

- Registo: erro de translação (mm) e de rotação (graus) de cada câmara;
- Segmentação: n.º de caixas encontradas, precisão e recall de cada conjunto de pontos;
- Caixas: erro de posição (mm), de yaw (graus), das dimensões (mm) e IoU 3D,
  tendo em conta as simetrias da caixa.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_T_table, rotation_angle_deg, transform_points, wrap_angle, yaw_of


# --------------------------------------------------------------------------------------
# Registo
# --------------------------------------------------------------------------------------
def camera_errors(T_est_list, ground_truth):
    """Lista de (imagem, erro de translação em mm, erro de rotação em graus)."""
    rows = []
    for T_est, cam in zip(T_est_list, ground_truth["cameras"]):
        T_gt = np.array(cam["T_camera0_camera"])
        t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000
        r_err = rotation_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3])   # ângulo de R_est^T R_gt
        rows.append((cam["image"], t_err, r_err))
    return rows


# --------------------------------------------------------------------------------------
# Caixas
# --------------------------------------------------------------------------------------
def boxes_to_camera0(params, T_camera0_table):
    """Converte parâmetros (referencial da mesa) em caixas no formato do result.json."""
    return [{"T_camera0_box": T_camera0_table @ box_T_table(p[0], p[1], p[2], p[5]),
             "length": p[3], "width": p[4], "height": p[5]} for p in params]


def _box_in_world(T_world_box, length, width, height):
    """Descrição de uma caixa no referencial do mundo (z para cima): centro, yaw e dimensões."""
    return {"center": T_world_box[:3, 3], "yaw": yaw_of(T_world_box[:3, :3]),
            "dims": np.array([length, width, height])}


def iou_3d(a, b):
    """IoU 3D de duas caixas com z para cima.

    Volume da interseção = área da interseção dos retângulos vistos de cima
                           x sobreposição dos intervalos de altura.
    """
    r1 = (tuple(a["center"][:2]), tuple(a["dims"][:2]), np.degrees(a["yaw"]))
    r2 = (tuple(b["center"][:2]), tuple(b["dims"][:2]), np.degrees(b["yaw"]))
    status, region = cv2.rotatedRectangleIntersection(r1, r2)
    area = 0.0
    if status != cv2.INTERSECT_NONE and region is not None and len(region) >= 3:
        area = cv2.contourArea(cv2.convexHull(region.reshape(-1, 2).astype(np.float32)))
    z1 = (a["center"][2] - a["dims"][2] / 2, a["center"][2] + a["dims"][2] / 2)
    z2 = (b["center"][2] - b["dims"][2] / 2, b["center"][2] + b["dims"][2] / 2)
    dz = max(0.0, min(z1[1], z2[1]) - max(z1[0], z2[0]))
    inter = area * dz
    union = np.prod(a["dims"]) + np.prod(b["dims"]) - inter
    return inter / union if union > 0 else 0.0


def box_errors(est, gt):
    """Erros de uma caixa estimada em relação à verdadeira, com as simetrias.

    Versões equivalentes da caixa verdadeira: rodar k*90° em torno de z (k = 0..3), trocando
    length com width quando k é ímpar. Ficamos com a versão de menor erro de yaw.
    """
    pos_err = np.linalg.norm(est["center"] - gt["center"]) * 1000
    best = None
    for k in range(4):
        yaw_err = abs(np.degrees(wrap_angle(est["yaw"] - gt["yaw"] - k * np.pi / 2)))
        dims_gt = gt["dims"][[1, 0, 2]] if k % 2 else gt["dims"]
        dim_err = (est["dims"] - dims_gt) * 1000
        score = yaw_err + np.abs(dim_err).sum() * 0.01      # yaw primeiro; dimensões desempatam (caixas quadradas)
        if best is None or score < best[0]:
            best = (score, yaw_err, dim_err)
    _, yaw_err, dim_err = best
    return {"pos_mm": pos_err, "yaw_deg": yaw_err, "dl_mm": dim_err[0], "dw_mm": dim_err[1],
            "dh_mm": dim_err[2], "iou": iou_3d(est, gt)}


def evaluate_boxes(est_boxes, ground_truth):
    """Associa cada caixa estimada a uma verdadeira (máximo IoU, algoritmo húngaro) e calcula os erros.

    Para comparar no referencial do mundo (onde z é exatamente perpendicular à mesa), as caixas
    estimadas são levadas para o mundo com a pose VERDADEIRA da câmara 0 (só na avaliação).
    Caixas verdadeiras sem correspondência contam com IoU = 0.
    """
    T_world_camera0 = np.array(ground_truth["cameras"][0]["T_world_camera"])
    gt = [_box_in_world(np.array(b["T_world_box"]), b["length"], b["width"], b["height"])
          for b in ground_truth["boxes"]]
    est = [_box_in_world(T_world_camera0 @ b["T_camera0_box"], b["length"], b["width"], b["height"])
           for b in est_boxes]

    iou = np.array([[iou_3d(e, g) for g in gt] for e in est]).reshape(len(est), len(gt))
    rows_est, cols_gt = linear_sum_assignment(-iou) if len(est) else ([], [])
    results = []
    matched = {}
    for i, j in zip(rows_est, cols_gt):
        matched[j] = box_errors(est[i], gt[j])
    for j in range(len(gt)):
        results.append(matched.get(j, {"pos_mm": np.nan, "yaw_deg": np.nan, "dl_mm": np.nan,
                                       "dw_mm": np.nan, "dh_mm": np.nan, "iou": 0.0}))
    return results


# --------------------------------------------------------------------------------------
# Segmentação
# --------------------------------------------------------------------------------------
def inside_gt_boxes(points_camera0, ground_truth, margin=0.005):
    """Para cada caixa verdadeira, máscara dos pontos (referencial da câmara 0) que estão dentro dela
    (com uma margem de 5 mm por causa do ruído)."""
    masks = []
    for b in ground_truth["boxes"]:
        T_box_camera0 = np.linalg.inv(np.array(b["T_camera0_box"]))
        local = transform_points(T_box_camera0, points_camera0)
        half = np.array([b["length"], b["width"], b["height"]]) / 2 + margin
        masks.append(np.all(np.abs(local) <= half, axis=1))
    return masks


def evaluate_segmentation(points_table, labels, T_camera0_table, all_points_camera0, ground_truth, voxel=0.004):
    """Precisão e recall dos conjuntos de pontos de cada caixa.

    points_table, labels: pontos segmentados (referencial da mesa) e caixa atribuída a cada um.
    all_points_camera0: TODOS os pontos da nuvem agregada (para saber quantos pontos tem cada caixa
                        verdadeira, incluindo os que a segmentação deitou fora).
    As duas nuvens têm densidades diferentes (uma foi reduzida), por isso o recall é calculado
    com voxels: fração dos voxels ocupados pela caixa verdadeira que ficaram no conjunto certo.
    """
    pts_c0 = transform_points(T_camera0_table, points_table)
    gt_masks_seg = inside_gt_boxes(pts_c0, ground_truth)
    gt_masks_all = inside_gt_boxes(all_points_camera0, ground_truth)

    def voxel_keys(p):
        return set(map(tuple, np.floor(p / voxel).astype(np.int64)))

    num_sets = len(set(labels[labels >= 0]))
    # Associação conjunto <-> caixa verdadeira pelo maior número de pontos em comum
    overlap = np.array([[np.sum((labels == s) & m) for m in gt_masks_seg] for s in range(num_sets)])
    rows, cols = linear_sum_assignment(-overlap) if num_sets else ([], [])
    results = []
    for s, j in zip(rows, cols):
        mine = labels == s
        precision = np.sum(mine & gt_masks_seg[j]) / max(np.sum(mine), 1)
        gt_keys = voxel_keys(all_points_camera0[gt_masks_all[j]])
        my_keys = voxel_keys(pts_c0[mine & gt_masks_seg[j]])
        recall = len(gt_keys & my_keys) / max(len(gt_keys), 1)
        results.append({"gt_box": int(j), "precision": precision, "recall": recall})
    return num_sets, len(ground_truth["boxes"]), results
