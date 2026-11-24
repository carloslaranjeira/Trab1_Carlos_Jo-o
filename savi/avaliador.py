#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. AVALIADOR: número de caixas e comparação do result.json com o ground truth.

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS das 4 partes)
    python savi/avaliador.py --scene test/cena_1                                    # result.json da cena
    python savi/avaliador.py --scene test/cena_1 --result results_test/cena_1/result.json
    python savi/avaliador.py --scene test/cena_1 --num_boxes 3 --visualize          # sem pergunta; janelas 3D

1. Pergunta no terminal o número REAL de caixas e diz se é o número de caixas do result.json. As 4 partes
   nunca recebem o número de caixas (a Parte 2 estima-o).
2. SÓ se a pasta da cena tiver ground_truth.json (as cenas do dia da apresentação não o têm): compara o
   result.json com o ground truth e escreve o relatório avaliacao.md ao lado do result.json, com os
   tópicos de "Comparação com o ground truth" do enunciado para cada parte, e as figuras em PNG.
Este é o ÚNICO ficheiro que lê o ground_truth.json: as partes nunca o leem.
O run_test.py chama-o no fim de cada cena (evaluate_scene).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO
=====================================================================================================
    1. Número de caixas    pergunta o número real e compara com o do result.json
    2. Parte 1 - registo   erro de translação (mm) e de rotação (graus) de cada câmara; nuvem estimada sobre
                           a verdadeira
    3. Parte 2 - segmentação  nuvens juntas com as poses verdadeiras; n.º de caixas, precisão e recall
    4. Parte 3 - caixas    erros de posição, altura, yaw e dimensões e IoU 3D dos dois métodos, com as
                           simetrias; caixas estimadas sobre as verdadeiras
    5. Extra - fotometria  s e b estimados de cada imagem contra os verdadeiros
    6. Relatório           avaliacao.md e figuras (PNG e janelas 3D)
    7. Programa principal  evaluate_scene() e main()
"""

import argparse
import json
from pathlib import Path

import cv2                                   # OpenCV: interseção de retângulos (IoU 3D)
import matplotlib.pyplot as plt              # figura PNG vista de cima
import numpy as np
from scipy.optimize import linear_sum_assignment   # emparelhar caixas / conjuntos com o ground truth
from scipy.spatial.transform import Rotation       # ângulo de uma rotação (Parte 08)

from comum import (  # geometria e leitura da cena (comuns a todas as partes)
    box_corners_2d, boxes_to_camera0, load_scene, OUTPUT_FOLDER, transform_points,
    wrap_angle, yaw_of,
)
from extra_fotometria import PHOTOMETRY_FILE
from parte1_registo import aggregate, load_registration, save_registration
from parte2_segmentacao import load_objects, run_segmentation
from parte3_ajuste import BOX_COLORS, load_fit, run_fit, top_view
from visualizacao import box_item, camera_view, close_view, colors_uint8, data_view, points_item, SceneWindows


GROUND_TRUTH_FILE = "ground_truth.json"
GT_FOLDER = "partes_gt"        # 3. Parte 2 e conjuntos de pontos com as poses verdadeiras (dentro da cena)
REPORT_FILE = "avaliacao.md"   # 6. relatório (ao lado do result.json)
FIGURE_NAME = "avaliador"      # 6. as figuras chamam-se avaliador_<vista>.png (ao lado do result.json)
INSIDE_MARGIN = 0.005          # 3. m: margem à volta das caixas verdadeiras (ruído), como no enunciado
FIGURE_VOXEL = 0.005           # 6. m: voxel das nuvens das figuras
EST_COLOR = (0.9, 0.1, 0.1)    # 2. nuvem com as poses estimadas: vermelho
GT_COLOR = (0.1, 0.7, 0.2)     # 2. nuvem com as poses verdadeiras: verde
GT_BOX_COLOR = (0.0, 0.0, 0.0)  # 4. caixas verdadeiras: preto
METHODS = (("otimizacao", "otimização (result.json)"), ("bounding_box", "bounding box (partes/ajuste.npz)"))


def load_ground_truth(scene_folder):
    """O ground_truth.json da cena, ou None se não existir. Só o avaliador o lê, e só para comparar."""
    path = Path(scene_folder) / GROUND_TRUTH_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# #####################################################################################################
# 1. NÚMERO DE CAIXAS
# #####################################################################################################
# O número real de caixas só é dado aqui, depois de as 4 partes acabarem: as partes não o usam (a
# Parte 2 estima-o), e o avaliador compara-o com o número de caixas do result.json.
def ask_num_boxes(name=None):
    """Pergunta no terminal o número REAL de caixas da cena (até ser um inteiro positivo)."""
    while True:
        answer = input(f"Número real de caixas{f' da cena {name!r}' if name else ''}: ").strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("  escreve um número inteiro positivo")


def box_count_verdict(found, real):
    """Frase que compara o número de caixas encontradas com o número real."""
    if found == real:
        return f"encontradas {found}, reais {real}: o número de caixas encontrado está certo"
    diff = f"{real - found} a menos" if found < real else f"{found - real} a mais"
    return f"encontradas {found}, reais {real}: o número de caixas encontrado está errado ({diff})"


# #####################################################################################################
# 2. PARTE 1 - REGISTO
# #####################################################################################################
# Como na Parte 08: com E = T_gt^-1 T_est, o erro de translação é a norma da translação de E e o erro de
# rotação é o ângulo da rotação de E, arccos((tr(R_E) - 1) / 2) (o mesmo ângulo de R_est^T R_gt, como
# no enunciado). A nuvem estimada (vermelho) é desenhada sobre a verdadeira (verde), com
# paint_uniform_color, como na Parte 07/08 e no draw_registration_result do tutorial do Open3D.
def transform_error(T_est, T_gt):
    """Erro de uma transformação estimada em relação à verdadeira: (translação em mm, rotação em graus).

    O ângulo é calculado com Rotation.from_matrix(R_E).magnitude(), a alternativa que a Parte 08 indica
    para confirmar o arccos((tr(R_E) - 1) / 2): dá o mesmo ângulo, mas o result.json guarda as matrizes
    com 6 casas decimais e, para erros de centésimas de grau, cos(theta) ~ 1 - 1e-7 fica abaixo desse
    arredondamento, e o arccos do traço dava 0° em vez de 0.037°. O from_matrix usa primeiro a matriz de
    rotação mais próxima, e não é afetado.
    """
    E = np.linalg.inv(T_gt) @ T_est
    return float(np.linalg.norm(E[:3, 3]) * 1000), float(np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude()))


def camera_errors(result, ground_truth):
    """Erro de cada câmara do result.json (T_camera0_camera) em relação à do ground truth.

    As câmaras são emparelhadas pelo nome da imagem. Saída: lista de {imagem, transl_mm, rot_deg}.
    """
    true = {c["image"]: np.array(c["T_camera0_camera"]) for c in ground_truth["cameras"]}
    rows = []
    for cam in result["cameras"]:
        if cam["image"] in true:
            t, r = transform_error(np.array(cam["T_camera0_camera"]), true[cam["image"]])
            rows.append({"imagem": cam["image"], "transl_mm": t, "rot_deg": r})
    return rows


def true_camera_poses(scene, ground_truth):
    """T_camera0_camera verdadeira de cada imagem da cena, pela ordem das imagens."""
    true = {c["image"]: np.array(c["T_camera0_camera"]) for c in ground_truth["cameras"]}
    return [true[v.name] for v in scene.views]


def registration_clouds(scene, result, ground_truth, registered=None):
    """A nuvem agregada com as poses do result.json e a nuvem agregada com as poses verdadeiras
    (referencial da câmara 000, voxel de 5 mm, só para as figuras)."""
    estimated = {c["image"]: np.array(c["T_camera0_camera"]) for c in result["cameras"]}
    T_est = [estimated[v.name] for v in scene.views]
    est, _ = aggregate(scene, T_est, registered)
    gt, _ = aggregate(scene, true_camera_poses(scene, ground_truth))
    return est.voxel_down_sample(FIGURE_VOXEL), gt.voxel_down_sample(FIGURE_VOXEL)


# #####################################################################################################
# 3. PARTE 2 - SEGMENTAÇÃO
# #####################################################################################################
# Para avaliar só a segmentação, sem os erros do registo (como pede o enunciado), as nuvens são juntadas
# com as poses VERDADEIRAS e as Partes 2 e 3 (que dá os conjuntos de pontos de cada caixa) correm sobre
# essa nuvem, numa pasta à parte (<cena>/partes_gt/: os resultados do sistema não são tocados). Os pontos
# de cada caixa verdadeira são os que ficam dentro dela, com uma margem de 5 mm (ruído).
# O que não está nas aulas: o teste "dentro da caixa" (pontos no referencial da caixa, como a nuvem no
# referencial da mesa na Parte 08, Ex. 4.3, e |p| <= dimensões / 2 + margem), o emparelhamento pelo
# algoritmo húngaro (scipy.optimize.linear_sum_assignment) e o recall calculado com voxels.
def segmentation_with_true_poses(scene, ground_truth, folder):
    """Corre as Partes 2 e 3 sobre a nuvem junta com as poses verdadeiras (em vez das do registo).

    O resultado da 'Parte 1' tem o mesmo formato que o de run_registration, mas com as poses do ground
    truth e todas as câmaras na nuvem. Grava tudo em folder (<cena>/partes_gt/).
    Saída: (r1, r2, r3) como os das partes.
    """
    T_true = true_camera_poses(scene, ground_truth)
    registered = [True] * len(T_true)
    exposures = []
    cloud, camera_index = aggregate(scene, T_true, registered, exposures=exposures)
    save_registration(folder, scene, T_true, T_true, registered)
    r1 = {"scene": scene, "scene_folder": scene.folder, "images": [v.name for v in scene.views],
          "T_cams": T_true, "T_chain": T_true, "registered": registered, "cloud": cloud,
          "camera_index": camera_index, "exposures": exposures, "folder": Path(folder)}
    r2 = run_segmentation(r1, verbose=False)
    r3 = run_fit(r1, r2, verbose=False)
    return r1, r2, r3


def inside_gt_boxes(points_camera0, ground_truth, margin=INSIDE_MARGIN):
    """Para cada caixa verdadeira, máscara dos pontos (referencial da câmara 000) que estão dentro dela
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

    points_table, labels: pontos segmentados (referencial da mesa) e caixa (conjunto) de cada um.
    all_points_camera0: TODOS os pontos da nuvem agregada (para saber quantos pontos tem cada caixa
                        verdadeira, incluindo os que a segmentação deitou fora).
    - Precisão: fração dos pontos do conjunto que pertence de facto à caixa correspondente;
    - Recall: fração dos pontos da caixa que ficou no conjunto certo. As duas nuvens têm densidades
      diferentes (a dos objetos foi reduzida), por isso conta-se com voxels: fração dos voxels ocupados
      pela caixa verdadeira que ficaram no conjunto certo.
    Cada conjunto é emparelhado com a caixa verdadeira com que tem mais pontos em comum.
    Saída: (n.º de conjuntos, n.º de caixas reais, lista de {gt_box, conjunto, pontos, precision, recall}).
    """
    pts_c0 = transform_points(T_camera0_table, points_table)
    gt_masks_seg = inside_gt_boxes(pts_c0, ground_truth)
    gt_masks_all = inside_gt_boxes(all_points_camera0, ground_truth)

    def voxel_keys(p):
        return set(map(tuple, np.floor(p / voxel).astype(np.int64)))

    num_sets = int(labels.max() + 1) if np.any(labels >= 0) else 0
    overlap = np.array([[np.sum((labels == s) & m) for m in gt_masks_seg] for s in range(num_sets)])
    rows, cols = linear_sum_assignment(-overlap) if num_sets else ([], [])
    results = []
    for s, j in sorted(zip(rows, cols), key=lambda sj: sj[1]):
        mine = labels == s
        precision = np.sum(mine & gt_masks_seg[j]) / max(np.sum(mine), 1)
        gt_keys = voxel_keys(all_points_camera0[gt_masks_all[j]])
        my_keys = voxel_keys(pts_c0[mine & gt_masks_seg[j]])
        recall = len(gt_keys & my_keys) / max(len(gt_keys), 1)
        results.append({"gt_box": int(j), "conjunto": int(s), "pontos": int(np.sum(mine)),
                        "precision": float(precision), "recall": float(recall)})
    return num_sets, len(ground_truth["boxes"]), results


# #####################################################################################################
# 4. PARTE 3 - CAIXAS
# #####################################################################################################
# Compara-se no referencial do MUNDO (origem no centro do tampo, z exatamente para cima): as caixas
# estimadas (referencial da câmara 000) passam para o mundo com a pose verdadeira da câmara 000.
# Do enunciado (não está nas aulas): o IoU 3D (cv2.rotatedRectangleIntersection + cv2.contourArea,
# multiplicado pela sobreposição das alturas), as simetrias da caixa e a associação pelo IoU
# (algoritmo húngaro, scipy.optimize.linear_sum_assignment).
def box_in_world(T_world_box, length, width, height):
    """Caixa no referencial do mundo (z para cima): centro, yaw e dimensões."""
    T_world_box = np.asarray(T_world_box)
    return {"center": T_world_box[:3, 3], "yaw": yaw_of(T_world_box[:3, :3]),
            "dims": np.array([length, width, height], float)}


def world_params(box):
    """Caixa do mundo no formato [x, y, yaw, length, width, height, z] (z = base), para as figuras."""
    c, d = box["center"], box["dims"]
    return np.array([c[0], c[1], box["yaw"], d[0], d[1], d[2], c[2] - d[2] / 2])


def iou_3d(a, b):
    """IoU 3D de duas caixas com z para cima.

    Volume da interseção = área da interseção dos retângulos vistos de cima
                           x sobreposição dos intervalos de altura (de z a z + height).
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
    return float(inter / union) if union > 0 else 0.0


def box_errors(est, gt):
    """Erros de uma caixa estimada em relação à verdadeira, com as simetrias.

    Versões equivalentes da caixa verdadeira: rodar k * 90° em torno de z (k = 0..3), trocando length com
    width quando k é ímpar (rodar 180° não muda a caixa). Fica a versão de menor erro de yaw (as
    dimensões só desempatam, nas caixas quadradas).
    Erro de altura: diferença entre os z, isto é, as alturas das BASES (centro - height / 2).
    """
    pos_err = np.linalg.norm(est["center"] - gt["center"]) * 1000
    best = None
    for k in range(4):
        yaw_err = abs(np.degrees(wrap_angle(est["yaw"] - gt["yaw"] - k * np.pi / 2)))
        dims_gt = gt["dims"][[1, 0, 2]] if k % 2 else gt["dims"]
        dim_err = (est["dims"] - dims_gt) * 1000
        score = yaw_err + np.abs(dim_err).sum() * 0.01
        if best is None or score < best[0]:
            best = (score, yaw_err, dim_err)
    _, yaw_err, dim_err = best
    z_err = ((est["center"][2] - est["dims"][2] / 2) - (gt["center"][2] - gt["dims"][2] / 2)) * 1000
    return {"pos_mm": float(pos_err), "dz_mm": float(z_err), "yaw_deg": float(yaw_err), "dl_mm": float(dim_err[0]),
            "dw_mm": float(dim_err[1]), "dh_mm": float(dim_err[2]), "iou": iou_3d(est, gt)}


def evaluate_boxes(est_boxes, ground_truth):
    """Associa cada caixa estimada a uma verdadeira (máximo IoU, algoritmo húngaro) e calcula os erros.

    est_boxes: caixas no formato do result.json (T_camera0_box, length, width, height).
    Saída: (uma linha por caixa verdadeira, com o índice da estimada associada ou None e IoU = 0 se
    nenhuma lhe foi associada; índices das caixas estimadas que ficaram a mais).
    """
    T_world_camera0 = np.array(ground_truth["cameras"][0]["T_world_camera"])
    gt = [box_in_world(b["T_world_box"], b["length"], b["width"], b["height"]) for b in ground_truth["boxes"]]
    est = [box_in_world(T_world_camera0 @ np.array(b["T_camera0_box"]), b["length"], b["width"], b["height"])
           for b in est_boxes]
    iou = np.array([[iou_3d(e, g) for g in gt] for e in est]).reshape(len(est), len(gt))
    rows_est, cols_gt = linear_sum_assignment(-iou) if len(est) else ([], [])
    matched = {j: i for i, j in zip(rows_est, cols_gt) if iou[i, j] > 0}
    rows = []
    for j in range(len(gt)):
        if j in matched:
            rows.append({"gt_box": j, "est_box": int(matched[j]), **box_errors(est[matched[j]], gt[j])})
        else:
            rows.append({"gt_box": j, "est_box": None, "pos_mm": np.nan, "dz_mm": np.nan, "yaw_deg": np.nan,
                         "dl_mm": np.nan, "dw_mm": np.nan, "dh_mm": np.nan, "iou": 0.0})
    extra = [i for i in range(len(est)) if i not in matched.values()]
    return rows, extra


def boxes_in_world(est_boxes, ground_truth):
    """Caixas (formato do result.json) no referencial do mundo, como [x, y, yaw, l, w, h, z] (figuras)."""
    T_world_camera0 = np.array(ground_truth["cameras"][0]["T_world_camera"])
    params = [world_params(box_in_world(T_world_camera0 @ np.array(b["T_camera0_box"]), b["length"],
                                        b["width"], b["height"])) for b in est_boxes]
    return np.array(params).reshape(-1, 7)


def bbox_boxes(parts_folder):
    """Caixas do método base (bounding box), gravadas pela Parte 3 em partes/ajuste.npz, no formato do
    result.json (o result.json só tem as do método escolhido, a otimização). None se não existirem."""
    try:
        table, _, _ = load_objects(parts_folder)
        params = load_fit(parts_folder)["params_bbox"]
    except (OSError, KeyError):
        return None
    return [{k: np.asarray(v) if k == "T_camera0_box" else float(v) for k, v in b.items()}
            for b in boxes_to_camera0(params, table.T_camera0_table)]


# #####################################################################################################
# 5. EXTRA - FOTOMETRIA
# #####################################################################################################
# O extra estima o ganho s e o desvio b de cada imagem em relação à imagem 000; aqui comparam-se com os do
# campo 'photometry' do ground_truth.json (se a cena foi gerada com "Exposure changes").
def true_exposures(ground_truth):
    """s e b VERDADEIROS de cada imagem em relação à imagem 000.

    O ground_truth.json dá, para cada imagem k, cor_k = s_k * cor + b_k. Tirando a cor verdadeira:
        cor_0 = s_0 / s_k * (cor_k - b_k) + b_0 = s_rel * cor_k + b_rel
    com s_rel = s_0 / s_k e b_rel = b_0 - s_0 * b_k / s_k. Saída: {imagem: (s_rel, b_rel)}, ou {}.
    """
    if not ground_truth or "photometry" not in ground_truth:
        return {}
    photometry = {p["image"]: (p["s"], p["b"]) for p in ground_truth["photometry"]}
    s0, b0 = photometry[min(photometry)]
    return {name: (s0 / s, b0 - s0 * b / s) for name, (s, b) in photometry.items()}


def photometry_errors(parts_folder, ground_truth):
    """s e b estimados (partes/fotometria.json) e verdadeiros de cada imagem. Lista vazia se faltar algum."""
    path = Path(parts_folder) / PHOTOMETRY_FILE
    truth = true_exposures(ground_truth)
    if not truth or not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [{"imagem": r["imagem"], "s": r["s"], "b": r["b"], "s_gt": truth[r["imagem"]][0],
             "b_gt": truth[r["imagem"]][1]} for r in rows if r["imagem"] in truth]


# #####################################################################################################
# 6. RELATÓRIO (avaliacao.md) E FIGURAS
# #####################################################################################################
def fmt(v, d=1):
    """Número com d casas decimais, ou "-" se não existir (para as tabelas)."""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def figure_data(name, est_cloud, gt_cloud, T_world_camera0, boxes, gt_params):
    """Dados das janelas do avaliador (só arrays, para o run_test.py), no referencial do mundo:
    a nuvem com as poses estimadas e a verdadeira, os pontos sobre a mesa à volta das caixas (com a cor
    real), as caixas de cada método e as verdadeiras."""
    est = transform_points(T_world_camera0, np.asarray(est_cloud.points))
    gt = transform_points(T_world_camera0, np.asarray(gt_cloud.points))
    lo, hi = gt_params[:, :2].min(axis=0) - 0.1, gt_params[:, :2].max(axis=0) + 0.1

    def region(points, margin, z_min):
        """Pontos à volta das caixas verdadeiras (vistas de cima, com uma margem) e acima de z_min."""
        return (np.all((points[:, :2] > lo - margin) & (points[:, :2] < hi + margin), axis=1)
                & (points[:, 2] > z_min) & (points[:, 2] < 0.6))

    near = region(est, 0.0, 0.008)                 # pontos dos objetos (sem a mesa)
    objects = est[near] if near.sum() > 100 else gt[:1]
    center = (gt_params[:, :2].min(axis=0) + gt_params[:, :2].max(axis=0)) / 2
    view = camera_view(T_world_camera0, np.array([center[0], center[1], 0.1]), np.array([0.0, 0.0, 1.0]))
    data = {"parte": "avaliador", "cena": name, "est_points": est.astype(np.float32),
            "gt_points": gt.astype(np.float32), "objects": objects.astype(np.float32),
            "est_near": est[region(est, 0.1, -0.03)].astype(np.float32),     # mesa e caixas (vista de perto)
            "gt_near": gt[region(gt, 0.1, -0.03)].astype(np.float32),
            "object_colors": colors_uint8(np.asarray(est_cloud.colors)[near] if near.sum() > 100 else [[0.5] * 3]),
            "gt_params": np.asarray(gt_params, float), **{f"view_{k}": v for k, v in view.items()}}
    for key, _ in METHODS:
        data[f"params_{key}"] = np.zeros((0, 7)) if boxes.get(key) is None else np.asarray(boxes[key], float)
    return data


def make_windows(data):
    """Vistas do avaliador:
    1. Parte 1: a nuvem com as poses estimadas (vermelho) sobre a nuvem com as poses verdadeiras (verde);
    2. o mesmo, de perto, à volta das caixas;
    3-6. Parte 3: as caixas de cada método (cores) sobre as verdadeiras (preto), vistas de cima e em 3D."""
    title = f"Avaliador - {data['cena']}"
    view = data_view(data)
    def clouds(est, gt):
        """A nuvem verdadeira (verde) e, por cima, a estimada (vermelho), cada uma com uma só cor."""
        return [points_item(gt, np.tile(colors_uint8(GT_COLOR), (len(gt), 1))),
                points_item(est, np.tile(colors_uint8(EST_COLOR), (len(est), 1)))]

    windows = [{"name": "_registo", "title": f"{title}: nuvem com as poses estimadas (vermelho) sobre a verdadeira (verde)",
                "items": clouds(data["est_points"], data["gt_points"]), "view": view},
               {"name": "_registo_perto", "title": f"{title}: estimada (vermelho) sobre a verdadeira (verde), mesa e caixas",
                "items": clouds(data["est_near"], data["gt_near"]), "view": close_view(view, data["objects"], factor=2.0)}]
    objects = points_item(data["objects"], data["object_colors"], 3.0)
    gt = [box_item(p, GT_BOX_COLOR, 2.0) for p in data["gt_params"]]
    for key, label in METHODS:
        if len(data[f"params_{key}"]) == 0:
            continue
        boxes = [box_item(p, BOX_COLORS[j % len(BOX_COLORS)]) for j, p in enumerate(data[f"params_{key}"])]
        windows.append({"name": f"_caixas_{key}_vista_de_cima",
                        "title": f"{title}: {label}, vista de cima (verdadeiras a preto)",
                        "items": [objects] + gt + boxes, "view": top_view(data["objects"])})
        windows.append({"name": f"_caixas_{key}_3d", "title": f"{title}: {label}, em 3D (verdadeiras a preto)",
                        "items": [objects] + gt + boxes, "view": close_view(view, data["objects"])})
    return windows


def save_top_view(path, data, title=""):
    """Figura PNG vista de cima (matplotlib): pontos sobre a mesa, caixas verdadeiras (preto) e as dos
    dois métodos (otimização a cores, bounding box a tracejado)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    pts, cols = data["objects"], data["object_colors"] / 255.0
    sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 15000), replace=False)
    ax.scatter(pts[sel, 0], pts[sel, 1], c=cols[sel], s=1)
    styles = {"gt": ({"color": "k", "lw": 2.5}, "verdadeira"), "bounding_box": ({"color": "gray", "ls": "--"}, "bounding box"),
              "otimizacao": ({}, "otimização")}
    for key, params in (("gt", data["gt_params"]), ("bounding_box", data["params_bounding_box"]),
                        ("otimizacao", data["params_otimizacao"])):
        style, label = styles[key]
        for j, p in enumerate(params):
            c = box_corners_2d(p[0], p[1], p[2], p[3], p[4])
            c = np.vstack([c, c[:1]])
            ax.plot(c[:, 0], c[:, 1], color=style.get("color") or BOX_COLORS[j % len(BOX_COLORS)],
                    lw=style.get("lw", 1.5), ls=style.get("ls", "-"), label=label if j == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x mundo (m)")
    ax.set_ylabel("y mundo (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(path, name, count, cams, seg, boxes, photometry):
    """Escreve o relatório avaliacao.md: os tópicos de "Comparação com o ground truth" de cada parte."""
    found, real, num_gt = count
    text = [f"# Avaliação da cena {name}", "",
            "Comparação do `result.json` com o `ground_truth.json` (o ground truth só é lido pelo avaliador).", "",
            "## Número de caixas", "", f"No `result.json`: {box_count_verdict(found, real)}."]
    if real != num_gt:
        text.append(f"(Atenção: o número real indicado, {real}, não é o do ground truth, {num_gt}.)")

    text += ["", "## Parte 1 - Registo", "",
             "Erro de cada câmara: E = T_gt⁻¹ · T_est; translação = norma da translação de E, rotação = "
             "ângulo da rotação de E (o mesmo de R_estᵀ R_gt).", "",
             "| câmara | erro de translação (mm) | erro de rotação (°) |", "|---|---|---|"]
    text += [f"| {c['imagem']} | {c['transl_mm']:.2f} | {c['rot_deg']:.3f} |" for c in cams]
    if len(cams) > 1:
        t, r = [c["transl_mm"] for c in cams[1:]], [c["rot_deg"] for c in cams[1:]]
        text.append(f"| média (sem a 000) | {np.mean(t):.2f} | {np.mean(r):.3f} |")
        text.append(f"| máximo | {max(t):.2f} | {max(r):.3f} |")
    text += ["", "Nuvem agregada com as poses estimadas (vermelho) sobre a nuvem com as poses verdadeiras (verde):", "",
             f"![registo]({FIGURE_NAME}_registo.png)", f"![registo de perto]({FIGURE_NAME}_registo_perto.png)"]

    text += ["", "## Parte 2 - Segmentação", "",
             "Para avaliar só a segmentação (sem os erros do registo), as nuvens foram juntadas com as poses "
             f"verdadeiras e as Partes 2 e 3 correram sobre essa nuvem (em `{GT_FOLDER}/`). Os pontos de cada "
             f"caixa verdadeira são os que ficam dentro dela, com {INSIDE_MARGIN * 1000:.0f} mm de margem.", ""]
    if isinstance(seg, str):
        text.append(f"Não foi possível avaliar a segmentação: {seg}")
    else:
        found_seg, n_sets, num_gt, rows = seg
        text += [f"Caixas encontradas: {found_seg}, reais: {num_gt} ({n_sets} conjuntos de pontos).", "",
                 "| caixa verdadeira | conjunto | pontos do conjunto | precisão | recall |", "|---|---|---|---|---|"]
        text += [f"| {r['gt_box']} | {r['conjunto']} | {r['pontos']} | {r['precision']:.3f} | {r['recall']:.3f} |"
                 for r in rows]
        if rows:
            text.append(f"| média | | | {np.mean([r['precision'] for r in rows]):.3f} | "
                        f"{np.mean([r['recall'] for r in rows]):.3f} |")
        missing = sorted(set(range(num_gt)) - {r["gt_box"] for r in rows})
        if missing:
            text.append(f"\nCaixas verdadeiras sem conjunto: {', '.join(map(str, missing))}.")

    text += ["", "## Parte 3 - Caixas", "",
             "Erros de cada caixa verdadeira em relação à caixa estimada associada (maior IoU 3D), no referencial "
             "do mundo; o yaw e as dimensões têm em conta as simetrias da caixa. dL, dW, dH = estimado − verdadeiro.", ""]
    for key, label in METHODS:
        if boxes.get(key) is None:
            text += [f"### {label}", "", "Não existe.", ""]
            continue
        rows, extra = boxes[key]
        text += [f"### {label}", "",
                 "| caixa verdadeira | caixa estimada | posição (mm) | altura z (mm) | yaw (°) | dL (mm) | dW (mm) | dH (mm) | IoU 3D |",
                 "|---|---|---|---|---|---|---|---|---|"]
        text += [f"| {r['gt_box']} | {'-' if r['est_box'] is None else r['est_box']} | {fmt(r['pos_mm'])} | "
                 f"{fmt(r['dz_mm'])} | {fmt(r['yaw_deg'], 2)} | {fmt(r['dl_mm'])} | {fmt(r['dw_mm'])} | "
                 f"{fmt(r['dh_mm'])} | {r['iou']:.3f} |" for r in rows]
        text.append(f"\nIoU 3D médio: **{np.mean([r['iou'] for r in rows]):.3f}** "
                    "(uma caixa verdadeira sem caixa estimada conta com IoU 0)."
                    + (f" Caixas estimadas a mais: {', '.join(map(str, extra))}." if extra else ""))
        text.append("")
    text += ["Caixas estimadas sobrepostas às verdadeiras (preto):", "",
             f"![vista de cima]({FIGURE_NAME}_vista_de_cima.png)",
             f"![otimização em 3D]({FIGURE_NAME}_caixas_otimizacao_3d.png)",
             f"![bounding box em 3D]({FIGURE_NAME}_caixas_bounding_box_3d.png)"]

    if photometry:
        text += ["", "## Extra - Fotometria", "", "cor_000 = s · cor + b de cada imagem:", "",
                 "| imagem | s estimado | s verdadeiro | b estimado | b verdadeiro |", "|---|---|---|---|---|"]
        text += [f"| {p['imagem']} | {p['s']:.3f} | {p['s_gt']:.3f} | {p['b']:+.3f} | {p['b_gt']:+.3f} |"
                 for p in photometry]
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


# #####################################################################################################
# 7. PROGRAMA PRINCIPAL
# #####################################################################################################
def evaluate_scene(scene_folder, result_path=None, num_boxes=None, verbose=True):
    """Avaliador numa cena: número de caixas e, se houver ground truth, o relatório avaliacao.md.

    Entrada: pasta da cena, o result.json (por omissão <cena>/result.json) e o número real de caixas
    (se faltar, é perguntado no terminal).
    Saída: os dados das figuras (figure_data, para o run_test.py as mostrar), ou None sem ground truth.
    """
    scene_folder = Path(scene_folder)
    result_path = Path(result_path) if result_path else scene_folder / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    parts = scene_folder / OUTPUT_FOLDER

    # 1. Número de caixas
    found = len(result["boxes"])
    real = num_boxes if num_boxes else ask_num_boxes(scene_folder.name)
    print(f"\nNúmero de caixas: {box_count_verdict(found, real)}")

    ground_truth = load_ground_truth(scene_folder)
    if ground_truth is None:
        print("A cena não tem ground_truth.json: não há mais nada a comparar.")
        return None
    scene = load_scene(scene_folder)

    # 2. Parte 1
    cams = camera_errors(result, ground_truth)
    try:
        registered = load_registration(parts)["registered"]       # câmaras que entraram na nuvem
    except OSError:
        registered = None
    est_cloud, gt_cloud = registration_clouds(scene, result, ground_truth, registered)

    # 3. Parte 2 (com as poses verdadeiras)
    if verbose:
        print("  a correr as Partes 2 e 3 com as poses verdadeiras (só para avaliar a segmentação)...")
    try:
        _, r2, r3 = segmentation_with_true_poses(scene, ground_truth, scene_folder / GT_FOLDER)
        n_sets, num_gt, seg_rows = evaluate_segmentation(r2["seg"].points, r3["box_labels"],
                                                         r2["table"].T_camera0_table,
                                                         np.asarray(r2["cloud"].points), ground_truth)
        seg = (int(r2["num_boxes"]), n_sets, num_gt, seg_rows)
    except Exception as error:                                    # a avaliação continua sem esta parte
        seg = f"{type(error).__name__}: {error}"

    # 4. Parte 3 (os dois métodos)
    est = {"otimizacao": result["boxes"], "bounding_box": bbox_boxes(parts)}
    boxes = {k: evaluate_boxes(v, ground_truth) if v is not None else None for k, v in est.items()}

    # 5. Extra
    photometry = photometry_errors(parts, ground_truth)

    # 6. Relatório e figuras
    T_world_camera0 = np.array(ground_truth["cameras"][0]["T_world_camera"])
    gt_params = np.array([world_params(box_in_world(b["T_world_box"], b["length"], b["width"], b["height"]))
                          for b in ground_truth["boxes"]])
    data = figure_data(scene_folder.name, est_cloud, gt_cloud, T_world_camera0,
                       {k: boxes_in_world(v, ground_truth) if v is not None else None for k, v in est.items()},
                       gt_params)
    out = result_path.parent
    save_top_view(out / f"{FIGURE_NAME}_vista_de_cima.png", data, title=scene_folder.name)
    write_report(out / REPORT_FILE, scene_folder.name, (found, real, len(ground_truth["boxes"])), cams, seg,
                 boxes, photometry)
    if verbose:
        t = [c["transl_mm"] for c in cams]
        print(f"  Parte 1: erro máximo das câmaras {max(t):.2f} mm, {max(c['rot_deg'] for c in cams):.3f}°")
        if not isinstance(seg, str) and seg[3]:
            print(f"  Parte 2: {seg[0]} caixas encontradas para {seg[2]}; precisão média "
                  f"{np.mean([r['precision'] for r in seg[3]]):.3f}, recall médio {np.mean([r['recall'] for r in seg[3]]):.3f}")
        for key, label in METHODS:
            if boxes[key] is not None:
                print(f"  Parte 3, {label}: IoU 3D médio {np.mean([r['iou'] for r in boxes[key][0]]):.3f}")
        print(f"  relatório escrito em {out / REPORT_FILE}")
    return data


def main():
    """Lê os argumentos da linha de comandos e avalia uma cena."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena (já com as 4 partes feitas)")
    parser.add_argument("--result", help="result.json a avaliar (por omissão: <cena>/result.json)")
    parser.add_argument("--num_boxes", type=int, help="número real de caixas (se faltar, é perguntado)")
    parser.add_argument("--visualize", action="store_true", help="abre as janelas 3D (as figuras são sempre gravadas)")
    args = parser.parse_args()

    result_path = Path(args.result) if args.result else Path(args.scene) / "result.json"
    data = evaluate_scene(args.scene, result_path, args.num_boxes)
    if data is None:
        return
    windows = SceneWindows(show=args.visualize)
    for spec in make_windows(data):
        windows.add(spec, result_path.parent / f"{FIGURE_NAME}{spec['name']}.png", group=str(data["cena"]))
    windows.run()


if __name__ == "__main__":
    main()
