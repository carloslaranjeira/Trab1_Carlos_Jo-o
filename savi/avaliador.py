#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. AVALIADOR: comparação com o ground truth (só para avaliar, NUNCA para estimar).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho)
    python savi/avaliador.py --scene scenes_offline/level_4        # depois das 4 partes: tabelas de erros
    python savi/avaliador.py --evaluate_scenes scenes_offline/level_* scenes_offline/stack_*
    python savi/avaliador.py --evaluate_scenes scenes_offline/level_* --gt_registration

O número REAL de caixas só é pedido aqui (--num_boxes, ou no terminal): as 4 partes não o usam (a
Parte 2 estima-o) e o avaliador compara-o com o número de caixas encontradas. Os erros precisam do
ground_truth.json da cena (as cenas do dia da apresentação não o têm). O run_test.py chama-o depois das
4 partes de cada cena.
--evaluate_scenes corre as 4 partes em várias cenas (o n.º real de caixas vem do ground truth) e escreve as
tabelas em results_offline/results.md; com --gt_registration usa as poses verdadeiras das câmaras
(avalia só a segmentação e o ajuste).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO
=====================================================================================================
    1. Registo            erro de translação (mm) e de rotação (graus) de cada câmara
    2. Segmentação        n.º de conjuntos, precisão e recall de cada conjunto de pontos
    3. Caixas             erro de posição, yaw, dimensões, altura da base e IoU 3D, com as simetrias
    4. Uma cena           tabelas no terminal (também usado pelo run_test.py)
    5. Várias cenas       corre as 4 partes em várias cenas e escreve as tabelas (--evaluate_scenes)
    6. Programa principal
(A comparação da fotometria com o ground truth está no extra_fotometria.py.)
"""

import argparse
import glob
from pathlib import Path

import cv2                                   # OpenCV: interseção de retângulos (IoU 3D)
import matplotlib.pyplot as plt              # figuras PNG do --evaluate_scenes
import numpy as np
from scipy.optimize import linear_sum_assignment   # emparelhar caixas / conjuntos com o ground truth

from extra_fotometria import run_photometry
from comum import (  # geometria e leitura da cena (comuns a todas as partes)
    boxes_to_camera0, load_ground_truth, load_scene, OUTPUT_FOLDER, rotation_angle_deg, transform_points,
    wrap_angle, yaw_of,
)
from parte1_registo import (  # funções da Parte 1 usadas aqui
    aggregate, load_registration, run_registration,
)
from parte2_segmentacao import load_objects, load_segmentation, run_segmentation
from parte3_ajuste import gt_params_in_table, load_fit, run_fit, save_evolution, save_top_view


# #####################################################################################################
# 1. REGISTO
# #####################################################################################################
# Registo: erro de translação (mm) e de rotação (graus) de cada câmara (Parte 1).
def camera_errors(T_est_list, ground_truth):
    """Lista de (imagem, erro de translação em mm, erro de rotação em graus)."""
    rows = []
    for T_est, cam in zip(T_est_list, ground_truth["cameras"]):
        T_gt = np.array(cam["T_camera0_camera"])
        t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000
        r_err = rotation_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3])   # ângulo de R_est^T R_gt
        rows.append((cam["image"], t_err, r_err))
    return rows


# #####################################################################################################
# 2. SEGMENTAÇÃO
# #####################################################################################################
# Segmentação: n.º de conjuntos encontrados, precisão e recall de cada conjunto de pontos (Parte 2).
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


# #####################################################################################################
# 3. CAIXAS
# #####################################################################################################
# Caixas: erro de posição (mm), de yaw (graus), das dimensões (mm), da altura da base (mm) e IoU 3D,
# tendo em conta as simetrias da caixa (rodar 180° ou trocar comprimento e largura é a mesma caixa) (Parte 3).
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
    # Erro de altura: diferença entre as alturas das BASES (z = centro - height/2)
    z_err = ((est["center"][2] - est["dims"][2] / 2) - (gt["center"][2] - gt["dims"][2] / 2)) * 1000
    return {"pos_mm": pos_err, "yaw_deg": yaw_err, "dl_mm": dim_err[0], "dw_mm": dim_err[1],
            "dh_mm": dim_err[2], "dz_mm": z_err, "iou": iou_3d(est, gt)}


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
                                       "dw_mm": np.nan, "dh_mm": np.nan, "dz_mm": np.nan,
                                       "iou": 0.0}))
    return results


# #####################################################################################################
# 4. UMA CENA: TABELAS DE ERROS NO TERMINAL
# #####################################################################################################
def full_cloud_points(r):
    """TODOS os pontos da nuvem agregada (referencial da câmara 0), só para o recall da segmentação.

    A Parte 2 só grava os pontos dos objetos; para saber quantos pontos tem cada caixa verdadeira
    (incluindo os que a segmentação deitou fora) é precisa a nuvem agregada: a que já existe, se a
    cena acabou de correr, ou volta-se a juntar as nuvens com as poses já estimadas (sem novo registo).
    """
    if r.get("cloud") is not None:
        return np.asarray(r["cloud"].points)
    cloud, _ = aggregate(load_scene(r["scene_folder"]), r["T_cams"], r["registered"])
    return np.asarray(cloud.points)


def ask_num_boxes(name=None):
    """Pergunta no terminal o número REAL de caixas da cena. Só o avaliador o pede: as 4 partes não o
    usam (a Parte 2 estima-o); aqui serve para comparar com o número de caixas encontradas."""
    while True:
        answer = input(f"Número real de caixas{f' da cena {name!r}' if name else ''}: ").strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("  escreve um número inteiro positivo")


def found_boxes(r):
    """Número de caixas que a segmentação encontrou (estimado na Parte 2)."""
    return int(sum(r["seg"].boxes_per_cluster.values()))


def print_box_count(found, real):
    """Compara o número de caixas encontradas (Parte 2) com o número real (dado ao avaliador)."""
    verdict = "certo" if found == real else (f"{real - found} a menos" if found < real else f"{found - real} a mais")
    print(f"\nNúmero de caixas: encontradas {found}, reais {real} ({verdict})")


def print_evaluation(r):
    """Mostra no terminal as tabelas de erros de uma cena (r = resultado de run())."""
    gt = r["ground_truth"]
    print("\nRegisto (erro de cada câmara):")
    print(f"  {'imagem':8s}{'transl. (mm)':>14s}{'rot. (°)':>10s}")
    for name, t, a in camera_errors(r["T_cams"], gt):
        print(f"  {name:8s}{t:14.2f}{a:10.3f}")

    n_sets, n_gt, seg_rows = evaluate_segmentation(
        r["seg"].points, r["box_labels"], r["table"].T_camera0_table, full_cloud_points(r), gt)
    print(f"\nSegmentação: {n_sets} conjuntos para {n_gt} caixas")
    for row in seg_rows:
        print(f"  caixa {row['gt_box']}: precisão {row['precision']:.3f}, recall {row['recall']:.3f}")

    T_ct = r["table"].T_camera0_table
    for name in ("bbox", "opt"):
        rows = evaluate_boxes(boxes_to_camera0(r[f"params_{name}"], T_ct), gt)
        print(f"\nCaixas - {'bounding box' if name == 'bbox' else 'otimização'}:")
        print(f"  {'caixa':6s}{'pos (mm)':>10s}{'yaw (°)':>9s}{'dL (mm)':>9s}{'dW (mm)':>9s}{'dH (mm)':>9s}"
              f"{'dz (mm)':>9s}{'IoU':>7s}")
        for j, e in enumerate(rows):
            print(f"  {j:<6d}{e['pos_mm']:10.1f}{e['yaw_deg']:9.2f}{e['dl_mm']:9.1f}{e['dw_mm']:9.1f}"
                  f"{e['dh_mm']:9.1f}{e['dz_mm']:9.1f}{e['iou']:7.3f}")
        print(f"  IoU médio: {np.mean([e['iou'] for e in rows]):.3f}")


# #####################################################################################################
# 5. VÁRIAS CENAS (opção --evaluate_scenes: gera as tabelas do Read_me)
# #####################################################################################################
ROOT_FOLDER = Path(__file__).resolve().parent.parent      # pasta principal do trabalho


def fmt(v, d=1):
    """Número com d casas decimais, ou "-" se não existir (para as tabelas)."""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def run_all(scene_folder, gt_registration=False, use_color=True, verbose=True):
    """Corre as 4 partes numa cena, uma a seguir à outra, e junta os resultados num só dicionário."""
    r1 = run_registration(scene_folder, gt_registration=gt_registration, verbose=verbose)   # Parte 1
    rx = run_photometry(r1, verbose=verbose)                                                # extra
    r2 = run_segmentation(r1, use_color=use_color, verbose=verbose)                         # Parte 2
    r3 = run_fit(r1, r2, use_color=use_color, verbose=verbose)                              # Parte 3
    r = {**r1, **rx, **r2, **r3}
    r["ground_truth"] = load_ground_truth(r1["scene_folder"])
    return r


def evaluate_scenes(patterns, gt_registration=False, use_color=True, output=None):
    """Corre VÁRIAS cenas com ground truth e escreve as tabelas de erros dos dois métodos (para o Read_me).

    patterns: pastas das cenas (aceita wildcards, por exemplo scenes_offline/level_*). As partes estimam
    o número de caixas; o número real vem do ground_truth.json (só para comparar).
    gt_registration=True usa as poses verdadeiras das câmaras (avalia só a segmentação e o ajuste).
    Escreve <output>/results.md (tabelas em Markdown) e duas figuras por cena. As cenas são
    processadas uma de cada vez.
    """
    plt.switch_backend("Agg")            # só grava figuras PNG (não abre janelas)
    folders = sorted({f for pattern in patterns for f in glob.glob(pattern) if Path(f, "ground_truth.json").exists()})
    out = Path(output) if output else ROOT_FOLDER / "results_offline"
    out.mkdir(parents=True, exist_ok=True)
    if not folders:
        print("Nenhuma cena com ground_truth.json encontrada.")
        return

    reg_lines = ["| cena | erro transl. médio (mm) | máx (mm) | erro rot. médio (°) | máx (°) | só encadeamento: transl. máx (mm) |",
                 "|---|---|---|---|---|---|"]
    seg_lines = ["| cena | caixas encontradas / reais | conjuntos | precisão média | recall médio |",
                 "|---|---|---|---|---|"]
    box_lines = ["| cena | caixa | método | pos (mm) | yaw (°) | dL (mm) | dW (mm) | dH (mm) | dz (mm) | IoU |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
    summary = ["| cena | IoU médio bounding box | IoU médio otimização |", "|---|---|---|"]

    for folder in folders:
        name = Path(folder).name
        r = run_all(folder, gt_registration=gt_registration, use_color=use_color)   # 4 partes
        gt = r["ground_truth"]

        if not gt_registration:
            cams = camera_errors(r["T_cams"], gt)
            chain = camera_errors(r["T_chain"], gt)
            t, a = np.array([c[1] for c in cams]), np.array([c[2] for c in cams])
            reg_lines.append(f"| {name} | {t[1:].mean():.2f} | {t.max():.2f} | {a[1:].mean():.3f} | {a.max():.3f} | "
                             f"{max(c[1] for c in chain):.2f} |")

        n_sets, n_gt, seg_rows = evaluate_segmentation(
            r["seg"].points, r["box_labels"], r["table"].T_camera0_table, full_cloud_points(r), gt)
        seg_lines.append(f"| {name} | {found_boxes(r)} / {n_gt} | {n_sets} | "
                         f"{np.mean([s['precision'] for s in seg_rows]):.3f} | "
                         f"{np.mean([s['recall'] for s in seg_rows]):.3f} |")

        ious = {}
        for method, label in (("bbox", "bounding box"), ("opt", "otimização")):
            rows = evaluate_boxes(boxes_to_camera0(r[f"params_{method}"], r["table"].T_camera0_table), gt)
            ious[method] = np.mean([e["iou"] for e in rows])
            for j, e in enumerate(rows):
                box_lines.append(f"| {name} | {j} | {label} | {fmt(e['pos_mm'])} | {fmt(e['yaw_deg'], 2)} | "
                                 f"{fmt(e['dl_mm'])} | {fmt(e['dw_mm'])} | {fmt(e['dh_mm'])} | {fmt(e['dz_mm'])} | {e['iou']:.3f} |")
        summary.append(f"| {name} | {ious['bbox']:.3f} | {ious['opt']:.3f} |")

        gt_params = gt_params_in_table(gt, r["table"].T_camera0_table)
        save_top_view(out / f"{name}_top_view.png", r["seg"].points, r["seg"].colors,
                                    r["params_opt"], r["params_bbox"], gt_params, title=name)
        save_evolution(out / f"{name}_optimization.png", r["fit_points"], r["problem"])

    text = ["# Resultados", "",
            f"Registo: {'poses verdadeiras (só segmentação e ajuste)' if gt_registration else 'estimado'}", ""]
    if not gt_registration:
        text += ["## Registo", ""] + reg_lines + [""]
    text += ["## Segmentação", ""] + seg_lines + [""]
    text += ["## Resumo (IoU 3D médio)", ""] + summary + [""]
    text += ["## Erros por caixa", ""] + box_lines + [""]
    (out / "results.md").write_text("\n".join(text), encoding="utf-8")
    print(f"\nTabelas escritas em {out / 'results.md'}")


# #####################################################################################################
# 6. PROGRAMA PRINCIPAL
# #####################################################################################################
def load_scene_results(scene_folder):
    """Junta os resultados gravados pelas 4 partes em <cena>/partes/ (para avaliar uma cena já corrida)."""
    folder = Path(scene_folder) / OUTPUT_FOLDER
    r = load_registration(folder)
    table, points, colors = load_objects(folder)
    r.update({"table": table, "object_points": points, "object_colors": colors,
              "seg": load_segmentation(folder)})
    r.update(load_fit(folder))
    r["ground_truth"] = load_ground_truth(r["scene_folder"])
    return r


def main():
    """Lê os argumentos da linha de comandos e mostra os erros de uma cena (ou de várias)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="pasta de uma cena já corrida (com <cena>/partes/)")
    parser.add_argument("--num_boxes", type=int, help="com --scene: número real de caixas (se faltar, é perguntado)")
    parser.add_argument("--evaluate_scenes", nargs="+", metavar="CENA",
                        help="corre as 4 partes em várias cenas com ground truth (aceita wildcards) e escreve as tabelas")
    parser.add_argument("--gt_registration", action="store_true",
                        help="com --evaluate_scenes: usa as poses verdadeiras (avalia só segmentação e ajuste)")
    parser.add_argument("--no_color", action="store_true", help="com --evaluate_scenes: não usar a cor")
    parser.add_argument("--output", help="com --evaluate_scenes: pasta das tabelas (por omissão results_offline)")
    args = parser.parse_args()

    if args.evaluate_scenes:
        evaluate_scenes(args.evaluate_scenes, gt_registration=args.gt_registration,
                        use_color=not args.no_color, output=args.output)
        return
    if args.scene is None:
        parser.error("indica --scene (uma cena já corrida) ou --evaluate_scenes")
    r = load_scene_results(args.scene)
    # O número real de caixas só é pedido aqui, no avaliador (as 4 partes estimam-no)
    print_box_count(found_boxes(r), args.num_boxes if args.num_boxes else ask_num_boxes(Path(args.scene).name))
    if r["ground_truth"] is None:
        print("A cena não tem ground_truth.json: não há nada a comparar.")
        return
    print_evaluation(r)


if __name__ == "__main__":
    main()
