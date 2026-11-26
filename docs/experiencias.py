#!/usr/bin/env python3
"""Experiências usadas no README.md para responder às perguntas do enunciado.

    python docs/experiencias.py --scenes <pasta com as cenas geradas> [--so icp,cadeia,registo,ruido,cor]

Cada experiência grava docs/resultados/<nome>.json, que o docs/gerar_figuras.py transforma em gráficos.
As experiências usam o ground_truth.json (só para medir erros e, em algumas, para isolar uma parte do
sistema das outras, como o avaliador faz na Parte 2): o sistema (savi/) nunca o lê.

    icp      Parte 1: ICP com a nuvem completa contra o ICP só com a mesa e as caixas (sem o fundo)
    cadeia   Parte 1: erro de cada câmara só com encadeamento e com fecho de ciclo + grafo de poses
    registo  Parte 1: efeito dos erros de registo na nuvem agregada (e nas caixas)
    ruido    Nível 1: bounding box contra otimização com cada vez mais ruído
    cor      Nível 4: as três formas de usar a cor, com cada vez mais pontos de cor errada
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FOLDER / "savi"))
sys.path.insert(0, str(FOLDER / "simulator"))

from avaliador import evaluate_boxes, transform_error, true_camera_poses   # noqa: E402
from comum import boxes_to_camera0, load_scene, make_T                     # noqa: E402
from parte1_registo import (                                                # noqa: E402
    aggregate, MAX_DEPTH, multiscale_icp, preprocess_view, register_scene,
)
from parte2_segmentacao import run_segmentation, Segmentation             # noqa: E402
from parte3_ajuste import fit_boxes, run_fit                               # noqa: E402

RESULTS = FOLDER / "docs" / "resultados"


def ground_truth(scene_folder):
    return json.loads((Path(scene_folder) / "ground_truth.json").read_text(encoding="utf-8"))


def random_rotation(rng, angle_deg):
    """Rotação de angle_deg graus em torno de um eixo aleatório (fórmula de Rodrigues)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def perturbation(rng, translation_m, angle_deg):
    """Transformação com uma translação e uma rotação dos tamanhos dados, em direções aleatórias."""
    t = rng.normal(size=3)
    return make_T(random_rotation(rng, angle_deg), translation_m * t / np.linalg.norm(t))


def boxes_iou(params, table, gt):
    """IoU 3D médio e erro médio das dimensões (mm) de um conjunto de caixas (referencial da mesa)."""
    boxes = [{k: np.asarray(v) if k == "T_camera0_box" else float(v) for k, v in b.items()}
             for b in boxes_to_camera0(params, table.T_camera0_table)]
    rows, _ = evaluate_boxes(boxes, gt)
    dims = [abs(r[k]) for r in rows for k in ("dl_mm", "dw_mm", "dh_mm") if np.isfinite(r[k])]
    return float(np.mean([r["iou"] for r in rows])), float(np.mean(dims)) if dims else float("nan")


def parts_2_3(scene, T_cams, folder, use_color=True):
    """Partes 2 e 3 sobre a nuvem junta com as poses dadas (como o avaliador, secção 3)."""
    registered = [True] * len(T_cams)
    cloud, camera_index = aggregate(scene, T_cams, registered)
    r1 = {"scene": scene, "scene_folder": scene.folder, "images": [v.name for v in scene.views],
          "T_cams": T_cams, "registered": registered, "cloud": cloud, "camera_index": camera_index,
          "folder": Path(folder)}
    r2 = run_segmentation(r1, verbose=False)
    r3 = run_fit(r1, r2, use_color=use_color, verbose=False)
    return r1, r2, r3


# #####################################################################################################
# Parte 1: ICP com e sem o fundo
# #####################################################################################################
class CloudsOnly:
    """O que o multiscale_icp usa de uma câmara: as nuvens pré-processadas de cada escala."""
    def __init__(self, clouds):
        self.clouds = clouds


def table_and_boxes(view, T_world_camera, table, margin=0.05):
    """A vista só com os pontos da mesa e das caixas (tampo + 60 cm acima), cortada com a mesa verdadeira."""
    points = np.asarray(view.cloud.points)
    world = points @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    keep = ((np.abs(world[:, 0]) < table["length"] / 2 + margin) & (np.abs(world[:, 1]) < table["width"] / 2 + margin)
            & (world[:, 2] > -0.03) & (world[:, 2] < 0.6))
    cropped = copy.copy(view)
    cropped.cloud = view.cloud.select_by_index(np.flatnonzero(keep))
    return cropped


def experiment_icp(scenes):
    """Para cada par de câmaras vizinhas, ICP a partir da pose verdadeira perturbada (3 cm e 3°, 8 cm e 8°,
    15 cm e 15°; 5 direções ao acaso cada), com as nuvens completas e só com a mesa e as caixas.
    Erro final em relação à pose verdadeira; 'certo' = menos de 5 mm e 0.5°."""
    out = {}
    for name in ("level_1", "level_4"):
        folder = scenes / name
        scene, gt = load_scene(folder), ground_truth(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        T0 = true_camera_poses(scene, gt)
        T_world = {c["image"]: np.array(c["T_world_camera"]) for c in gt["cameras"]}
        full = [CloudsOnly(preprocess_view(v, max_depth=MAX_DEPTH)) for v in scene.views]
        crop = [CloudsOnly(preprocess_view(table_and_boxes(v, T_world[v.name], config["table"]))) for v in scene.views]
        rng = np.random.default_rng(0)
        rows = []
        for i in range(len(scene.views) - 1):
            T_true = np.linalg.inv(T0[i]) @ T0[i + 1]          # T_i_(i+1)
            for size_m, angle in ((0.03, 3.0), (0.08, 8.0), (0.15, 15.0)):
                for trial in range(5):
                    T_init = T_true @ perturbation(rng, size_m, angle)
                    for label, cams in (("completa", full), ("so_mesa_caixas", crop)):
                        T, _, _ = multiscale_icp(cams[i + 1], cams[i], T_init)
                        t_mm, r_deg = transform_error(T, T_true)
                        rows.append({"par": f"{i}-{i + 1}", "perturbacao_cm": size_m * 100, "nuvem": label,
                                     "transl_mm": t_mm, "rot_deg": r_deg})
        out[name] = rows
        for size in (3, 8, 15):
            for label in ("completa", "so_mesa_caixas"):
                mine = [r for r in rows if r["nuvem"] == label and round(r["perturbacao_cm"]) == size]
                ok = np.mean([r["transl_mm"] < 5 and r["rot_deg"] < 0.5 for r in mine])
                print(f"icp {name} {size} cm {label}: mediana {np.median([r['transl_mm'] for r in mine]):.1f} mm, "
                      f"{np.median([r['rot_deg'] for r in mine]):.2f}°; certos {ok:.0%}")
    return out


# #####################################################################################################
# Parte 1: encadeamento contra fecho de ciclo
# #####################################################################################################
def experiment_chain(scenes):
    out = {}
    for name in ("level_1", "level_4"):
        scene, gt = load_scene(scenes / name), ground_truth(scenes / name)
        T_true = true_camera_poses(scene, gt)
        out[name] = {}
        for label, loop in (("encadeamento", False), ("grafo_de_poses", True)):
            T_est, _ = register_scene(scene, loop_closure=loop, verbose=False)
            errors = [transform_error(T, Tt) for T, Tt in zip(T_est, T_true)]
            out[name][label] = [{"camera": k, "transl_mm": e[0], "rot_deg": e[1]} for k, e in enumerate(errors)]
            print(f"cadeia {name} {label}: " + ", ".join(f"{e[0]:.1f}" for e in errors))
    return out


# #####################################################################################################
# Parte 1: efeito dos erros de registo na nuvem agregada
# #####################################################################################################
def experiment_registration(scenes, work):
    out = {}
    levels = ((0.0, 0.0), (0.002, 0.2), (0.005, 0.5), (0.01, 1.0), (0.02, 2.0))
    for name in ("level_1", "level_4"):
        scene, gt = load_scene(scenes / name), ground_truth(scenes / name)
        T_true = true_camera_poses(scene, gt)
        rows = []
        for t, r in levels:
            rng = np.random.default_rng(1)
            T_cams = [T_true[0]] + [T @ perturbation(rng, t, r) for T in T_true[1:]]
            _, r2, r3 = parts_2_3(scene, T_cams, work / f"registo_{name}_{int(t * 1000)}")
            iou_opt, dims_opt = boxes_iou(r3["params_opt"], r2["table"], gt)
            iou_bbox, dims_bbox = boxes_iou(r3["params_bbox"], r2["table"], gt)
            rows.append({"transl_mm": t * 1000, "rot_deg": r, "iou_opt": iou_opt, "iou_bbox": iou_bbox,
                         "dims_opt_mm": dims_opt, "dims_bbox_mm": dims_bbox, "caixas": int(r2["num_boxes"])})
            print(f"registo {name} {t * 1000:.0f} mm/{r}°: IoU opt {iou_opt:.3f} bbox {iou_bbox:.3f}, "
                  f"dims {dims_opt:.1f}/{dims_bbox:.1f} mm")
        out[name] = rows
    return out


# #####################################################################################################
# Nível 1: ruído
# #####################################################################################################
def experiment_noise(work):
    """Cena do Nível 1 gerada com o ruído de profundidade (depth_std e flying_pixels) multiplicado por f."""
    from savi_box_simulator import read_config, write_scene
    out = []
    for f in (0.0, 0.5, 1.0, 2.0, 4.0):
        config = read_config(FOLDER / "simulator" / "configs" / "level_1.json")
        config["noise"]["depth_std"] *= f
        config["noise"]["flying_pixels"] = min(1.0, config["noise"]["flying_pixels"] * f)
        folder = work / f"ruido_{f}"
        write_scene(config, folder, with_noise=f > 0)
        scene, gt = load_scene(folder), ground_truth(folder)
        _, r2, r3 = parts_2_3(scene, true_camera_poses(scene, gt), folder / "partes")
        iou_opt, dims_opt = boxes_iou(r3["params_opt"], r2["table"], gt)
        iou_bbox, dims_bbox = boxes_iou(r3["params_bbox"], r2["table"], gt)
        out.append({"fator": f, "depth_std_mm_a_1m": config["noise"]["depth_std"] * 1000 if f > 0 else 0.0,
                    "iou_opt": iou_opt, "iou_bbox": iou_bbox, "dims_opt_mm": dims_opt, "dims_bbox_mm": dims_bbox})
        print(f"ruido x{f}: IoU opt {iou_opt:.3f} bbox {iou_bbox:.3f}, dims {dims_opt:.1f}/{dims_bbox:.1f} mm")
    return out


# #####################################################################################################
# Nível 4: três formas de usar a cor
# #####################################################################################################
def fit_each_color_group(r2, r3):
    """Separar por cor e depois ajustar CADA caixa sozinha: uma otimização por conjunto de pontos (os
    conjuntos da estimativa inicial, que no Nível 4 são os grupos de cor), sem cor na função objetivo."""
    seg = r2["seg"]
    params = []
    for k, init in enumerate(r3["init_params"]):
        mine = r3["box_labels"] == k
        sub = Segmentation(seg.points[mine], seg.colors[mine], np.zeros(mine.sum(), int), {0: 1}, {0: None})
        p, _, _ = fit_boxes(sub, [init], [None], use_color=False, verbose=False)
        params.append(p[0])
    return np.array(params)


def experiment_color(work):
    """Nível 4 com o desalinhamento cor/profundidade (misalignment_px) a aumentar: mais pontos junto às
    fronteiras ficam com a cor da caixa vizinha. 3 sementes do ruído por valor (o resultado de uma só cena
    varia muito de semente para semente)."""
    from savi_box_simulator import read_config, write_scene
    out = []
    for px, seed in [(px, seed) for px in (0.7, 3.0, 6.0, 10.0) for seed in range(3)]:
        config = read_config(FOLDER / "simulator" / "configs" / "level_4.json")
        config["noise"]["misalignment_px"] = px
        config["seed"] += seed
        folder = work / f"cor_{px}_{seed}"
        write_scene(config, folder)
        scene, gt = load_scene(folder), ground_truth(folder)
        T_true = true_camera_poses(scene, gt)
        _, r2, r3 = parts_2_3(scene, T_true, folder / "partes", use_color=True)
        _, r2n, r3n = parts_2_3(scene, T_true, folder / "partes_sem_cor", use_color=False)
        row = {"misalignment_px": px, "semente": seed,
               "cor_depois_bbox": boxes_iou(r3["params_bbox"], r2["table"], gt)[0],
               "cor_depois_ajuste_por_caixa": boxes_iou(fit_each_color_group(r2, r3), r2["table"], gt)[0],
               "cor_na_funcao_objetivo": boxes_iou(r3["params_opt"], r2["table"], gt)[0],
               "ajuste_conjunto_sem_cor": boxes_iou(r3n["params_opt"], r2n["table"], gt)[0]}
        out.append(row)
        print("cor", {k: round(v, 3) for k, v in row.items()}, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", required=True, help="pasta com as cenas level_1 e level_4 (com config.json)")
    parser.add_argument("--work", help="pasta de trabalho (cenas geradas e partes); por omissão <scenes>/_experiencias")
    parser.add_argument("--so", default="icp,cadeia,registo,ruido,cor", help="experiências a correr")
    args = parser.parse_args()
    scenes = Path(args.scenes)
    work = Path(args.work) if args.work else scenes / "_experiencias"
    RESULTS.mkdir(parents=True, exist_ok=True)
    experiments = {"icp": lambda: experiment_icp(scenes), "cadeia": lambda: experiment_chain(scenes),
                   "registo": lambda: experiment_registration(scenes, work), "ruido": lambda: experiment_noise(work),
                   "cor": lambda: experiment_color(work)}
    for name in args.so.split(","):
        t0 = time.time()
        result = experiments[name]()
        (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"== {name}: {time.time() - t0:.0f} s -> {RESULTS / (name + '.json')}")


if __name__ == "__main__":
    main()
