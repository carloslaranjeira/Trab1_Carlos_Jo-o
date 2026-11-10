#!/usr/bin/env python3
"""SAVIBoxes - Estimação da pose e das dimensões de caixas sobre uma mesa, a partir de várias imagens RGB-D.

Utilização:
    python main.py --scene simulator/scenes/level_1 --num_boxes 1
    python main.py --scene simulator/scenes/level_4 --num_boxes 3 --evaluate --visualize

Escreve o ficheiro result.json (por omissão dentro da pasta da cena) com a pose de cada câmara
e a pose e as dimensões de cada caixa, tudo no referencial da câmara 000.

Pipeline:
    Parte 1  registo         -> T_camera0_camera de cada imagem e nuvem agregada
    Parte 2  segmentação     -> mesa (RANSAC), pontos acima da mesa, clusters e cor
    Parte 3  ajuste          -> estimativa inicial + otimização (least_squares); bounding box para comparação
"""

import argparse
import time
from pathlib import Path

import numpy as np

from savi import evaluation, visualization
from savi.bbox_method import bbox_method
from savi.fitting import fit_boxes, initial_boxes
from savi.geometry import yaw_of
from savi.registration import aggregate, register_scene
from savi.scene_io import load_scene, write_result
from savi.segmentation import detect_table, segment


def gt_params_in_table(ground_truth, T_camera0_table):
    """Caixas verdadeiras no referencial da mesa estimada (só para desenhar nas figuras)."""
    T_table_camera0 = np.linalg.inv(T_camera0_table)
    params = []
    for b in ground_truth["boxes"]:
        T = T_table_camera0 @ np.array(b["T_camera0_box"])
        params.append([T[0, 3], T[1, 3], yaw_of(T[:3, :3]), b["length"], b["width"], b["height"]])
    return np.array(params)


def run(scene_folder, num_boxes, use_color=True, loop_closure=True, gt_registration=False, verbose=True):
    """Corre o sistema completo numa cena e devolve um dicionário com todos os resultados intermédios.

    gt_registration=True usa as poses VERDADEIRAS das câmaras em vez do registo. Só serve para
    avaliar a segmentação e o ajuste isoladamente (como pede o enunciado); nunca no resultado final.
    """
    t0 = time.time()
    scene = load_scene(scene_folder)
    if verbose:
        print(f"Cena {scene.folder.name}: {len(scene.views)} imagens, {num_boxes} caixas")

    # ---------------- Parte 1: registo ----------------
    if gt_registration:
        T_cams = [np.array(c["T_camera0_camera"]) for c in scene.ground_truth["cameras"]]
        T_chain = T_cams
    else:
        T_cams, T_chain = register_scene(scene, loop_closure=loop_closure, verbose=verbose)
    cloud, _ = aggregate(scene, T_cams)

    # ---------------- Parte 2: segmentação ----------------
    table = detect_table(cloud, T_cams, verbose=verbose)
    seg = segment(cloud, table, num_boxes, use_color=use_color, verbose=verbose)

    # ---------------- Parte 3: ajuste ----------------
    init_params, init_colors, box_labels = initial_boxes(seg)
    params_bbox = bbox_method(seg.points, box_labels, len(init_params))
    params_opt, problem, fit_points, fit_assoc = fit_boxes(seg, init_params, init_colors,
                                                           use_color=use_color, verbose=verbose)
    if verbose:
        print(f"  tempo total: {time.time() - t0:.1f} s")

    return {"scene": scene, "T_cams": T_cams, "T_chain": T_chain, "cloud": cloud, "table": table, "seg": seg,
            "init_params": np.array(init_params), "box_labels": box_labels,
            "params_bbox": params_bbox, "params_opt": params_opt, "problem": problem,
            "fit_points": fit_points, "fit_assoc": fit_assoc}


def print_evaluation(r):
    """Mostra no terminal as tabelas de erros (só se a cena tiver ground_truth.json)."""
    gt = r["scene"].ground_truth
    print("\nRegisto (erro de cada câmara):")
    print(f"  {'imagem':8s}{'transl. (mm)':>14s}{'rot. (°)':>10s}")
    for name, t, a in evaluation.camera_errors(r["T_cams"], gt):
        print(f"  {name:8s}{t:14.2f}{a:10.3f}")

    n_sets, n_gt, seg_rows = evaluation.evaluate_segmentation(
        r["seg"].points, r["box_labels"], r["table"].T_camera0_table, np.asarray(r["cloud"].points), gt)
    print(f"\nSegmentação: {n_sets} conjuntos para {n_gt} caixas")
    for row in seg_rows:
        print(f"  caixa {row['gt_box']}: precisão {row['precision']:.3f}, recall {row['recall']:.3f}")

    T_ct = r["table"].T_camera0_table
    for name in ("bbox", "opt"):
        rows = evaluation.evaluate_boxes(evaluation.boxes_to_camera0(r[f"params_{name}"], T_ct), gt)
        print(f"\nCaixas - {'bounding box' if name == 'bbox' else 'otimização'}:")
        print(f"  {'caixa':6s}{'pos (mm)':>10s}{'yaw (°)':>9s}{'dL (mm)':>9s}{'dW (mm)':>9s}{'dH (mm)':>9s}{'IoU':>7s}")
        for j, e in enumerate(rows):
            print(f"  {j:<6d}{e['pos_mm']:10.1f}{e['yaw_deg']:9.2f}{e['dl_mm']:9.1f}{e['dw_mm']:9.1f}"
                  f"{e['dh_mm']:9.1f}{e['iou']:7.3f}")
        print(f"  IoU médio: {np.mean([e['iou'] for e in rows]):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena")
    parser.add_argument("--num_boxes", type=int, required=True, help="número de caixas na mesa")
    parser.add_argument("--output", help="ficheiro de saída (por omissão: <cena>/result.json)")
    parser.add_argument("--method", choices=["opt", "bbox"], default="opt",
                        help="método usado no result.json: otimização (opt) ou bounding box (bbox)")
    parser.add_argument("--no_color", action="store_true", help="não usar a cor (segmentação nem otimização)")
    parser.add_argument("--no_loop_closure", action="store_true", help="registo só por encadeamento")
    parser.add_argument("--evaluate", action="store_true", help="compara com o ground_truth.json, se existir")
    parser.add_argument("--visualize", action="store_true", help="abre janelas 3D com cada etapa")
    args = parser.parse_args()

    r = run(args.scene, args.num_boxes, use_color=not args.no_color, loop_closure=not args.no_loop_closure)

    # Escrita do result.json
    params = r["params_opt"] if args.method == "opt" else r["params_bbox"]
    boxes = evaluation.boxes_to_camera0(params, r["table"].T_camera0_table)
    output = Path(args.output) if args.output else Path(args.scene) / "result.json"
    write_result(output, [v.name for v in r["scene"].views], r["T_cams"], boxes)
    print(f"\nResultado escrito em {output}")

    # Figuras (vista de cima e evolução da otimização), ao lado do result.json
    gt = r["scene"].ground_truth
    gt_params = gt_params_in_table(gt, r["table"].T_camera0_table) if gt else None
    visualization.save_top_view(output.parent / "top_view.png", r["seg"].points, r["seg"].colors,
                                r["params_opt"], r["params_bbox"], gt_params, title=Path(args.scene).name)
    visualization.save_evolution(output.parent / "optimization.png", r["fit_points"], r["problem"])

    if args.evaluate and gt:
        print_evaluation(r)

    if args.visualize:
        T_gt = [np.array(c["T_camera0_camera"]) for c in gt["cameras"]] if gt else None
        visualization.show_registration(r["scene"], r["T_cams"], T_gt=T_gt)
        visualization.show_segmentation(r["seg"], r["box_labels"])
        visualization.animate_optimization(r["fit_points"], np.full_like(r["fit_points"], 0.6), r["problem"])
        visualization.show_final(r["seg"].points, r["seg"].colors, r["params_opt"], r["params_bbox"], gt_params)


if __name__ == "__main__":
    main()
