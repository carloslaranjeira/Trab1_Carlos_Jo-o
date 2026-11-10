#!/usr/bin/env python3
"""Avaliação em várias cenas: gera as tabelas de erros dos dois métodos (para o README).

Utilização (as cenas têm de ter ground_truth.json; o número de caixas é lido do config.json,
o que é permitido porque só serve para avaliar):
    python evaluate.py simulator/scenes/level_*
    python evaluate.py simulator/scenes/level_3_* --gt_registration     # isola segmentação + ajuste

Escreve results/results.md (tabelas em Markdown) e uma figura por cena em results/.
As cenas são processadas uma de cada vez.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import main
from savi import evaluation, visualization


def fmt(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def main_evaluate():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+", help="pastas de cenas (aceita wildcards)")
    parser.add_argument("--gt_registration", action="store_true",
                        help="usa as poses verdadeiras das câmaras (avalia só segmentação e ajuste)")
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    folders = sorted({f for pattern in args.scenes for f in glob.glob(pattern) if Path(f, "ground_truth.json").exists()})
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    reg_lines = ["| cena | erro transl. médio (mm) | máx (mm) | erro rot. médio (°) | máx (°) | só encadeamento: transl. máx (mm) |",
                 "|---|---|---|---|---|---|"]
    seg_lines = ["| cena | conjuntos / caixas | precisão média | recall médio |", "|---|---|---|---|"]
    box_lines = ["| cena | caixa | método | pos (mm) | yaw (°) | dL (mm) | dW (mm) | dH (mm) | IoU |",
                 "|---|---|---|---|---|---|---|---|---|"]
    summary = ["| cena | IoU médio bounding box | IoU médio otimização |", "|---|---|---|"]

    for folder in folders:
        num_boxes = len(json.loads(Path(folder, "config.json").read_text(encoding="utf-8"))["boxes"])
        name = Path(folder).name
        r = main.run(folder, num_boxes, use_color=not args.no_color, gt_registration=args.gt_registration)
        gt = r["scene"].ground_truth

        if not args.gt_registration:
            cams = evaluation.camera_errors(r["T_cams"], gt)
            chain = evaluation.camera_errors(r["T_chain"], gt)
            t, a = np.array([c[1] for c in cams]), np.array([c[2] for c in cams])
            reg_lines.append(f"| {name} | {t[1:].mean():.2f} | {t.max():.2f} | {a[1:].mean():.3f} | {a.max():.3f} | "
                             f"{max(c[1] for c in chain):.2f} |")

        n_sets, n_gt, seg_rows = evaluation.evaluate_segmentation(
            r["seg"].points, r["box_labels"], r["table"].T_camera0_table, np.asarray(r["cloud"].points), gt)
        seg_lines.append(f"| {name} | {n_sets} / {n_gt} | {np.mean([s['precision'] for s in seg_rows]):.3f} | "
                         f"{np.mean([s['recall'] for s in seg_rows]):.3f} |")

        ious = {}
        for method, label in (("bbox", "bounding box"), ("opt", "otimização")):
            rows = evaluation.evaluate_boxes(evaluation.boxes_to_camera0(r[f"params_{method}"], r["table"].T_camera0_table), gt)
            ious[method] = np.mean([e["iou"] for e in rows])
            for j, e in enumerate(rows):
                box_lines.append(f"| {name} | {j} | {label} | {fmt(e['pos_mm'])} | {fmt(e['yaw_deg'], 2)} | "
                                 f"{fmt(e['dl_mm'])} | {fmt(e['dw_mm'])} | {fmt(e['dh_mm'])} | {e['iou']:.3f} |")
        summary.append(f"| {name} | {ious['bbox']:.3f} | {ious['opt']:.3f} |")

        gt_params = main.gt_params_in_table(gt, r["table"].T_camera0_table)
        visualization.save_top_view(out / f"{name}_top_view.png", r["seg"].points, r["seg"].colors,
                                    r["params_opt"], r["params_bbox"], gt_params, title=name)
        visualization.save_evolution(out / f"{name}_optimization.png", r["fit_points"], r["problem"])

    text = ["# Resultados", "",
            f"Registo: {'poses verdadeiras (só segmentação e ajuste)' if args.gt_registration else 'estimado'}", ""]
    if not args.gt_registration:
        text += ["## Registo", ""] + reg_lines + [""]
    text += ["## Segmentação", ""] + seg_lines + [""]
    text += ["## Resumo (IoU 3D médio)", ""] + summary + [""]
    text += ["## Erros por caixa", ""] + box_lines + [""]
    (out / "results.md").write_text("\n".join(text), encoding="utf-8")
    print(f"\nTabelas escritas em {out / 'results.md'}")


if __name__ == "__main__":
    main_evaluate()
