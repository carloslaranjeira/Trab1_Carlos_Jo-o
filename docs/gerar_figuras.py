#!/usr/bin/env python3
"""Figuras e gráficos do README.md (gravados em docs/).

    python docs/gerar_figuras.py --results <pasta(s) do run_test.py> [--fotometria <cena>]

--results: uma ou mais pastas escritas pelo run_test.py (uma subpasta por cena, com result.json e partes/);
           as cenas são procuradas pelo nome em todas elas.
Os gráficos das experiências leem docs/resultados/*.json (docs/experiencias.py).
As nuvens de pontos são desenhadas com o draw_points do simulator/generate_figures.py (o mesmo código
que fez as figuras do enunciado), num referencial do mundo; a pose verdadeira da câmara 000 só serve
para pôr a nuvem (que o sistema estima no referencial da câmara 000) de pé nas figuras.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent
DOCS = FOLDER / "docs"
RESULTS = DOCS / "resultados"
sys.path.insert(0, str(FOLDER / "savi"))
sys.path.insert(0, str(FOLDER / "simulator"))

from generate_figures import close_view, draw_points, look_at         # noqa: E402  (código do professor)
from avaliador import (                                                # noqa: E402
    box_in_world, boxes_in_world, evaluate_boxes, bbox_boxes, true_camera_poses, true_exposures,
    world_params,
)
from comum import load_scene, OUTPUT_FOLDER, transform_points           # noqa: E402
from extra_fotometria import figure_data as photometry_figure   # noqa: E402
from parte1_registo import aggregate, load_registration                  # noqa: E402
from parte2_segmentacao import load_objects, load_segmentation           # noqa: E402
from parte3_ajuste import draw_rects, load_fit, NUM_PARAMS               # noqa: E402
from visualizacao import CAMERA_COLORS                                   # noqa: E402

# Cores dos gráficos: paleta categórica por ordem fixa (1 azul, 2 laranja, 3 verde-água, 4 amarelo)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "legend.frameon": False, "savefig.dpi": 120, "savefig.bbox": "tight"})

ORDER = ["level_1", "level_2", "level_3_heights", "level_3_widths", "level_3_rotated", "level_3_aligned",
         "level_4", "stack_1", "stack_2", "stack_3"]
CLOUD_VIEW = look_at((1.25, -1.45, 1.05), (-0.05, 0.1, 0.0))      # a mesma vista das figuras do enunciado
CLOSE_VIEW = look_at((0.55, -0.65, 0.5), (0.0, 0.0, 0.08))        # perto das caixas


def save(fig, name):
    fig.savefig(DOCS / name)
    plt.close(fig)
    print(name)


def find(results, name):
    for folder in results:
        if (folder / name / "result.json").exists():
            return folder / name
    return None


def scene_of(out):
    r1 = load_registration(out / OUTPUT_FOLDER)
    folder = Path(r1["scene_folder"])
    return r1, load_scene(folder), json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))


def world(gt):
    return np.array(gt["cameras"][0]["T_world_camera"])


def render(points, colors, view=CLOUD_VIEW, size=(1000, 700), f=900, voxel=0.008):
    return np.clip(draw_points(points, colors, view, f, size[0], size[1], voxel=voxel), 0, 1)


def grid(images, titles, columns, name, width=12.0):
    rows = int(np.ceil(len(images) / columns))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, columns, figsize=(width, (width / columns) * h / w * rows + 0.5 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image)
        ax.set_title(title, color=INK)
    fig.tight_layout(pad=0.4)
    save(fig, name)


# #####################################################################################################
# Cenas
# #####################################################################################################
def figure_scenes():
    names = {"level_1": "Nível 1", "level_2": "Nível 2", "level_3_heights": "Nível 3: alturas",
             "level_3_widths": "Nível 3: larguras", "level_3_rotated": "Nível 3: rodadas",
             "level_3_aligned": "Nível 3: alinhadas", "level_4": "Nível 4", "stack_2": "Pilha (stack_2)",
             "stack_3": "Pilha de 3 (stack_3)"}
    grid([close_view(n) for n in names], list(names.values()), 3, "cenas.png")


# #####################################################################################################
# Parte 1
# #####################################################################################################
def figure_registration(out):
    r1, scene, gt = scene_of(out)
    cloud, cam = aggregate(scene, r1["T_cams"], r1["registered"])
    T = world(gt)
    points = transform_points(T, np.asarray(cloud.points))
    colors = np.asarray(cloud.colors)
    grid([render(points, colors), render(points, CAMERA_COLORS[cam % len(CAMERA_COLORS)])],
         ["Nuvem agregada (poses estimadas), cor dos pixels", "A mesma nuvem, uma cor por câmara"], 2,
         "parte1_nuvem_agregada.png")
    # Nuvem estimada (vermelho) sobre a verdadeira (verde), perto das caixas
    true_cloud, _ = aggregate(scene, true_camera_poses(scene, gt))
    est = cloud.voxel_down_sample(0.003)
    tru = true_cloud.voxel_down_sample(0.003)
    p = np.vstack([transform_points(T, np.asarray(tru.points)), transform_points(T, np.asarray(est.points))])
    c = np.vstack([np.tile([0.1, 0.7, 0.2], (len(tru.points), 1)), np.tile([0.9, 0.1, 0.1], (len(est.points), 1))])
    order = np.random.default_rng(0).permutation(len(p))
    grid([render(p[order], c[order], CLOSE_VIEW, voxel=0.003)],
         ["Poses estimadas (vermelho) sobre as verdadeiras (verde)"], 1, "parte1_sobreposicao.png", width=8)


def figure_chain():
    data = json.loads((RESULTS / "cadeia.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, len(data), figsize=(11, 3.6), sharey=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), data.items()):
        for k, (label, text) in enumerate((("encadeamento", "só encadeamento"),
                                          ("grafo_de_poses", "fecho de ciclo + grafo de poses"))):
            cams = [r["camera"] for r in rows[label]]
            ax.plot(cams, [r["transl_mm"] for r in rows[label]], "-o", color=SERIES[k], lw=2, ms=6, label=text)
        ax.set_title(name)
        ax.set_xlabel("câmara (ordem do encadeamento)")
        ax.set_xticks(cams)
    np.atleast_1d(axes)[0].set_ylabel("erro de translação (mm)")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    save(fig, "parte1_encadeamento.png")


def figure_registration_effect():
    data = json.loads((RESULTS / "registo.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (name, rows) in enumerate(data.items()):
        x = list(range(len(rows)))           # níveis de erro igualmente espaçados (os valores estão nas etiquetas)
        axes[0].plot(x, [r["iou_opt"] for r in rows], "-o", color=SERIES[k], lw=2, ms=6, label=f"{name}, otimização")
        axes[0].plot(x, [r["iou_bbox"] for r in rows], "--s", color=SERIES[k], lw=2, ms=6,
                     label=f"{name}, bounding box")
        axes[1].plot(x, [r["dims_opt_mm"] for r in rows], "-o", color=SERIES[k], lw=2, ms=6, label=f"{name}, otimização")
        axes[1].plot(x, [r["dims_bbox_mm"] for r in rows], "--s", color=SERIES[k], lw=2, ms=6,
                     label=f"{name}, bounding box")
    labels = [f"{r['transl_mm']:.0f} mm\n{r['rot_deg']:.1f}°" for r in rows]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("erro acrescentado à pose de cada câmara")
    axes[0].set_ylabel("IoU 3D médio")
    axes[1].set_ylabel("erro médio das dimensões (mm)")
    axes[1].legend(fontsize=8)
    save(fig, "parte1_efeito_registo.png")


# #####################################################################################################
# Parte 2
# #####################################################################################################
GROUP_COLORS = np.array([[0.9, 0.1, 0.1], [0.1, 0.4, 0.9], [0.1, 0.7, 0.2], [0.95, 0.6, 0.0], [0.6, 0.2, 0.8],
                         [0.0, 0.7, 0.7]])


def segment_colors(seg):
    """Uma cor por conjunto (cluster e, nos clusters separados pela cor, grupo de cor)."""
    labels = np.zeros(len(seg.points), int)
    n = 0
    for k in sorted(seg.boxes_per_cluster):
        mine = np.flatnonzero(seg.clusters == k)
        groups = seg.color_groups.get(k)
        groups = np.zeros(len(mine), int) if groups is None else groups
        for g in np.unique(groups):
            labels[mine[groups == g]] = n
            n += 1
    return GROUP_COLORS[labels % len(GROUP_COLORS)]


def figure_segmentation(outs):
    images, titles = [], []
    for out in outs:
        r1, scene, gt = scene_of(out)
        cloud, _ = aggregate(scene, r1["T_cams"], r1["registered"])
        table, _, _ = load_objects(out / OUTPUT_FOLDER)
        seg = load_segmentation(out / OUTPUT_FOLDER)
        T = world(gt)
        rest = transform_points(T, np.asarray(cloud.voxel_down_sample(0.006).points))
        objects = transform_points(T @ table.T_camera0_table, seg.points)
        gray = np.full((len(rest), 3), 0.78)
        images.append(render(np.vstack([rest, objects]), np.vstack([gray, segment_colors(seg)]), voxel=0.006))
        titles.append(f"{out.name}: objetos isolados (uma cor por conjunto)")
    grid(images, titles, 2, "parte2_segmentacao.png")


# #####################################################################################################
# Parte 3
# #####################################################################################################
def top_views(outs, name, columns=3):
    n = len(outs)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, out in zip(axes.ravel(), outs):
        r1, scene, gt = scene_of(out)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        table, _, _ = load_objects(out / OUTPUT_FOLDER)
        seg = load_segmentation(out / OUTPUT_FOLDER)
        T_world_table = world(gt) @ table.T_camera0_table
        pts = transform_points(T_world_table, seg.points)
        sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 12000), replace=False)
        ax.set_axis_on()
        ax.scatter(pts[sel, 0], pts[sel, 1], c=seg.colors[sel], s=1)
        gt_params = np.array([world_params(box_in_world(b["T_world_box"], b["length"], b["width"], b["height"]))
                              for b in gt["boxes"]])
        draw_rects(ax, gt_params, {"color": "k", "lw": 3.0}, "ground truth")
        bbox = bbox_boxes(out / OUTPUT_FOLDER)
        if bbox is not None:
            draw_rects(ax, np.asarray(boxes_in_world(bbox, gt)), {"color": "gray", "ls": "--"}, "bounding box")
        draw_rects(ax, np.asarray(boxes_in_world(result["boxes"], gt)), {}, "otimização")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(out.name)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    save(fig, name)


def figure_iou(outs):
    names, opt, bbox = [], [], []
    for out in outs:
        _, _, gt = scene_of(out)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        names.append(out.name)
        opt.append(np.mean([r["iou"] for r in evaluate_boxes(result["boxes"], gt)[0]]))
        b = bbox_boxes(out / OUTPUT_FOLDER)
        bbox.append(np.mean([r["iou"] for r in evaluate_boxes(b, gt)[0]]) if b else 0.0)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, opt, 0.38, color=SERIES[0], label="otimização")
    ax.bar(x + 0.2, bbox, 0.38, color=SERIES[1], label="bounding box")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("IoU 3D médio")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "parte3_iou.png")


def figure_evolution(out):
    """Evolução da otimização: as caixas em 4 passos (vistas de cima) e o custo em cada passo guardado."""
    fit = load_fit(out / OUTPUT_FOLDER)
    seg = load_segmentation(out / OUTPUT_FOLDER)
    snaps, costs = np.asarray(fit["snapshots"]), np.asarray(fit["snapshot_costs"])
    # só as iterações (como o snapshot_frames): nas avaliações do jacobiano muda um só parâmetro
    iterations = [0]
    for k in range(1, len(snaps)):
        if np.sum(snaps[k] != snaps[iterations[-1]]) > 1:
            iterations.append(k)
    if iterations[-1] != len(snaps) - 1:
        iterations.append(len(snaps) - 1)
    picks = sorted(set(np.linspace(0, len(iterations) - 1, 4).round().astype(int)))
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(4.0 * (len(picks) + 1), 4.0))
    sel = np.random.default_rng(0).choice(len(seg.points), min(len(seg.points), 10000), replace=False)
    for ax, i in zip(axes, picks):
        ax.scatter(seg.points[sel, 0], seg.points[sel, 1], c=seg.colors[sel], s=1)
        draw_rects(ax, snaps[iterations[i]].reshape(-1, NUM_PARAMS), {})
        ax.set_aspect("equal")
        ax.set_title("estimativa inicial" if i == 0 else f"iteração {i} (custo {costs[iterations[i]]:.0f})")
        ax.tick_params(labelsize=7)
    axes[-1].plot(range(len(iterations)), costs[iterations], "-o", color=SERIES[0], lw=2, ms=4)
    axes[-1].set_xlabel("iteração")
    axes[-1].set_ylabel("custo")
    axes[-1].set_title("custo ao longo da otimização")
    fig.suptitle(f"Evolução da otimização ({out.name}, referencial da mesa)")
    fig.tight_layout()
    save(fig, "parte3_evolucao.png")


# #####################################################################################################
# Níveis
# #####################################################################################################
def figure_noise():
    rows = json.loads((RESULTS / "ruido.json").read_text(encoding="utf-8"))
    x = [r["fator"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (key, label) in enumerate((("opt", "otimização"), ("bbox", "bounding box"))):
        axes[0].plot(x, [r[f"iou_{key}"] for r in rows], "-o", color=SERIES[k], lw=2, ms=6, label=label)
        axes[1].plot(x, [r[f"dims_{key}_mm"] for r in rows], "-o", color=SERIES[k], lw=2, ms=6, label=label)
    for ax in axes:
        ax.set_xlabel("ruído de profundidade (× o do simulador)")
        ax.set_xticks(x)
    axes[0].set_ylabel("IoU 3D")
    axes[1].set_ylabel("erro médio das dimensões (mm)")
    axes[0].legend()
    fig.suptitle("Nível 1 com cada vez mais ruído (depth_std e flying_pixels; poses verdadeiras)")
    save(fig, "nivel1_ruido.png")


def figure_color():
    rows = json.loads((RESULTS / "cor.json").read_text(encoding="utf-8"))
    x = sorted({r["misalignment_px"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (key, label) in enumerate((("cor_depois_bbox", "separar por cor + bounding box"),
                                      ("cor_depois_ajuste_por_caixa", "separar por cor + ajustar cada caixa"),
                                      ("cor_na_funcao_objetivo", "cor na função objetivo"),
                                      ("ajuste_conjunto_sem_cor", "ajuste conjunto sem cor"))):
        # linha: média das sementes; pontos pequenos: cada semente
        values = [[r[key] for r in rows if r["misalignment_px"] == px] for px in x]
        ax.plot(x, [np.mean(v) for v in values], "-o", color=SERIES[k], lw=2, ms=6, label=label)
        for px, v in zip(x, values):
            ax.plot([px + 0.12 * (k - 1.5)] * len(v), v, ".", color=SERIES[k], ms=5, alpha=0.6)
    ax.set_xlabel("desalinhamento cor/profundidade (pixels): mais pontos com a cor da caixa vizinha")
    ax.set_ylabel("IoU 3D médio (linha: média de 3 sementes)")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.set_title("Nível 4: formas de usar a cor (poses verdadeiras)")
    save(fig, "nivel4_cor.png")


# #####################################################################################################
# Extra: fotometria
# #####################################################################################################
def figure_photometry(outs):
    out = outs[0]
    r1 = load_registration(out / OUTPUT_FOLDER)
    rx = json.loads((out / OUTPUT_FOLDER / "fotometria.json").read_text(encoding="utf-8"))
    data = photometry_figure(r1, rx)
    plt.imsave(DOCS / "extra_mosaico.png", data["images"])
    print("extra_mosaico.png")
    # s e b estimados contra os verdadeiros, de todas as cenas
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for k, out in enumerate(outs):
        _, _, gt = scene_of(out)
        rows = json.loads((out / OUTPUT_FOLDER / "fotometria.json").read_text(encoding="utf-8"))["imagens"]
        truth = true_exposures(gt)
        for ax, key, i in ((axes[0], "s", 0), (axes[1], "b", 1)):
            # marcadores ocos e de formas diferentes: as cenas têm a mesma semente, logo o mesmo s e b verdadeiros
            ax.plot([truth[r["imagem"]][i] for r in rows], [r[key] for r in rows], "os^"[k % 3], ms=9 - 2 * k,
                    markerfacecolor="none", markeredgewidth=2, color=SERIES[k % 4], label=out.name)
    for ax, key in zip(axes, ("s", "b")):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", label="estimado = verdadeiro")
        ax.set_xlabel(f"{key} verdadeiro (ground truth, em relação à 000)")
        ax.set_ylabel(f"{key} estimado")
        ax.set_aspect("equal")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "extra_s_b.png")


def figure_photometry_segmentation(out):
    """Conjuntos da Parte 2 sem correção e com a correção estimada (vista de cima, referencial da mesa)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, (k, title) in zip(axes, ((0, "sem correção"), (1, "com a correção estimada"))):
        folder = out / "partes_gt_fotometria" / f"versao_{k}"
        seg = load_segmentation(folder)
        sel = np.random.default_rng(0).choice(len(seg.points), min(len(seg.points), 12000), replace=False)
        ax.scatter(seg.points[sel, 0], seg.points[sel, 1], c=segment_colors(seg)[sel], s=1)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {len(np.unique(segment_colors(seg), axis=0))} conjuntos")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{out.name}: segmentação por cor (uma cor por conjunto)")
    fig.tight_layout()
    save(fig, "extra_segmentacao.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="pastas escritas pelo run_test.py")
    args = parser.parse_args()
    results = [Path(p) for p in args.results]
    outs = [find(results, n) for n in ORDER]
    outs = [o for o in outs if o is not None]

    figure_scenes()
    figure_registration(find(results, "level_4"))
    figure_segmentation([find(results, "level_2"), find(results, "level_4")])
    top_views(outs, "parte3_vistas_de_cima.png")
    figure_iou(outs)
    figure_evolution(find(results, "level_3_widths"))
    for name, fn in (("cadeia", figure_chain), ("registo", figure_registration_effect), ("ruido", figure_noise),
                     ("cor", figure_color)):
        if (RESULTS / f"{name}.json").exists():
            fn()
    photo = [find(results, n) for n in ("level_4_cores_proximas_exposure", "level_4_exposure", "stack_2_exposure")]
    photo = [o for o in photo if o is not None]
    if photo:
        figure_photometry(photo)
        figure_photometry_segmentation(photo[0])


if __name__ == "__main__":
    main()
