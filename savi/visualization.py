"""Visualizações: janelas 3D do Open3D (interativas) e figuras do matplotlib (guardadas em PNG).

As janelas só abrem com a opção --visualize do main.py; as figuras são sempre guardadas
na pasta de saída (servem para o README).
"""

import copy

import matplotlib
matplotlib.use("Agg")            # gera ficheiros PNG sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from .fitting import NUM_PARAMS, evolution_snapshots, normalize_box
from .geometry import box_corners_2d, box_lines_3d, box_T_table

CAMERA_COLORS = plt.cm.tab10(np.arange(10))[:, :3]
BOX_COLORS = [(0.9, 0.1, 0.1), (0.1, 0.4, 0.9), (0.1, 0.7, 0.2), (0.9, 0.6, 0.0), (0.6, 0.2, 0.8), (0.0, 0.7, 0.7)]


# --------------------------------------------------------------------------------------
# Open3D
# --------------------------------------------------------------------------------------
def box_lineset(T, length, width, height, color):
    """LineSet com as 12 arestas de uma caixa com pose T."""
    corners, edges = box_lines_3d(T, length, width, height)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(edges))
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def params_linesets(params, color=None):
    """LineSets das caixas (parâmetros no referencial da mesa)."""
    return [box_lineset(box_T_table(p[0], p[1], p[2], p[5]), p[3], p[4], p[5],
                        color or BOX_COLORS[j % len(BOX_COLORS)]) for j, p in enumerate(params)]


def show_registration(scene, T_est, camera_index_cloud=None, T_gt=None):
    """Nuvem agregada: cores reais | cor por câmara | (se houver GT) estimada a vermelho sobre a verdadeira a verde."""
    from .registration import aggregate
    cloud, cam = aggregate(scene, T_est)
    cloud = cloud.voxel_down_sample(0.005)
    o3d.visualization.draw_geometries([cloud], window_name="Nuvem agregada (cores reais)")

    by_camera = o3d.geometry.PointCloud()
    for k, (view, T) in enumerate(zip(scene.views, T_est)):
        c = copy.deepcopy(view.cloud).voxel_down_sample(0.005)
        c.transform(T)
        c.paint_uniform_color(CAMERA_COLORS[k % 10])
        by_camera += c
    o3d.visualization.draw_geometries([by_camera], window_name="Nuvem agregada (cor = câmara)")

    if T_gt is not None:
        est, _ = aggregate(scene, T_est)
        gt, _ = aggregate(scene, T_gt)
        est = est.voxel_down_sample(0.005)
        gt = gt.voxel_down_sample(0.005)
        est.paint_uniform_color((0.9, 0.1, 0.1))
        gt.paint_uniform_color((0.1, 0.7, 0.2))
        o3d.visualization.draw_geometries([gt, est], window_name="Estimada (vermelho) vs verdadeira (verde)")


def show_segmentation(seg, labels):
    """Pontos dos objetos coloridos pela caixa a que foram atribuídos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg.points))
    pc.colors = o3d.utility.Vector3dVector([BOX_COLORS[l % len(BOX_COLORS)] if l >= 0 else (0.5, 0.5, 0.5)
                                            for l in labels])
    o3d.visualization.draw_geometries([pc], window_name="Segmentação (cor = caixa)")


def animate_optimization(points, colors, problem, delay=0.15):
    """Mostra a evolução das caixas ao longo da otimização, sobre a nuvem de pontos."""
    import time
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Evolução da otimização")
    vis.add_geometry(pc)
    current = []
    for x in evolution_snapshots(problem.history):
        for g in current:
            vis.remove_geometry(g, reset_bounding_box=False)
        current = params_linesets(x.reshape(-1, NUM_PARAMS))
        for g in current:
            vis.add_geometry(g, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)
    vis.run()                    # fica aberta no estado final até o utilizador fechar
    vis.destroy_window()


def show_final(points, colors, params_opt, params_bbox=None, gt_params=None):
    """Caixas finais: otimização (cores por caixa), bounding box (cinzento), verdadeiras (preto)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    geoms = [pc] + params_linesets(params_opt)
    if params_bbox is not None:
        geoms += params_linesets(params_bbox, (0.5, 0.5, 0.5))
    if gt_params is not None:
        geoms += params_linesets(gt_params, (0.0, 0.0, 0.0))
    o3d.visualization.draw_geometries(geoms, window_name="Caixas finais")


# --------------------------------------------------------------------------------------
# Figuras (matplotlib, vista de cima no referencial da mesa)
# --------------------------------------------------------------------------------------
def _draw_rects(ax, params, style, label=None):
    for j, p in enumerate(params):
        c = box_corners_2d(p[0], p[1], p[2], p[3], p[4])
        c = np.vstack([c, c[:1]])
        color = style.get("color") or BOX_COLORS[j % len(BOX_COLORS)]
        ax.plot(c[:, 0], c[:, 1], color=color, lw=style.get("lw", 1.5), ls=style.get("ls", "-"),
                label=label if j == 0 else None)


def save_top_view(path, points, colors, params_opt, params_bbox=None, gt_params=None, title=""):
    """Vista de cima: pontos + caixas (otimização, bounding box e, se houver, verdadeiras)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sel = np.random.default_rng(0).choice(len(points), min(len(points), 15000), replace=False)
    ax.scatter(points[sel, 0], points[sel, 1], c=colors[sel], s=1)
    if gt_params is not None:
        _draw_rects(ax, gt_params, {"color": "k", "lw": 2.5}, "verdadeira")
    if params_bbox is not None:
        _draw_rects(ax, params_bbox, {"color": "gray", "ls": "--"}, "bounding box")
    _draw_rects(ax, params_opt, {}, "otimização")
    ax.set_aspect("equal")
    ax.set_xlabel("x mesa (m)")
    ax.set_ylabel("y mesa (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_evolution(path, points, problem, num_frames=6):
    """Várias fases da otimização lado a lado (da estimativa inicial ao resultado final)."""
    snaps = evolution_snapshots(problem.history)
    picks = np.unique(np.linspace(0, len(snaps) - 1, min(num_frames, len(snaps))).astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4), squeeze=False)
    sel = np.random.default_rng(0).choice(len(points), min(len(points), 6000), replace=False)
    for ax, k in zip(axes[0], picks):
        ax.scatter(points[sel, 0], points[sel, 1], c="0.6", s=0.5)
        params = [normalize_box(p) for p in snaps[k].reshape(-1, NUM_PARAMS)]
        _draw_rects(ax, params, {})
        ax.set_aspect("equal")
        ax.set_title(f"passo {k}/{len(snaps) - 1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Evolução das caixas durante a otimização (vista de cima)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
