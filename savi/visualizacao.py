#!/usr/bin/env python3
"""SAVIBoxes - VISUALIZAÇÃO comum a todas as partes (não pertence a nenhuma parte em particular).

As janelas são janelas 3D do Open3D (as do draw_geometries), como a figura da nuvem agregada do
enunciado. Cada parte tem duas funções próprias, que usam as funções deste ficheiro:
    figure_data(...)   -> dicionário só com arrays (pode ser gravado num .npz e desenhado noutro processo:
                          é assim que o run_test.py abre as janelas sem parar o programa);
    make_windows(data) -> lista de janelas, cada uma um dicionário com o título, o que desenhar
                          (points_item, lines_item, box_item, frame_item, ou uma imagem 2D:
                          image_item) e o ponto de vista.
SceneWindows abre as janelas (ou, sem janelas, só grava cada uma num PNG).
"""

import os
import time

import cv2                                   # OpenCV: gravar os PNG
import numpy as np
import open3d as o3d                         # janelas 3D

from comum import box_lines_3d, box_T_table          # para desenhar as caixas (box_item)


# #####################################################################################################
# 1. CORES, PONTOS DE VISTA E ELEMENTOS A DESENHAR
# #####################################################################################################
# Uma cor por câmara (as 10 cores 'tab10' do matplotlib)
CAMERA_COLORS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                          [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5],
                          [0.74, 0.74, 0.13], [0.09, 0.75, 0.81]])
CAMERA_NAMES = ["azul", "laranja", "verde", "vermelho", "roxo", "castanho", "rosa", "cinzento", "amarelo", "ciano"]


# Uma janela por cena, com uma vista de cada vez (as setas <- e -> mudam de vista).
WINDOW_SIZE = (1100, 750)     # pixels de cada janela (e de cada PNG)
POINT_SIZE = 2.0              # tamanho de cada ponto desenhado (pixels)
WHITE = (1.0, 1.0, 1.0)       # cor do fundo
FRAME_SECONDS = 0.15          # duração de cada passo das animações
REPEAT_SECONDS = 2.0          # pausa no fim de uma animação, antes de recomeçar


def points_item(points, colors, size=POINT_SIZE):
    """Uma nuvem de pontos a desenhar (cores RGB entre 0 e 1, ou uint8 entre 0 e 255)."""
    return {"type": "points", "points": np.asarray(points, np.float32), "colors": np.asarray(colors), "size": size}


def lines_item(points, lines, color, width=3.0):
    """Segmentos de reta a desenhar (pares de índices em points), todos com a mesma cor."""
    return {"type": "lines", "points": np.asarray(points, float), "lines": np.asarray(lines, int),
            "color": tuple(color), "width": width}


def box_item(p, color, width=3.0):
    """As 12 arestas de uma caixa [x, y, yaw, length, width, height, z] (referencial da mesa)."""
    corners, edges = box_lines_3d(box_T_table(p[0], p[1], p[2], p[5], p[6]), p[3], p[4], p[5])
    return lines_item(corners, edges, color, width)


def image_item(image):
    """Uma imagem RGB (uint8) mostrada em 2D na janela, em vez de uma cena 3D (ex.: um gráfico do
    matplotlib). Uma vista com uma imagem não tem ponto de vista ("view")."""
    return {"type": "image", "image": np.ascontiguousarray(np.asarray(image, np.uint8)[..., :3])}


def camera_item(T, K, width, height, color, depth=0.15):
    """Uma câmara desenhada como uma pirâmide: o bico na posição da câmara (pose T) e a base, a depth
    metros à frente, com a forma da imagem (os 4 cantos da imagem projetados para trás com K)."""
    corners = [[(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth]
               for u, v in ((0, 0), (width, 0), (width, height), (0, height))]
    points = np.vstack([[0.0, 0.0, 0.0], corners])                       # referencial da câmara
    points = (np.asarray(T)[:3, :3] @ points.T).T + np.asarray(T)[:3, 3]   # -> referencial da nuvem
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return lines_item(points, lines, color, width=4.0)


def frame_item(size=0.1):
    """Os eixos de um referencial na origem (x vermelho, y verde, z azul)."""
    return {"type": "frame", "size": size}


def colors_uint8(colors):
    """Cores entre 0 e 1 -> uint8 (os .npz das janelas ficam 8 vezes mais pequenos)."""
    return (np.clip(np.asarray(colors), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def look_center(T_cams):
    """Ponto para onde as câmaras olham: o mais próximo de todos os eixos óticos (mínimos quadrados).
    Com uma só câmara (ou eixos quase paralelos): 1.2 m à frente da primeira câmara."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T in T_cams:
        axis = T[:3, 2]
        P = np.eye(3) - np.outer(axis, axis)       # projeção no plano perpendicular ao eixo ótico
        A += P
        b += P @ T[:3, 3]
    if len(T_cams) < 2 or np.linalg.cond(A) > 1e6:
        return T_cams[0][:3, 3] + 1.2 * T_cams[0][:3, 2]
    return np.linalg.solve(A, b)


def camera_view(T_cam, center, up, back=0.35, above=0.25):
    """Ponto de vista de uma janela: um pouco atrás e acima da câmara T_cam, a olhar para center."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    away = T_cam[:3, 3] - center
    eye = T_cam[:3, 3] + back * away / np.linalg.norm(away) + above * up
    return {"center": np.asarray(center, float), "eye": eye, "up": up}


def scene_view(T_cams, registered):
    """Ponto de vista das janelas no referencial da câmara 000: o da própria câmara 000, um pouco atrás e
    acima dela (a imagem fica direita, como a da câmara: o eixo y da câmara aponta para baixo), a olhar
    na mesma direção, até à altura do ponto para onde as câmaras olham."""
    T0 = T_cams[0]
    used = [T for T, ok in zip(T_cams, registered) if ok]
    forward, up = T0[:3, 2], -T0[:3, 1]
    distance = max(0.5, float((look_center(used) - T0[:3, 3]) @ forward))
    return {"center": T0[:3, 3] + distance * forward, "eye": T0[:3, 3] - 0.4 * forward + 0.15 * up, "up": up}


def close_view(view, points, factor=1.5, minimum=0.25):
    """O mesmo ponto de vista (a mesma direção), mas centrado nos pontos dados e perto deles: a distância
    é factor vezes o tamanho dos pontos (pelo menos minimum metros)."""
    points = np.asarray(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2    # centro da caixa que envolve os pontos
    direction = np.asarray(view["eye"], float) - np.asarray(view["center"], float)
    distance = max(minimum, factor * float(np.ptp(np.asarray(points), axis=0).max()))
    return {"center": center, "eye": center + distance * direction / np.linalg.norm(direction), "up": view["up"]}


def save_png(image, path):
    """Grava uma imagem RGB (uint8) em PNG (cv2.imwrite falha no Windows com caminhos com acentos)."""
    rgb = np.asarray(image)[..., :3]
    cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]))[1].tofile(str(path))


# #####################################################################################################
# 2. JANELAS DO OPEN3D QUE NÃO PARAM O PROGRAMA
# #####################################################################################################
def _tubes(points, lines, color, radius):
    """Segmentos de reta como tubos finos (o visualizador desenha as linhas sempre com 1 pixel)."""
    mesh = o3d.geometry.TriangleMesh()
    for a, b in lines:
        start, end = points[a], points[b]
        length = np.linalg.norm(end - start)
        if length < 1e-9:
            continue
        tube = o3d.geometry.TriangleMesh.create_cylinder(radius, length, resolution=8)    # ao longo de z
        axis = (end - start) / length
        # rotação que leva o eixo z para axis (fórmula de Rodrigues)
        v, c = np.cross([0.0, 0.0, 1.0], axis), axis[2]
        if np.linalg.norm(v) < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx / (1.0 + c)
        tube.rotate(R, center=(0, 0, 0))
        tube.translate((start + end) / 2)
        mesh += tube
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _geometry(item):
    """Geometria do Open3D de um elemento a desenhar."""
    if item["type"] == "points":
        geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(item["points"].astype(float)))
        colors = item["colors"]
        geometry.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.dtype == np.uint8 else colors)
        return geometry
    if item["type"] == "lines":
        if item["width"] > 1.5:                               # linhas grossas (caixas): tubos
            return _tubes(item["points"], item["lines"], item["color"], 0.001 * item["width"])
        geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector(item["points"]),
                                        o3d.utility.Vector2iVector(item["lines"]))
        geometry.paint_uniform_color(item["color"])
        return geometry
    if item["type"] == "image":
        return o3d.geometry.Image(item["image"])
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=item["size"])


def _look_at(vis, view):
    """Põe o ponto de vista da janela no olho view["eye"], a olhar para view["center"], com view["up"]
    para cima (sem view fica a vista inicial do draw_geometries)."""
    if view is None:
        return
    center, eye, up = (np.asarray(view[k], float) for k in ("center", "eye", "up"))
    forward = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    extrinsic = np.eye(4)                                      # pontos do mundo -> referencial do olho
    extrinsic[:3, :3] = np.stack([right, down, forward])      # (x direita, y para baixo, z para a frente)
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
    control = vis.get_view_control()
    params = control.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


KEY_RIGHT, KEY_LEFT = 262, 263      # códigos das setas -> e <- no visualizador do Open3D (GLFW)


def _set_window_title(window, text):
    """Muda o título de uma janela (o visualizador do Open3D não o permite: usa-se a API do Windows)."""
    if os.name != "nt":
        return
    import ctypes
    user32 = ctypes.windll.user32
    if not window.get("hwnd"):
        window["hwnd"] = user32.FindWindowW(None, window["created_title"])
    if window["hwnd"]:
        user32.SetWindowTextW(window["hwnd"], text)


class SceneWindows:
    """Janelas 3D do Open3D (as do draw_geometries) que NÃO param o programa: UMA janela por cena (grupo),
    com todas as vistas dessa cena; as setas <- e -> do teclado mudam de vista (o título diz qual é).

    Cada vista é aberta com add(), e as janelas só respondem ao rato e ao teclado (e as animações só
    avançam) enquanto se chama tick(); run() espera que todas sejam fechadas. Cada vista pode também ser
    gravada num PNG (numa janela invisível). show=False: não abre janelas, só grava os PNG.
    Uma vista (dicionário) tem: "name" (sufixo do PNG), "title", "items" (o que desenhar), "view"
    ({"center", "eye", "up"}; sem view, a vista inicial do draw_geometries) e, para as animações,
    "frames" (o que desenhar em cada passo, por cima dos items). Uma vista pode também ser só uma imagem
    2D (image_item), por exemplo um gráfico do matplotlib.
    """

    def __init__(self, show=True):
        self.show = show
        self.groups = {}                 # janela de cada cena: {nome: dicionário da janela}

    @property
    def windows(self):
        """Janelas abertas."""
        return list(self.groups.values())

    @staticmethod
    def _draw(vis, spec, frame=None):
        """Desenha uma vista numa janela (e, nas animações, o passo frame). Saída: geometrias do passo."""
        vis.clear_geometries()
        for item in spec["items"]:
            vis.add_geometry(_geometry(item))
        shown = []
        if spec.get("frames") is not None:
            shown = [_geometry(item) for item in spec["frames"][-1 if frame is None else frame]]
            for geometry in shown:
                vis.add_geometry(geometry)
        option = vis.get_render_option()
        option.background_color = np.asarray(spec.get("background", WHITE), float)
        option.point_size = POINT_SIZE
        if spec.get("view") is None:
            vis.reset_view_point(True)
        _look_at(vis, spec.get("view"))
        return shown

    def save(self, spec, png):
        """Grava a imagem de uma vista num PNG (janela invisível; nas animações, o último passo). Uma vista
        que é só uma imagem 2D é gravada tal como está."""
        if len(spec["items"]) == 1 and spec["items"][0]["type"] == "image":
            save_png(spec["items"][0]["image"], png)
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window("png", *WINDOW_SIZE, visible=False)
        self._draw(vis, spec)
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        save_png((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8), png)
        vis.destroy_window()

    def add(self, spec, png=None, group="vistas"):
        """Junta uma vista à janela do grupo (a cena), abrindo-a se for a primeira, e mostra-a.
        png: onde gravar a imagem da vista."""
        if png is not None:
            self.save(spec, png)
        if not self.show:
            return
        window = self.groups.get(group)
        if window is None:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            created_title = f"SAVIBoxes {group} {len(self.groups)} {time.time():.0f}"   # único
            vis.create_window(created_title, *WINDOW_SIZE)
            window = {"vis": vis, "group": group, "views": [], "current": -1, "shown": [], "frame": None,
                      "next": 0.0, "created_title": created_title}
            vis.register_key_callback(KEY_RIGHT, lambda _vis, w=window: self._step(w, +1))
            vis.register_key_callback(KEY_LEFT, lambda _vis, w=window: self._step(w, -1))
            self.groups[group] = window
        window["views"].append(spec)
        if len(window["views"]) == 1:
            self._show(window, 0)                         # janela nova: mostra a primeira vista
        else:
            self._update_title(window)                    # mantém a vista e atualiza o n.º de vistas

    def _show(self, window, k):
        """Mostra a vista k de uma janela."""
        spec = window["views"][k]
        window["current"] = k
        window["shown"] = self._draw(window["vis"], spec)
        window["frame"] = len(spec["frames"]) - 1 if spec.get("frames") is not None else None
        window["next"] = time.time() + REPEAT_SECONDS
        self._update_title(window)

    @staticmethod
    def _update_title(window):
        """Título da janela: a vista que está a ser mostrada, de quantas."""
        k = window["current"]
        _set_window_title(window, f"{window['group']}  [{k + 1}/{len(window['views'])}]  "
                                  f"{window['views'][k]['title']}   (setas <- -> mudam de vista)")

    def _step(self, window, direction):
        """Vista anterior / seguinte (setas do teclado)."""
        self._show(window, (window["current"] + direction) % len(window["views"]))
        return True

    def tick(self):
        """Deixa as janelas responder ao rato e ao teclado e avança as animações; fecha as que o
        utilizador fechou."""
        now = time.time()
        for group, window in list(self.groups.items()):
            vis = window["vis"]
            frames = window["views"][window["current"]].get("frames")
            if frames is not None and now >= window["next"]:
                last = len(frames) - 1
                window["frame"] = 0 if window["frame"] >= last else window["frame"] + 1
                for geometry in window["shown"]:
                    vis.remove_geometry(geometry, reset_bounding_box=False)
                window["shown"] = [_geometry(item) for item in frames[window["frame"]]]
                for geometry in window["shown"]:
                    vis.add_geometry(geometry, reset_bounding_box=False)
                window["next"] = now + (REPEAT_SECONDS if window["frame"] == last else FRAME_SECONDS)
            if not vis.poll_events():                       # a janela foi fechada
                vis.destroy_window()
                del self.groups[group]
                continue
            vis.update_renderer()

    def run(self):
        """Espera que todas as janelas sejam fechadas."""
        while self.groups:
            self.tick()
            time.sleep(0.01)


def show_windows(windows):
    """Abre as vistas de make_windows numa janela (setas para mudar de vista) e espera que seja fechada
    (quando uma parte corre sozinha)."""
    group = windows[0]["title"].split(":")[0]               # ex.: "Parte 1 - cena_1"
    viewer = SceneWindows()
    for spec in windows:
        viewer.add(spec, group=group)
    viewer.run()


def data_view(data):
    """O ponto de vista guardado em figure_data."""
    return {k: data[f"view_{k}"] for k in ("center", "eye", "up")}
