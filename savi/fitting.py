"""Parte 3 (método principal): ajuste de caixas à nuvem de pontos por otimização.

Cada caixa é definida no referencial da mesa por 6 parâmetros:
    x, y      -> posição do centro no plano da mesa
    yaw       -> rotação em torno do eixo z (perpendicular à mesa)
    length    -> dimensão ao longo do eixo x da caixa
    width     -> dimensão ao longo do eixo y da caixa
    height    -> dimensão ao longo de z (a caixa assenta na mesa, por isso o centro está a height/2)

Função objetivo (um vetor de resíduos que o scipy.optimize.least_squares minimiza):
    1. Dados -> modelo: para cada ponto, a distância à superfície da caixa que melhor o explica.
       A associação ponto -> caixa é recalculada em CADA avaliação da função (argmin), dentro
       da otimização, sem ciclo exterior;
    2. Modelo -> dados: pontos amostrados nas faces visíveis de cada caixa têm de ter dados por perto.
       Impede que uma caixa cresça para espaço vazio sem custo;
    3. Sobreposição: penaliza a área em que duas caixas se intersetam (duas caixas não podem ocupar
       o mesmo espaço);
    4. (opcional) Cor: cada caixa tem uma cor e um ponto escolhe a caixa que melhor explica a sua
       posição E a sua cor (Nível 4).
A função de perda robusta 'soft_l1' faz com que uns poucos pontos errados (flying pixels,
cores trocadas junto às fronteiras) não estraguem o resultado.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import box_signed_distance, wrap_angle
from .segmentation import chroma_features

NUM_PARAMS = 6                 # x, y, yaw, length, width, height
FIT_VOXEL = 0.006              # os pontos são reduzidos a este voxel antes da otimização
MAX_FIT_POINTS = 12000         # e limitados a este número (velocidade)
GEOMETRIC_SIGMA = 0.003        # ruído geométrico típico (m), usado para pesar a cor
LOSS_SCALE = 0.004             # f_scale da perda robusta: resíduos >> 4 mm contam pouco
HIDDEN_MARGIN = 0.02           # faces a menos de 2 cm de outra caixa são consideradas escondidas


# --------------------------------------------------------------------------------------
# Estimativa inicial
# --------------------------------------------------------------------------------------
def rect_params(points_xy):
    """Retângulo de área mínima de um conjunto de pontos 2D -> (x, y, yaw, length, width)."""
    (cx, cy), (a, b), angle = cv2.minAreaRect(points_xy.astype(np.float32))
    return np.array([cx, cy, np.radians(angle), max(a, 0.02), max(b, 0.02)])


def largest_component(points, eps=0.012):
    """Índices da maior componente ligada (DBSCAN) de um conjunto de pontos."""
    if len(points) < 20:
        return np.arange(len(points))
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps, 5))
    if labels.max() < 0:
        return np.arange(len(points))
    return np.nonzero(labels == np.bincount(labels[labels >= 0]).argmax())[0]


def top_surface_points(points, cell=0.02, tolerance=0.006):
    """Pontos que estão no topo de alguma caixa.

    Divide o plano da mesa numa grelha de células de 2 cm e, em cada célula, fica com os pontos
    que estão a menos de 'tolerance' do ponto mais alto da célula. Os pontos das faces laterais
    ficam de fora (estão mais abaixo do que o topo da mesma célula).
    """
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    cell_max = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(cell_max, inverse, points[:, 2])
    return np.nonzero((points[:, 2] > cell_max[inverse] - tolerance) & (points[:, 2] > 0.02))[0]


def split_kmeans(group):
    """Divide um grupo de pontos do topo em dois com k-means no plano (x, y)."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, lab, _ = cv2.kmeans(group[:, :2].astype(np.float32), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    return group[lab == 0], group[lab == 1]


def split_profile(group, bin_size=0.005, min_piece=0.04, min_step=0.01):
    """Divide um grupo de pontos do topo em dois, no 'degrau' do contorno.

    Caixas da mesma altura encostadas formam um topo plano único. Se tiverem larguras diferentes
    (ou estiverem desalinhadas), o contorno desse topo tem um degrau onde acaba uma caixa e começa
    a outra. Para o encontrar:
      1. Eixo principal do topo (PCA em x, y): u ao longo da fila de caixas, v na perpendicular;
      2. Fatias de 5 mm ao longo de u; em cada fatia, os limites do topo em v (percentis 10 e 90);
      3. O degrau é onde estes limites mais mudam (comparando fatias 1 cm antes e 1 cm depois);
      4. Corta-se aí, desde que fiquem pelo menos 4 cm de cada lado.
    Devolve os dois pedaços, ou None se não houver nenhum degrau claro (> 1 cm).
    """
    xy = group[:, :2]
    center = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - center, full_matrices=False)
    u, v = (xy - center) @ Vt[0], (xy - center) @ Vt[1]

    edges = np.arange(u.min(), u.max() + bin_size, bin_size)
    which = np.digitize(u, edges) - 1
    lo = np.full(len(edges), np.nan)
    hi = np.full(len(edges), np.nan)
    for b in range(len(edges)):
        vb = v[which == b]
        if len(vb) >= 5:
            lo[b], hi[b] = np.percentile(vb, 10), np.percentile(vb, 90)

    k = 2                                   # compara fatias a 2 posições (1 cm) de distância
    step = np.full(len(edges), -np.inf)
    for b in range(k, len(edges) - k):
        if edges[b] - u.min() < min_piece or u.max() - edges[b] < min_piece:
            continue
        s = abs(hi[b + k] - hi[b - k]) + abs(lo[b + k] - lo[b - k])
        if not np.isnan(s):
            step[b] = s
    best = int(np.argmax(step))
    if step[best] < min_step:
        return None
    cut = edges[best] + bin_size / 2
    return group[u < cut], group[u >= cut]


def split_geometric(points, n, method="kmeans"):
    """Estimativa inicial de n caixas da mesma cor que se tocam (um só cluster).

    1. Ficamos só com os pontos dos topos das caixas;
    2. Agrupamos os topos com DBSCAN em 3D: caixas de alturas diferentes têm topos a alturas
       diferentes e ficam separadas;
    3. Se houver menos grupos que caixas, dividimos o grupo com maior área em dois, até termos
       n grupos. A divisão é feita com:
         - method="kmeans":  k-means no plano (x, y) - divide a área ao meio, sem olhar ao contorno;
         - method="profile": no degrau do contorno (split_profile); se não houver degrau, usa k-means.
       Esta divisão é só uma aproximação: a otimização é que corrige a fronteira entre as caixas.
       Mas, se começar muito longe, a otimização pode ficar presa num mínimo local, por isso o
       main.py experimenta os dois métodos e fica com o que acaba com menor custo.
    Devolve uma lista de n arrays de parâmetros iniciais.
    """
    top = points[top_surface_points(points)]
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top)).cluster_dbscan(0.012, 5))
    groups = [top[labels == k] for k in range(labels.max() + 1) if np.sum(labels == k) >= 30]
    groups.sort(key=len, reverse=True)
    groups = groups[:n] if groups else [top]

    while len(groups) < n:
        # Divide o grupo com maior área (retângulo mínimo)
        areas = [rect_params(g[:, :2])[3] * rect_params(g[:, :2])[4] for g in groups]
        g = groups.pop(int(np.argmax(areas)))
        pieces = split_profile(g) if method == "profile" else None
        groups += list(pieces if pieces is not None else split_kmeans(g))

    return [np.append(rect_params(g[:, :2]), np.median(g[:, 2])) for g in groups]


def needs_geometric_split(seg):
    """True se algum cluster tem várias caixas e a cor não as separa (Nível 3)."""
    return any(n > 1 and seg.color_groups.get(k) is None for k, n in seg.boxes_per_cluster.items())


def initial_boxes(seg, split_method="kmeans"):
    """Estimativa inicial de todas as caixas a partir da segmentação.

    - Cluster com uma caixa: retângulo mínimo do cluster visto de cima e altura = percentil 98 de z;
    - Cluster com várias caixas e cores distintas: o mesmo, para cada grupo de cor;
    - Cluster com várias caixas da mesma cor: split_geometric (topos + k-means).
    Devolve (lista de parâmetros, lista de cores (a*, b*) de cada caixa, rótulo da caixa de cada
    ponto da segmentação). Os rótulos são os 'conjuntos de pontos' de cada caixa, usados pela
    bounding box e na avaliação da segmentação.
    """
    params, colors = [], []
    labels = np.full(len(seg.points), -1)
    for k, n in seg.boxes_per_cluster.items():
        idx = np.nonzero(seg.clusters == k)[0]
        pts, cols = seg.points[idx], seg.colors[idx]
        groups = seg.color_groups.get(k)
        first = len(params)                       # índice da primeira caixa deste cluster
        if n == 1:
            params.append(np.append(rect_params(pts[:, :2]), np.percentile(pts[:, 2], 98)))
            colors.append(np.median(chroma_features(cols), axis=0))
            labels[idx] = first
        elif groups is not None:
            for g in range(n):
                sel = np.nonzero(groups == g)[0]
                sel = sel[largest_component(pts[sel])]       # retira pontos soltos com a cor errada
                params.append(np.append(rect_params(pts[sel, :2]), np.percentile(pts[sel, 2], 98)))
                colors.append(np.median(chroma_features(cols[sel]), axis=0))
                labels[idx[sel]] = first + g
        else:
            for p in split_geometric(pts, n, split_method):
                params.append(p)
                colors.append(np.median(chroma_features(cols), axis=0))
        # Pontos ainda sem caixa: ficam com a caixa (deste cluster) cuja superfície está mais perto
        free = labels[idx] < 0
        if free.any():
            D = distances_to_boxes(pts[free], params[first:first + n])
            labels[idx[free]] = first + np.argmin(D, axis=1)
    return params, colors, labels


# --------------------------------------------------------------------------------------
# Modelo da caixa
# --------------------------------------------------------------------------------------
def to_box_frame(points, p):
    """Escreve pontos (referencial da mesa) no referencial de uma caixa com parâmetros p."""
    x, y, yaw, _, _, h = p
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    # Rotação inversa de yaw (R^T) e translação para o centro da caixa (que está a h/2 da mesa)
    return np.stack([c * dx + s * dy, -s * dx + c * dy, points[:, 2] - h / 2], axis=1)


NORMAL_PENALTY = 0.02          # custo extra (m) de explicar um ponto com uma face de orientação errada


def distances_to_boxes(points, params, normals=None):
    """Matriz M x B com a distância de cada ponto à superfície de cada caixa.

    Sem normais: distância à superfície (valor absoluto da signed distance function).

    Com normais: distância à FACE mais próxima cuja orientação é compatível com a normal do ponto.
    Porquê? Quando duas caixas se tocam, os pontos do topo da caixa mais baixa junto à caixa mais
    alta estão a poucos mm da face LATERAL da caixa alta. Só com distâncias, a caixa baixa pode
    encolher alguns mm sem o erro aumentar (esses pontos passam a ser 'explicados' pela face lateral).
    Exigindo que um ponto com normal vertical seja explicado por uma face horizontal (e vice-versa)
    a ambiguidade desaparece. Faces com orientação errada não são proibidas, apenas custam mais
    NORMAL_PENALTY (pontos nas arestas têm normais pouco fiáveis).
    """
    D = np.empty((len(points), len(params)))
    for j, p in enumerate(params):
        half = np.array([p[3], p[4], p[5]]) / 2
        local = to_box_frame(points, p)
        if normals is None:
            D[:, j] = np.abs(box_signed_distance(local, half))
            continue
        # Normais no referencial da caixa (só a rotação de yaw)
        c, s = np.cos(p[2]), np.sin(p[2])
        n_local = np.abs(np.stack([c * normals[:, 0] + s * normals[:, 1],
                                   -s * normals[:, 0] + c * normals[:, 1], normals[:, 2]], axis=1))
        best = np.full(len(points), np.inf)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if axis == 2 and sign < 0:
                    continue                        # a face de baixo está assente na mesa
                # Projeção do ponto no retângulo da face: fixa a coordenada do eixo e limita as outras
                proj = np.clip(local, -half, half)
                proj[:, axis] = sign * half[axis]
                d = np.linalg.norm(local - proj, axis=1)
                d += NORMAL_PENALTY * (n_local[:, axis] < 0.7)   # orientação incompatível
                best = np.minimum(best, d)
        D[:, j] = best
    return D


# Amostras na superfície de uma caixa com dimensões unitárias ([-1, 1] em cada eixo).
# São multiplicadas pelas meias-dimensões de cada caixa. Só usamos o topo e as 4 faces laterais
# (a face de baixo está encostada à mesa e nunca é vista).
def _unit_surface_samples(n=7):
    t = np.linspace(-1, 1, n)
    a, b = np.meshgrid(t, t)
    a, b = a.ravel(), b.ravel()
    one = np.ones_like(a)
    faces = [np.stack([a, b, one], 1),                                   # topo
             np.stack([one, a, b], 1), np.stack([-one, a, b], 1),        # faces +x e -x
             np.stack([a, one, b], 1), np.stack([a, -one, b], 1)]        # faces +y e -y
    samples = np.unique(np.round(np.concatenate(faces), 6), axis=0)
    return samples[samples[:, 2] > -0.75]   # sem a parte de baixo das faces (removida com a mesa)


UNIT_SAMPLES = _unit_surface_samples()


def surface_samples(p):
    """Pontos amostrados na superfície visível da caixa p, no referencial da mesa."""
    x, y, yaw, l, w, h = p
    local = UNIT_SAMPLES * np.array([l, w, h]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * local[:, 0] - s * local[:, 1],
                     y + s * local[:, 0] + c * local[:, 1],
                     local[:, 2] + h / 2], axis=1)


def overlap_area(p, q):
    """Área da interseção dos retângulos de duas caixas vistas de cima (cv2.rotatedRectangleIntersection)."""
    r1 = ((p[0], p[1]), (p[3], p[4]), np.degrees(p[2]))
    r2 = ((q[0], q[1]), (q[3], q[4]), np.degrees(q[2]))
    status, region = cv2.rotatedRectangleIntersection(r1, r2)
    if status == cv2.INTERSECT_NONE or region is None or len(region) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(region.reshape(-1, 2).astype(np.float32))))


# --------------------------------------------------------------------------------------
# Função objetivo
# --------------------------------------------------------------------------------------
@dataclass
class FitProblem:
    points: np.ndarray                 # M x 3, referencial da mesa
    tree: cKDTree                      # KD-tree dos pontos (para o termo modelo -> dados)
    num_boxes: int
    normals: np.ndarray = None         # M x 3, normal estimada de cada ponto
    chroma: np.ndarray = None         # M x 2, cor (a*, b*) de cada ponto, se a cor for usada
    box_chroma: np.ndarray = None      # B x 2, cor de cada caixa
    color_weight: float = 0.0          # lambda: converte diferenças de cor em 'metros equivalentes'
    coverage_weight: float = 1.0
    overlap_weight: float = 1.0
    history: list = field(default_factory=list)   # parâmetros e custo em cada avaliação

    def residuals(self, x):
        params = x.reshape(self.num_boxes, NUM_PARAMS)
        M = len(self.points)

        # 1. Dados -> modelo, com associação recalculada AQUI (em cada avaliação)
        D = distances_to_boxes(self.points, params, self.normals)
        if self.chroma is not None:
            # Cor: distância de cor de cada ponto a cada caixa, pesada por lambda
            C = np.linalg.norm(self.chroma[:, None, :] - self.box_chroma[None], axis=2) * self.color_weight
            assoc = np.argmin(D ** 2 + C ** 2, axis=1)       # caixa que melhor explica posição + cor
            rows = np.arange(M)
            res_points = np.concatenate([D[rows, assoc], C[rows, assoc]])
        else:
            assoc = np.argmin(D, axis=1)                      # caixa mais próxima do ponto
            res_points = D[np.arange(M), assoc]

        # 2. Modelo -> dados: cada amostra da superfície tem de ter um ponto medido por perto.
        #    Amostras dentro de outra caixa, ou a menos de HIDDEN_MARGIN dela (faces de contacto,
        #    que as câmaras não conseguem ver), não contam. Uma margem pequena (ex.: 5 mm) criava
        #    uma barreira: a face só deixava de ser penalizada quando já estava encostada à outra
        #    caixa, e o otimizador ficava parado a ~1 cm dela.
        res_cover = []
        scale = np.sqrt(M / (self.num_boxes * len(UNIT_SAMPLES)))   # equilibra com o n.º de pontos
        for j, p in enumerate(params):
            samples = surface_samples(p)
            hidden = np.zeros(len(samples), bool)
            for k, q in enumerate(params):
                if k != j:
                    hidden |= box_signed_distance(to_box_frame(samples, q), np.array(q[3:6]) / 2) < HIDDEN_MARGIN
            dist, _ = self.tree.query(samples, k=1)
            dist[hidden] = 0.0
            res_cover.append(self.coverage_weight * scale * dist)

        # 3. Sobreposição entre caixas (raiz da área -> unidades de comprimento)
        res_overlap = [self.overlap_weight * np.sqrt(M) * np.sqrt(overlap_area(params[i], params[j]))
                       for i in range(self.num_boxes) for j in range(i + 1, self.num_boxes)]

        residuals = np.concatenate([res_points] + res_cover + [np.array(res_overlap)])
        # Guarda o histórico para visualizar a evolução da otimização
        self.history.append((x.copy(), float(np.sum(np.sqrt(1 + (residuals / LOSS_SCALE) ** 2) - 1))))
        return residuals

    def associations(self, x):
        """Caixa associada a cada ponto (para avaliar a segmentação final)."""
        params = x.reshape(self.num_boxes, NUM_PARAMS)
        D = distances_to_boxes(self.points, params, self.normals)
        if self.chroma is not None:
            C = np.linalg.norm(self.chroma[:, None, :] - self.box_chroma[None], axis=2) * self.color_weight
            return np.argmin(D ** 2 + C ** 2, axis=1)
        return np.argmin(D, axis=1)


def normalize_box(p):
    """Põe a caixa numa forma canónica: length >= width e yaw em [-90°, 90°).

    Trocar length com width e rodar 90° (ou rodar 180°) dá a mesma caixa (ver 'simetrias').
    """
    x, y, yaw, l, w, h = p
    if w > l:
        l, w, yaw = w, l, yaw + np.pi / 2
    yaw = wrap_angle(2 * yaw) / 2          # módulo 180°
    return np.array([x, y, yaw, l, w, h])


def evolution_snapshots(history):
    """Do histórico de avaliações, fica só com as que melhoraram o custo (a 'trajetória')."""
    best, snapshots = np.inf, []
    for x, cost in history:
        if cost < best - 1e-9:
            best = cost
            snapshots.append(x)
    return snapshots


def fit_boxes(seg, init_params, init_colors, use_color=True, use_normals=True, verbose=True):
    """Ajusta todas as caixas ao mesmo tempo com scipy.optimize.least_squares.

    Devolve (parâmetros finais N x 6, problema com o histórico, pontos usados, associações).
    """
    # Redução dos pontos (velocidade): voxel e, se ainda forem muitos, amostra aleatória
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg.points))
    pc.colors = o3d.utility.Vector3dVector(seg.colors)
    pc = pc.voxel_down_sample(FIT_VOXEL)
    points, colors = np.asarray(pc.points), np.asarray(pc.colors)
    if len(points) > MAX_FIT_POINTS:
        sel = np.random.default_rng(0).choice(len(points), MAX_FIT_POINTS, replace=False)
        points, colors = points[sel], colors[sel]

    # Normais de cada ponto (PCA dos vizinhos num raio de 1.5 cm). O sinal não interessa.
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    normals = np.asarray(pc.normals)

    n = len(init_params)
    problem = FitProblem(points, cKDTree(points), n, normals=normals if use_normals else None)

    # A cor só entra na função objetivo se a segmentação mostrou que as caixas têm cores distintas
    color_informative = any(g is not None for g in seg.color_groups.values())
    if use_color and color_informative and n > 1:
        problem.chroma = chroma_features(colors)
        problem.box_chroma = np.array(init_colors)
        # Peso da cor (Parte 06, Ex. 3): cada erro é dividido pelo seu desvio padrão.
        # lambda = sigma_geométrico / sigma_cor, para que 1 sigma de cor 'valha' 1 sigma de distância.
        nearest = np.argmin(np.linalg.norm(problem.chroma[:, None] - problem.box_chroma[None], axis=2), axis=1)
        spread = np.median(np.linalg.norm(problem.chroma - problem.box_chroma[nearest], axis=1)) * 1.4826
        problem.color_weight = GEOMETRIC_SIGMA / max(spread, 1e-3)

    x0 = np.concatenate(init_params)
    # Limites: dimensões entre 2 cm e 1 m; a posição e o yaw são livres
    lower = np.tile([-np.inf, -np.inf, -np.inf, 0.02, 0.02, 0.02], n)
    upper = np.tile([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0], n)
    x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
    # Escala típica de cada parâmetro (ajuda o otimizador): 1 cm nas distâncias, ~3° no yaw
    x_scale = np.tile([0.01, 0.01, 0.05, 0.01, 0.01, 0.01], n)

    result = least_squares(problem.residuals, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=LOSS_SCALE, x_scale=x_scale, diff_step=1e-4, max_nfev=200)
    params = np.array([normalize_box(p) for p in result.x.reshape(n, NUM_PARAMS)])
    if verbose:
        print(f"  otimização: {result.nfev} iterações, custo {problem.history[0][1]:.1f} -> "
              f"{min(h[1] for h in problem.history):.1f}, "
              f"cor {'sim' if problem.chroma is not None else 'não'} ({len(points)} pontos)")
    return params, problem, points, problem.associations(result.x)
