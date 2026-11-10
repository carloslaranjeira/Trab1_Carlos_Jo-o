"""Parte 1: registo das nuvens de pontos.

Objetivo: estimar a pose T_camera0_camera de cada câmara e construir uma nuvem agregada,
toda escrita no referencial da câmara 000.

Passos (como no enunciado):
    1. Pré-processamento: filtragem, downsampling e normais;
    2. Estimativa inicial: features das imagens a cores (ASIFT) -> pares de pontos 3D -> RANSAC (Kabsch).
       O RANSAC devolve várias hipóteses diferentes; cada uma é refinada com ICP e escolhemos a que
       melhor alinha as nuvens completas (geometria + cor);
    3. ICP ponto-a-plano em várias escalas, com a nuvem COMPLETA (mesa, caixas e fundo);
    4. (opcional) Otimização de um grafo de poses com fecho de ciclo (última imagem -> primeira),
       para distribuir o erro acumulado pelo encadeamento;
    5. Agregação: aplicar as transformações às nuvens com a resolução original.

Notas sobre as escolhas (resultado de testes com as cenas do simulador):
    - SIFT normal encontra muito poucas associações corretas entre câmaras a 45° (a perspetiva
      muda muito). O ASIFT (Affine-SIFT) simula várias inclinações da imagem antes de calcular
      os descritores e encontra 3 a 10 vezes mais associações corretas.
    - Features 3D (FPFH) falham nesta cena: está cheia de planos (mesa, chão, paredes), que têm
      todos o mesmo descritor, e o RANSAC acaba por alinhar planos errados.
    - O ruído da profundidade cresce com z^2 (a 4 m são ~2.5 cm). No ICP ignoramos os pontos a mais
      de MAX_ICP_DEPTH metros e usamos um kernel robusto (Tukey). O fundo próximo (paredes,
      móveis) continua a ser usado, e é ele que impede a nuvem de 'deslizar' sobre a mesa.
"""

import copy

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .geometry import kabsch, rotation_angle_deg, transform_points

reg = o3d.pipelines.registration

MAX_ICP_DEPTH = 3.0                    # metros: pontos mais longe são demasiado ruidosos para o ICP
ICP_VOXELS = (0.04, 0.02, 0.01, 0.005)  # escalas do ICP, do grosseiro para o fino


# --------------------------------------------------------------------------------------
# 1. Pré-processamento
# --------------------------------------------------------------------------------------
def crop_depth(cloud, max_depth=MAX_ICP_DEPTH):
    """Mantém só os pontos com profundidade (z da câmara) inferior a max_depth."""
    z = np.asarray(cloud.points)[:, 2]
    return cloud.select_by_index(np.nonzero(z < max_depth)[0])


def preprocess(cloud, voxel):
    """Downsampling com uma grelha de voxels, remoção de outliers e estimação das normais.

    - voxel_down_sample: substitui todos os pontos dentro de cada voxel pela sua média,
      o que torna a densidade uniforme e o ICP muito mais rápido;
    - remove_statistical_outlier: retira pontos cuja distância média aos vizinhos é anormal
      (por exemplo os 'flying pixels' junto às arestas dos objetos);
    - estimate_normals: o ICP ponto-a-plano precisa da normal de cada ponto.
    """
    down = cloud.voxel_down_sample(voxel)
    down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return down


# --------------------------------------------------------------------------------------
# 2. Estimativa inicial a partir de features das imagens
# --------------------------------------------------------------------------------------
def detect_features(view):
    """Deteta keypoints e descritores ASIFT na imagem a cores, só onde há profundidade."""
    detector = cv2.AffineFeature_create(cv2.SIFT_create(nfeatures=0))
    gray = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    mask = (view.xyz_image[..., 2] > 0).astype(np.uint8) * 255
    return detector.detectAndCompute(gray, mask)


def match_3d_pairs(view_a, feats_a, view_b, feats_b, ratio=0.8):
    """Associa features entre duas imagens e devolve os pares de pontos 3D correspondentes.

    1. Para cada descritor de B procura os 2 mais parecidos em A e aplica o teste da razão de Lowe
       (o melhor tem de ser claramente melhor que o segundo, senão a associação é ambígua);
    2. Para cada associação 2D vai buscar o ponto 3D do pixel em cada imagem (xyz_image).
    Devolve (pontos em A, pontos em B), ambos N x 3.
    """
    (kp_a, des_a), (kp_b, des_b) = feats_a, feats_b
    if des_a is None or des_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    height, width = view_a.xyz_image.shape[:2]

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_b, des_a, k=2)
    points_a, points_b = [], []
    for pair in matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > ratio * second.distance:   # teste da razão de Lowe
            continue
        ub, vb = np.round(kp_b[best.queryIdx].pt).astype(int)
        ua, va = np.round(kp_a[best.trainIdx].pt).astype(int)
        pa = view_a.xyz_image[min(va, height - 1), min(ua, width - 1)]
        pb = view_b.xyz_image[min(vb, height - 1), min(ub, width - 1)]
        if pa[2] > 0 and pb[2] > 0:                     # os dois pixels têm de ter medição 3D
            points_a.append(pa)
            points_b.append(pb)
    return np.array(points_a).reshape(-1, 3), np.array(points_b).reshape(-1, 3)


def ransac_hypotheses(points_src, points_dst, threshold=0.02, iterations=3000, max_hypotheses=4, seed=0):
    """RANSAC que devolve as melhores hipóteses DIFERENTES de T_dst_src (pares de pontos 3D).

    Em cada iteração escolhe 3 pares, calcula a transformação com Kabsch e conta quantos
    pares ficam a menos de 'threshold' metros (inliers).

    Amostragem guiada por consistência: uma transformação rígida preserva distâncias, por isso
    dois pares corretos (a1,b1) e (a2,b2) têm |a1-a2| ~ |b1-b2|. Escolhemos o 1.º par ao acaso e os
    outros dois entre os pares compatíveis com ele. Com 4% de inliers, o RANSAC normal precisaria
    de dezenas de milhares de iterações; assim bastam poucas centenas.

    Porque várias hipóteses? Há texturas repetidas (tábuas do chão, livros) que geram grupos
    de associações erradas mas coerentes entre si. Às vezes um modelo errado tem mais inliers
    que o correto. Por isso guardamos as melhores hipóteses distintas e deixamos o ICP decidir.
    """
    rng = np.random.default_rng(seed)
    n = len(points_src)
    if n < 3:
        return []

    # Matriz de compatibilidade entre associações (n x n)
    d_src = np.linalg.norm(points_src[:, None] - points_src[None], axis=-1)
    d_dst = np.linalg.norm(points_dst[:, None] - points_dst[None], axis=-1)
    compatible = (np.abs(d_src - d_dst) < threshold) & (d_src > 0.05)   # 5 cm: evita amostras degeneradas
    neighbours = [np.nonzero(row)[0] for row in compatible]

    hypotheses = []
    for _ in range(iterations):
        first = rng.integers(n)
        if len(neighbours[first]) < 2:
            continue
        second, third = rng.choice(neighbours[first], 2, replace=False)
        if not compatible[second, third]:
            continue
        sample = [first, second, third]
        T = kabsch(points_src[sample], points_dst[sample])
        inliers = np.linalg.norm(transform_points(T, points_src) - points_dst, axis=1) < threshold
        if inliers.sum() >= 3:
            # Refinamento: Kabsch com todos os inliers deste modelo
            T = kabsch(points_src[inliers], points_dst[inliers])
            hypotheses.append((int(inliers.sum()), T))

    # Ordena por número de inliers e fica com as melhores que sejam realmente diferentes
    hypotheses.sort(key=lambda h: -h[0])
    distinct = []
    for count, T in hypotheses:
        different = all(np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.05 or
                        rotation_angle_deg(T[:3, :3].T @ U[:3, :3]) > 5.0 for _, U in distinct)
        if different:
            distinct.append((count, T))
        if len(distinct) == max_hypotheses:
            break
    return distinct


# --------------------------------------------------------------------------------------
# 3. ICP
# --------------------------------------------------------------------------------------
def multiscale_icp(source, target, T_init, voxels=ICP_VOXELS):
    """ICP ponto-a-plano em várias escalas (do grosseiro para o fino), com kernel robusto de Tukey.

    Em cada escala as nuvens são reduzidas com um voxel diferente e a distância máxima de
    correspondência é proporcional a esse voxel. Começar grosseiro dá uma bacia de convergência
    larga; acabar fino dá precisão. O kernel de Tukey dá peso zero aos resíduos grandes, por isso
    pontos com muito ruído ou zonas que só uma das câmaras vê não puxam a solução.
    Devolve a transformação final e a matriz de informação (usada no grafo de poses).
    """
    source, target = crop_depth(source), crop_depth(target)
    T = T_init
    for voxel in voxels:
        src, dst = preprocess(source, voxel), preprocess(target, voxel)
        estimation = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=voxel))
        result = reg.registration_icp(src, dst, voxel * 1.5, T, estimation,
                                      reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7,
                                                                 max_iteration=60))
        T = result.transformation
    information = reg.get_information_matrix_from_point_clouds(src, dst, voxels[-1] * 1.5, T)
    return T, result, information


def alignment_score(source, target, T, distance=0.02, max_color_diff=0.08):
    """Avalia um alinhamento com as nuvens completas (incluindo o fundo distante) e a cor.

    Devolve a fração de pontos da fonte que, depois de transformados, têm um ponto do alvo
    a menos de 'distance' metros E com cor parecida. Um alinhamento errado que faz 'deslizar'
    a mesa sobre si própria pode ter boa sobreposição geométrica local, mas falha no fundo
    e na cor (põe chão em cima de mesa, por exemplo).
    """
    src = source.voxel_down_sample(0.02)
    dst = target.voxel_down_sample(0.02)
    src.transform(T)
    # KD-tree com os pontos do alvo: para cada ponto da fonte encontra o vizinho mais próximo
    tree = cKDTree(np.asarray(dst.points))
    dist, idx = tree.query(np.asarray(src.points), k=1)
    color_diff = np.abs(np.asarray(dst.colors)[idx] - np.asarray(src.colors)).mean(axis=1)
    good = (dist < distance) & (color_diff < max_color_diff)
    return good.mean()


# --------------------------------------------------------------------------------------
# Registo de um par de imagens
# --------------------------------------------------------------------------------------
def register_pair(view_a, feats_a, view_b, feats_b, verbose=True):
    """Estima T_a_b: a transformação que leva pontos da câmara b para a câmara a.

    1. Pares 3D a partir das features ASIFT;
    2. Várias hipóteses de RANSAC;
    3. Cada hipótese é refinada com ICP e avaliada com alignment_score; fica a melhor.
    """
    pts_a, pts_b = match_3d_pairs(view_a, feats_a, view_b, feats_b)
    hypotheses = ransac_hypotheses(pts_b, pts_a)
    if not hypotheses:
        # Sem associações suficientes: partimos da identidade (as câmaras estão próximas)
        hypotheses = [(0, np.eye(4))]

    best = None
    for inliers, T_init in hypotheses:
        # ICP só com as escalas grosseiras para escolher a hipótese (mais rápido)
        T, _, _ = multiscale_icp(view_b.cloud, view_a.cloud, T_init, voxels=ICP_VOXELS[:2])
        score = alignment_score(view_b.cloud, view_a.cloud, T)
        if best is None or score > best[0]:
            best = (score, T, inliers)

    # Refinamento final da hipótese vencedora em todas as escalas
    score, T_coarse, inliers = best
    T, result, information = multiscale_icp(view_b.cloud, view_a.cloud, T_coarse)
    if verbose:
        print(f"    {view_a.name} <- {view_b.name}: {len(pts_a)} pares 3D, {len(hypotheses)} hipóteses, "
              f"escolhida com {inliers} inliers (score={score:.3f}); "
              f"ICP fitness={result.fitness:.3f}, rmse={result.inlier_rmse * 1000:.2f} mm")
    return T, information


# --------------------------------------------------------------------------------------
# Registo de todas as imagens
# --------------------------------------------------------------------------------------
def register_scene(scene, loop_closure=True, verbose=True):
    """Regista todas as imagens da cena e devolve (T_camera0_camera com grafo, só encadeamento).

    As imagens estão em anel à volta da mesa, por isso registamos cada imagem com a seguinte
    (i -> i+1). Encadeando as transformações obtemos a pose de cada câmara em relação à câmara 0:
        T_0_(i+1) = T_0_i @ T_i_(i+1)
    O erro de cada par acumula-se ao longo da cadeia. Com loop_closure=True registamos também a
    última imagem com a primeira e otimizamos um grafo de poses, que distribui o erro pelo anel.
    """
    views = scene.views
    n = len(views)
    if verbose:
        print("Parte 1: registo")
        print("  a detetar features ASIFT...")
    features = [detect_features(v) for v in views]

    # Registo dos pares consecutivos (odometria)
    edges = []
    for i in range(n - 1):
        T, info = register_pair(views[i], features[i], views[i + 1], features[i + 1], verbose)
        edges.append((i, i + 1, T, info, False))

    # Encadeamento das transformações: estimativa só com odometria
    T_chain = [np.eye(4)]
    for i, j, T, _, _ in edges:
        T_chain.append(T_chain[i] @ T)

    if not loop_closure or n < 3:
        return T_chain, T_chain

    # Fecho de ciclo: a última imagem também é vizinha da primeira (o anel fecha)
    T, info = register_pair(views[n - 1], features[n - 1], views[0], features[0], verbose)
    edges.append((n - 1, 0, T, info, True))

    # Grafo de poses: cada nó é uma câmara (com a pose T_0_i), cada aresta é um registo de um par
    graph = reg.PoseGraph()
    for T_node in T_chain:
        graph.nodes.append(reg.PoseGraphNode(T_node))
    for i, j, T, info, uncertain in edges:
        # Aresta (source=j, target=i, T_i_j): o Open3D quer que T leve pontos de 'source' para 'target'
        graph.edges.append(reg.PoseGraphEdge(j, i, T, info, uncertain=uncertain))

    option = reg.GlobalOptimizationOption(max_correspondence_distance=ICP_VOXELS[-1] * 1.5,
                                          edge_prune_threshold=0.25, reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), option)
    T_graph = [np.array(node.pose) for node in graph.nodes]

    # Garantimos que a câmara 0 é exatamente a identidade (é a referência)
    T0_inv = np.linalg.inv(T_graph[0])
    T_graph = [T0_inv @ T for T in T_graph]
    return T_graph, T_chain


# --------------------------------------------------------------------------------------
# 4. Agregação
# --------------------------------------------------------------------------------------
def aggregate(scene, T_camera0_cameras):
    """Junta as nuvens com a resolução original, todas no referencial da câmara 0.

    Devolve a nuvem agregada e, para cada ponto, o índice da câmara que o adquiriu
    (útil para colorir cada ponto pela câmara, como na figura do enunciado).
    """
    aggregated = o3d.geometry.PointCloud()
    camera_index = []
    for k, (view, T) in enumerate(zip(scene.views, T_camera0_cameras)):
        cloud = copy.deepcopy(view.cloud)
        cloud.transform(T)
        aggregated += cloud
        camera_index.append(np.full(len(cloud.points), k))
    return aggregated, np.concatenate(camera_index)
