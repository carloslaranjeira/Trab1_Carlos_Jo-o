"""Funções de geometria usadas em várias partes do trabalho.

Convenção: T_A_B é a matriz 4x4 que transforma pontos do referencial B para o referencial A,
p_A = T_A_B @ p_B (coordenadas homogéneas).
"""

import numpy as np


# --------------------------------------------------------------------------------------
# Transformações rígidas
# --------------------------------------------------------------------------------------
def make_T(R=np.eye(3), t=np.zeros(3)):
    """Junta uma rotação 3x3 e uma translação numa matriz homogénea 4x4."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T, points):
    """Aplica a transformação T (4x4) a um conjunto de pontos N x 3."""
    return points @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw):
    """Rotação de um ângulo yaw (radianos) em torno do eixo z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_angle_deg(R):
    """Ângulo (em graus) de uma matriz de rotação: theta = acos((traço(R) - 1) / 2)."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def kabsch(A, B):
    """Transformação rígida T que melhor leva os pontos A para os pontos B (mínimos quadrados).

    Método de Kabsch / Umeyama sem escala: centra os dois conjuntos, calcula a matriz de
    covariância H = A_c^T B_c e obtém a rotação a partir da sua SVD.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    # A correção d evita que a solução seja uma reflexão (det = -1) em vez de uma rotação
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return make_T(R, t)


def wrap_angle(a):
    """Coloca um ângulo (radianos) no intervalo [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------------------
# Caixas sobre a mesa
# --------------------------------------------------------------------------------------
# Uma caixa é descrita no referencial da mesa por 6 parâmetros: x, y, yaw, length, width, height.
# O centro da caixa fica a height/2 acima da mesa, porque a caixa assenta sempre na mesa.

def box_T_table(x, y, yaw, height):
    """Pose T_table_box de uma caixa sobre a mesa: origem no centro, z para cima."""
    return make_T(rot_z(yaw), np.array([x, y, height / 2.0]))


def box_corners_2d(x, y, yaw, length, width):
    """Quatro cantos do retângulo da caixa vista de cima (no plano da mesa)."""
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.array([length, width]) / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return half @ R.T + np.array([x, y])


def box_lines_3d(T, length, width, height):
    """Os 8 cantos e as 12 arestas de uma caixa com pose T, para desenhar como LineSet."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    corners = transform_points(T, signs * np.array([length, width, height]) / 2.0)
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    return corners, edges


def box_signed_distance(points_box, half_dims):
    """Distância com sinal de pontos (no referencial da caixa) à superfície de uma caixa centrada.

    É a 'signed distance function' clássica de um paralelepípedo:
        q = |p| - meias_dimensões
        fora  -> norma da parte positiva de q (distância à face, aresta ou canto mais próximo)
        dentro -> o maior componente de q, que é negativo (menos a distância à face mais próxima)
    O valor absoluto é a distância do ponto à superfície da caixa.
    """
    q = np.abs(points_box) - half_dims
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def yaw_of(R):
    """Yaw (rotação em torno de z) de uma matriz de rotação cujo eixo z aponta para cima."""
    return np.arctan2(R[1, 0], R[0, 0])
