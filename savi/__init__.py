# SAVI 2026-2027 - Trabalho Prático 1: SAVIBoxes
#
# Um ficheiro por parte do enunciado (correm por esta ordem; cada um grava o resultado em <cena>/partes/):
#   parte1_registo.py      -> PARTE 1: registo das imagens e nuvem agregada (registo.json)
#   extra_fotometria.py    -> EXTRA: correção fotométrica, s e b de cada imagem, imagens corrigidas e nuvens
#                             coloridas de novo (fotometria.json, fotometria/NNN.png)
#   parte2_segmentacao.py  -> PARTE 2: mesa, pontos dos objetos e separação das caixas (estima o n.º de caixas)
#   parte3_ajuste.py       -> PARTE 3: ajuste das caixas (otimização e bounding box) -> result.json
#   avaliador.py           -> comparação com o ground truth
#
# Código comum (não pertence a nenhuma parte):
#   comum.py               -> geometria (matrizes 4x4, caixas), leitura da cena, result.json, convenções
#   visualizacao.py        -> janelas 3D do Open3D (SceneWindows) e elementos a desenhar
