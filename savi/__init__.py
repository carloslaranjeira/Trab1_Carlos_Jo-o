# SAVI 2026-2027 - Trabalho Prático 1: SAVIBoxes
#
# Pacote com todas as etapas do sistema:
#   scene_io.py      -> leitura das cenas e escrita do result.json
#   geometry.py      -> funções de geometria (transformações, caixas, distâncias)
#   registration.py  -> Parte 1: registo das nuvens de pontos (features + RANSAC + ICP)
#   segmentation.py  -> Parte 2: deteção da mesa e segmentação dos objetos
#   bbox_method.py   -> Parte 3: método base (bounding box orientada)
#   fitting.py       -> Parte 3: método principal (ajuste por otimização)
#   evaluation.py    -> comparação com o ground truth (erros, IoU 3D, precisão/recall)
#   visualization.py -> visualizações 3D (Open3D) e figuras (matplotlib)
