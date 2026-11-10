# SAVIBoxes — Trabalho Prático 1 de SAVI

Estimação da pose e das dimensões de caixas sobre uma mesa a partir de várias imagens RGB-D.

> No repositório GitHub este ficheiro deve chamar-se `README.md`.
> (Aqui está em `Read_me/Read_me_Código.md` porque, no Windows, `README.md` e o `readme.md` do enunciado são o mesmo ficheiro.)

## Como correr

Todos os comandos são corridos a partir da pasta principal do trabalho.

```
pip install -r Read_me/requirements.txt
python simulator/savi_box_simulator.py --batch simulator/configs/level_*.json simulator/configs/stack_*.json --output scenes_offline
python savi/trabalho1_savi.py --scene scenes_offline/level_4 --num_boxes 3                # escreve result.json
python savi/trabalho1_savi.py --scene scenes_offline/level_4 --num_boxes 3 --evaluate    # + tabelas de erros
python savi/trabalho1_savi.py --scene scenes_offline/level_4 --num_boxes 3 --visualize   # + janelas 3D
python savi/trabalho1_savi.py --evaluate_scenes scenes_offline/level_* scenes_offline/stack_*     # tabelas em results_offline/
python savi/trabalho1_savi.py --evaluate_scenes scenes_offline/level_* --gt_registration          # isola segmentação + ajuste
python run_test.py --download <link do GitHub>                                           # dia do teste: todas as cenas
```

Pastas:

| pasta | o que tem |
|---|---|
| `savi/` | todo o código, num só ficheiro: `trabalho1_savi.py` |
| `simulator/` | simulador do professor e configurações das cenas (`configs/`, incluindo `stack_*.json`) |
| `scenes_offline/` | cenas de desenvolvimento geradas pelo simulador (com `ground_truth.json`; não vão para o git) |
| `results_offline/` | tabelas (`results.md`) e figuras da opção `--evaluate_scenes` |
| `test/` | cenas do dia do teste (copiadas do GitHub pelo `run_test.py`; não vão para o git) |
| `results_test/` | um `result.json` (e `top_view.png`) por cena do teste, escritos pelo `run_test.py` |
| `Read_me/` | este relatório, o enunciado e o `requirements.txt` |

Opções do `trabalho1_savi.py`:

| opção | o que faz |
|---|---|
| `--scene`, `--num_boxes` | pasta da cena e número de caixas (obrigatórias) |
| `--output` | onde escrever o `result.json` (por omissão: dentro da pasta da cena) |
| `--method bbox` | usa a bounding box no `result.json`, em vez da otimização |
| `--no_color` | não usa a cor (nem na segmentação nem na otimização) |
| `--no_loop_closure` | registo só por encadeamento, sem fecho de ciclo |
| `--evaluate` | compara com o `ground_truth.json` (se existir) e mostra as tabelas de erros |
| `--visualize` | abre janelas 3D com cada etapa (registo, segmentação, otimização, resultado) |
| `--figures` | grava `top_view.png` (caixas vistas de cima) e `optimization.png` (evolução da otimização) |
| `--evaluate_scenes CENA...` | avalia várias cenas com ground truth e escreve as tabelas (não precisa de `--scene`) |
| `--gt_registration` | com `--evaluate_scenes`: usa as poses verdadeiras (avalia só segmentação e ajuste) |

**Nota:** foi feita uma alteração de uma linha no simulador (`cv2.imwrite` → `cv2.imencode(...).tofile(...)`), porque
no Windows o OpenCV não escreve ficheiros em caminhos com acentos (como "Trabalho prático") e as imagens RGB não eram gravadas.

## Estrutura do código

**Todo o código está num só ficheiro, `savi/trabalho1_savi.py`**: o sistema (registo, segmentação, ajuste,
`result.json`), a avaliação com o ground truth e a visualização. O único outro ficheiro é o `run_test.py`
(na pasta principal), que no dia do teste corre todas as cenas de `test/` e escreve
`results_test/<cena>/result.json`, usando as funções do `trabalho1_savi.py`.

Secções do `trabalho1_savi.py` (pela ordem do enunciado). Cada função tem um comentário que diz o que faz, o que
recebe, o que devolve e porque foi feita assim.

| Secção | Funções principais | Para que serve |
|---|---|---|
| 0. Geometria | `make_T`, `transform_points`, `kabsch`, `rotation_angle_deg`, `box_signed_distance` | Matrizes 4x4, rotações, distância de um ponto a uma caixa |
| 1. Leitura da cena | `load_scene`, `cloud_to_xyz_image`, `boxes_to_camera0`, `write_result` | Ler as imagens e nuvens; ponto 3D de cada pixel; escrever o `result.json` |
| 2. Parte 1 — Registo | `detect_features`, `match_3d_pairs`, `ransac_hypotheses`, `multiscale_icp`, `alignment_score`, `free_space_violation`, `table_hypotheses`, `register_pair`, `register_scene`, `aggregate` | Pose de cada câmara em relação à câmara 000 e nuvem agregada |
| 3. Parte 2 — Segmentação | `table_plane_guess`, `detect_table`, `extract_objects`, `euclidean_clusters`, `allocate_boxes`, `count_top_levels`, `chroma_features`, `color_split`, `segment` | Mesa e referencial da mesa; pontos dos objetos; clusters; caixas por cluster; cor |
| 4. Parte 3 — Ajuste | `initial_boxes`, `split_geometric`, `split_profile`, `place_on_supports`, `distances_to_boxes`, `FitProblem.residuals`, `fit_boxes` | Estimativa inicial e otimização com `least_squares` |
| 5. Método base | `bbox_box`, `bbox_method` | Bounding box orientada, para comparar |
| 6. Avaliação | `camera_errors`, `evaluate_segmentation`, `evaluate_boxes`, `iou_3d`, `print_evaluation` | Erros em relação ao ground truth (só `--evaluate` / `--evaluate_scenes`) |
| 7. Visualização | `show_registration`, `show_segmentation`, `animate_optimization`, `show_final`, `save_top_view`, `save_evolution` | Janelas 3D e figuras PNG (só `--visualize` / `--figures`) |
| 8. Várias cenas | `evaluate_scenes` | Corre várias cenas e escreve as tabelas em `results_offline/results.md` |
| 9. Programa principal | `run`, `main` | Junta tudo; lê os argumentos e escreve o `result.json` |

## Parte 1 — Registo

1. **Pré-processamento:** downsampling por voxel, remoção estatística de outliers (retira os *flying pixels*) e normais.
2. **Estimativa inicial:** cada ponto da nuvem vem de um pixel, por isso projetando a nuvem com K obtém-se o ponto 3D
   de cada pixel. As features das imagens dão então pares de pontos 3D, e um RANSAC com Kabsch estima a transformação.
   - O **SIFT** normal deu muito poucas associações corretas entre câmaras a 45° (2 a 13 em centenas).
     O **ASIFT** (`cv2.AffineFeature`), que simula inclinações da imagem, dá 3 a 10 vezes mais.
   - O **FPFH** (features 3D) falhou em todos os pares: a cena está cheia de planos com descritores iguais.
   - O RANSAC usa **amostragem guiada por consistência** (um movimento rígido preserva distâncias entre pontos),
     o que torna possível encontrar o modelo com apenas ~4% de inliers.
   - Texturas repetidas (tábuas do chão, livros) criam modelos errados com mais inliers que o correto. Por isso o RANSAC
     devolve **várias hipóteses**; cada uma é refinada com ICP e escolhe-se a que melhor alinha as nuvens completas
     (pontos a < 2 cm **e** com cor parecida).
3. **ICP** ponto-a-plano em 4 escalas (4 cm → 5 mm), com kernel robusto de Tukey, sobre a nuvem completa (mesa, caixas e fundo).
   Os pontos a mais de 3 m são ignorados: o ruído cresce com z² (a 4 m já são ~2.5 cm).
4. **Validação de cada par:** se o melhor alinhamento tiver um score baixo (< 0.18; os pares corretos dão 0.26–0.48),
   todas as hipóteses estavam erradas. Tenta-se outra vez com mais esforço (teste de Lowe 0.9, 10 000 iterações,
   10 hipóteses). Se continuar baixo, **essa aresta não é usada**: como as câmaras formam um anel, as câmaras seguintes
   são posicionadas pelo outro lado, através do fecho de ciclo. Se uma câmara ficar isolada, é registada com uma câmara
   a duas posições de distância.
   Num teste, um único par falhado punha uma câmara com 840 mm de erro e todas as seguintes também. Com a validação,
   o erro máximo ficou em 1.2 mm.
5. **Grafo de poses** com fecho de ciclo (imagem 7 → imagem 0) e só com as arestas válidas, otimizado com o Open3D.
6. **Agregação** das nuvens na resolução original, no referencial da câmara 000.

Resultado no nível 1: erro máximo de **1.9 mm / 0.07°** com grafo de poses, contra 2.0 mm / 0.10° só com encadeamento.
No nível 4: erros entre 0.4 e 2.4 mm e abaixo de 0.05°.

**Questões:**
- *Retirar o fundo e deixar só a mesa e as caixas?* A mesa é um plano quase sem textura geométrica e as caixas são
  pequenas: o ICP fica mal condicionado (a nuvem pode "deslizar" sobre o plano da mesa e rodar em torno da normal),
  converge para mínimos errados ou acumula erros grandes nas direções paralelas à mesa. O fundo (paredes, móveis)
  é que fixa essas direções.
- *Encadear muitas imagens?* O erro de cada par soma-se ao longo da cadeia (deriva), e a última imagem não fecha
  exatamente com a primeira. O grafo de poses com a aresta 7→0 distribui esse erro pelo anel.
- *Efeito dos erros de registo na nuvem agregada?* As faces das caixas aparecem "duplicadas" ou engrossadas
  (vários mm), o que aumenta a dispersão dos pontos e enviesa principalmente as dimensões estimadas.

### Robustez a poucas câmaras (2 a 4, num arco, 45°–75° entre vizinhas)

- **Hipóteses da mesa:** cada câmara deteta sozinha o tampo e o seu referencial; para um par,
  `T_a_b = T_a_mesa · Rz(0° ou 180°) · T_b_mesa⁻¹` fica a ~1 cm da solução (as features chegam a ter só 4
  associações corretas em 400). As caixas desfazem a simetria de 180° através do score.
- **Teste de espaço livre:** cada pixel de uma câmara diz que entre ela e a superfície medida não há nada; uma
  hipótese que põe pontos da outra câmara nesse espaço vazio (com margem 3 cm + 0.005·z², o ruído cresce com
  z²) é descartada se mais de 10% dos pontos o violarem (corretas: 1–5%; erradas: 19–40%).
- **Confiança:** score ≥ 0.18, ou score ≥ 0.05 e pelo menos 1.5× melhor que a melhor hipótese diferente.
- **Sem fecho de ciclo em arco:** se a primeira e a última câmara olharem com mais de 100° de diferença não
  têm nada em comum, e o par 3→0 só dava soluções erradas.
- **Faces nunca vistas:** no ajuste, o termo de cobertura só conta faces viradas para alguma câmara.

Teste 3 (2–4 câmaras): IoU médio 0.206 → ~0.97 (todas as cenas entre 0.94 e 0.99); cena de 8 câmaras sem
regressão (0.97, erro das câmaras 1.8 mm).

## Parte 2 — Segmentação

1. **Direção "para cima":** RANSAC (`segment_plane`) nos pontos do **centro das imagens** (as câmaras apontam para
   a mesa); a normal do plano é orientada para o lado das câmaras.
2. **Tampo, como na Parte 08 Ex. 4, com o RANSAC guiado pelas normais (Parte 07 Ex. 8):**
   - calcula-se a normal de cada ponto e ficam só os de superfícies horizontais (normal a menos de 10° da vertical,
     θ = arccos|n·up|, Parte 07 Ex. 6). Assim o RANSAC não encontra "planos" que cortam paredes ou móveis na diagonal;
   - **chão** = o maior plano desses pontos; a altura de cada ponto é a sua distância ao chão;
   - **mesa** = RANSAC só nos pontos entre **0.5 e 1.0 m** acima do chão; *clustering* euclidiano do plano e fica o
     maior grupo (outras superfícies à mesma altura, como um aparador, ficam de fora). Só contam os grupos que
     aparecem no centro das imagens;
   - **recurso:** se o chão não aparecer (poucas câmaras muito perto da mesa), usa-se o plano do centro das imagens:
     dos planos paralelos fica o mais baixo que não esteja mais de 35 cm abaixo do píxel central, e a componente
     ligada (DBSCAN) dos pontos a menos de 1.5 cm desse plano.
3. O plano é reajustado por SVD e o retângulo de área mínima do tampo dá o
   **referencial da mesa** (origem no centro, x ao longo do comprimento, z para cima) e os seus limites.
4. Ficam os pontos entre 8 mm e 60 cm acima da mesa e dentro dos limites; voxel de 4 mm e remoção de outliers.
5. **DBSCAN** (eps = 1.5 cm) separa os objetos. Se há menos clusters que caixas, as caixas extra vão para o
   cluster com maior área por caixa.
6. **Cor:** dentro de um cluster com várias caixas, k-means na **cromaticidade normalizada**
   r = R/(R+G+B), g = G/(R+G+B). A cor só é usada se os grupos estiverem bem separados (distância entre centros
   > 4× a dispersão).
   Uma sombra multiplica R, G e B pelo mesmo fator, que se cancela na divisão, por isso todas as faces de uma caixa têm
   a mesma cromaticidade. A primeira versão usava (a\*, b\*) do espaço Lab, que não é invariante à escala: nas faces
   escuras (a\*, b\*) aproximam-se de zero. Numa cena de teste, um azul e um roxo deixaram de se distinguir
   (IoU 0.00–0.58); com (r, g) passaram a 0.991–0.998.

A avaliação usa as caixas verdadeiras com margem de 5 mm. A precisão fica acima de 0.98. O recall (~0.66) é mais
baixo porque a referência inclui os pontos dos primeiros 8 mm junto à mesa e a margem de 5 mm, que a segmentação
retira de propósito.

## Parte 3 — Ajuste das caixas

### Método base: bounding box
`get_minimal_oriented_bounding_box()` sobre cada conjunto de pontos, depois de remover outliers (estatístico + maior
componente ligada). O eixo mais alinhado com o z da mesa passa a ser a altura. Se a caixa assenta na mesa, a base é
posta na mesa; se está empilhada (decidido na estimativa inicial), a base é o fundo da bounding box.

### Método principal: otimização
Parâmetros por caixa: `x, y, yaw, length, width, height, z` (referencial da mesa), em que `z` é a altura da base acima
da mesa (0 para uma caixa na mesa, o topo da caixa de baixo para uma caixa empilhada). No código `z` vem em último lugar
para os índices dos outros parâmetros não mudarem. Nas caixas assentes na mesa `z` fica fixo em 0 (não é otimizado):
se fosse livre, a caixa podia subir alguns mm sem custo, porque a segmentação corta os pontos até 8 mm acima da mesa.
Resíduos (`least_squares`, perda `soft_l1`):

1. **Dados → modelo:** distância de cada ponto à face mais próxima **com orientação compatível com a sua normal**,
   na caixa que melhor o explica. A associação (argmin) é recalculada **em cada avaliação** da função objetivo.
2. **Modelo → dados:** amostras nas faces visíveis de cada caixa têm de ter pontos medidos por perto
   (amostras dentro ou a < 2 cm de outra caixa não contam: são faces de contacto, invisíveis).
3. **Sobreposição:** raiz da área de interseção entre cada par de caixas (`cv2.rotatedRectangleIntersection`), só quando
   os intervalos de altura `[z, z+height]` também se cruzam (uma caixa empilhada está por cima da outra, não dentro dela).
   **Apoio:** uma caixa empilhada tem de assentar no topo da caixa mais alta que está por baixo dela.
4. **Cor (Nível 4):** cada caixa tem uma cor; o ponto escolhe a caixa que minimiza
   `d_geom² + (λ·d_cor)²`, com λ = σ_geom / σ_cor (cada erro dividido pelo seu desvio padrão, para as duas grandezas
   ficarem na mesma unidade).

**Pesos dos termos (Parte 06, Ex. 3):** na aula, os 4 erros de brilho/contraste são multiplicados por √n para valerem
tanto como os n erros de píxel. Aqui é igual: os resíduos de **sobreposição** e de **apoio** (um por par de caixas / por
caixa empilhada) são multiplicados por √M, com M o número de pontos. Os de **cobertura** são muitos (S amostras por
caixa), por isso o fator é √(M / (N·S)): o conjunto da cobertura pesa tanto como o conjunto dos pontos.

**Estimativa inicial:** retângulo de área mínima de cada cluster/grupo de cor. Para caixas da mesma cor em contacto:
pontos do topo agrupados por altura (DBSCAN 3D) e, se faltarem grupos, divisão do topo em duas partes, repetida até haver
tantas partes como caixas. Há dois métodos de divisão:
- **k-means no plano:** divide a área ao meio, sem olhar ao contorno;
- **degraus do contorno:** percorre o topo ao longo do seu eixo principal, em fatias de 5 mm, e corta onde os limites
  laterais mudam de repente, que é onde acaba uma caixa e começa a outra.

**Caixas empilhadas:** a contagem de caixas por cluster também conta os topos a alturas diferentes (uma pilha tem
pouca área vista de cima, mas 2 topos → pelo menos 2 caixas). Na estimativa inicial, uma caixa está por cima de outra
se o topo da outra está ≥ 2 cm mais abaixo e os retângulos vistos de cima se sobrepõem em ≥ 50% da área do mais
pequeno; a base fica no topo da caixa de baixo. (Caixas encostadas lado a lado quase não se sobrepõem vistas de cima.)
Resultados com as poses verdadeiras das câmaras (`simulator/configs/stack_*.json`):

| cena | caixas | IoU bounding box | IoU otimização | erro máx. de z (otimização) |
|---|---|---|---|---|
| stack_1 | pilha de 2, mesma cor, a de cima rodada | 0.87 / 0.82 | **0.99 / 0.99** | 0.2 mm |
| stack_2 | pilha de 2 com cores diferentes + 1 isolada | 0.87 / 0.88 / 0.86 | **0.98 / 0.99 / 0.95** | 0.7 mm |
| stack_3 | pilha de 3, mesma cor, a de cima rodada | 0.88 / 0.82 / 0.82 | **0.99 / 0.96 / 0.99** | 3.0 mm |

Caso impossível: duas caixas da mesma cor com exatamente o mesmo retângulo, uma em cima da outra. Por fora parecem
uma só caixa alta e a junta entre elas não deixa nenhum vestígio geométrico nem de cor (como no `level_3_aligned`).

**Multi-start:** o programa corre a otimização a partir das duas estimativas e fica com a que acaba com **menor custo**.
Os custos são comparáveis, porque a função objetivo e os pontos são os mesmos.
Numa cena de teste com 3 caixas da mesma altura e cor em fila, o k-means pôs as fronteiras até 11 cm fora do sítio e a
otimização ficou presa num mínimo local (IoU 0.21–0.60, custo 4810). A partir dos degraus, o IoU passou a 0.945–0.991
(custo 2280).

Resultados do sistema completo (registo estimado) nas cenas fornecidas, com ruído.
Tabelas completas (erro por câmara, por caixa, precisão/recall) em `results_offline/results.md`
(geradas com `python savi/trabalho1_savi.py --evaluate_scenes ...`), figuras em `results_offline/` (`*_top_view.png` e `*_optimization.png`).

| cena | IoU bounding box | IoU otimização | erro máx. de posição (otimização) |
|---|---|---|---|
| level_1 | 0.907 | **0.995** | 0.3 mm |
| level_2 | 0.882 | **0.993** | 0.3 mm |
| level_3_heights | 0.800 | **0.973** | 2.1 mm |
| level_3_rotated | 0.651 | **0.988** | 0.6 mm |
| level_3_widths | 0.527 | **0.978** | 2.4 mm |
| level_3_aligned | 0.606 | 0.645 | 52 mm (caso impossível) |
| level_4 | 0.900 | **0.981** | 1.6 mm |

Registo: erro máximo de 1.2 a 3.6 mm e 0.04° a 0.23° por câmara, conforme a cena.

A bounding box sobrestima sempre as dimensões em 4–11 mm, porque se encosta aos pontos mais extremos, que são os
mais ruidosos. A otimização erra menos de 1 mm nas caixas isoladas. Nos níveis 3 a bounding box é muito pior,
porque depende da divisão inicial dos pontos entre as caixas, que é apenas aproximada. A otimização corrige essa divisão.

**Questões:**
- *Uma caixa pode crescer sem que o erro aumente?* Sim, só com o termo dados→modelo: se crescer para espaço vazio,
  nenhum ponto fica mais longe da sua superfície. Também quando duas caixas se tocam: no nível 3 (alturas diferentes),
  a caixa baixa encolhia ~7 mm porque os pontos do seu topo junto à caixa alta eram "explicados" pela face lateral desta.
  Resolvido com o termo modelo→dados (cobertura) e com a compatibilidade das normais.
- *Duas caixas podem ocupar o mesmo espaço?* Se for permitido, duas caixas podem colapsar sobre o mesmo grupo de
  pontos (ambas explicam os mesmos pontos com custo zero) e outra parte da nuvem fica sem caixa. O termo de
  sobreposição impede isto.
- *Porque não é preciso um ciclo exterior?* Porque a associação ponto→caixa é um `min` dentro da função de custo:
  o custo de cada ponto é a distância à caixa que melhor o explica, que é contínua nos parâmetros. O otimizador vê
  diretamente o efeito de mudar de associação, sem fixar associações erradas do início.
- *Que pontos determinam cada parâmetro?* A altura é determinada pelos pontos do topo; o comprimento pelos pontos
  das faces das extremidades. Um ponto no meio do topo não ajuda a estimar o comprimento (a sua distância à
  superfície não muda se a caixa crescer no plano).
- *Porque é a bounding box tão sensível ao ruído?* Usa só os pontos extremos (o invólucro convexo): um único
  *flying pixel* ou um ponto ruidoso numa face alarga a caixa. O ajuste usa todos os pontos com uma perda robusta.

## Níveis de dificuldade

- **Nível 1 e 2:** um cluster por caixa; ambos os métodos funcionam, a otimização é mais exata nas dimensões.
- **Nível 3:** um só cluster. Casos **impossíveis**: caixas **iguais e perfeitamente alinhadas** (`level_3_aligned`)
  — a união das duas é um paralelepípedo perfeito com o topo plano, e qualquer posição da fronteira entre elas
  explica os pontos igualmente bem; a fronteira não deixa nenhum vestígio geométrico nem de cor.
  Com alturas diferentes, larguras diferentes ou rotações, o contorno da união tem um degrau ou um ângulo que fixa a fronteira.
  Os resultados confirmam isto: nesses três casos o IoU é 0.97–0.99. No caso alinhado, a fronteira fica ~10 cm
  fora do sítio (IoU 0.645), embora as dimensões exteriores da união (largura e altura) estejam certas ao milímetro.
- **Nível 4:** a cor separa as caixas; a abordagem mais robusta a pontos com cor errada é pôr a cor na função
  objetivo (um ponto com cor errada mas bem explicado geometricamente continua bem associado), em vez de separar
  primeiro por cor (os erros de cor passam diretamente para a bounding box).

## Dificuldades

- OpenCV no Windows com caminhos com acentos (imagens não gravadas/lidas) — resolvido com `imencode/imdecode`.
- SIFT e FPFH insuficientes para o registo com 45° entre câmaras — resolvido com ASIFT, RANSAC guiado e escolha de
  hipóteses pelo alinhamento com cor.
- Na deteção da mesa, o chão tinha mais pontos que o tampo. Agora o chão é encontrado primeiro, de propósito, e a mesa procura-se entre 0.5 e 1.0 m acima dele (Parte 08, Ex. 4).
- Mínimos locais e ambiguidades da função objetivo nas caixas em contacto — ver Parte 3.
