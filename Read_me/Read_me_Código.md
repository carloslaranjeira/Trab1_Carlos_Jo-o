# SAVIBoxes — Trabalho Prático 1 de SAVI

Estimação da pose e das dimensões de caixas sobre uma mesa a partir de várias imagens RGB-D.

> No repositório GitHub este ficheiro deve chamar-se `README.md`.
> (Aqui chama-se `RELATORIO.md` porque, no Windows, `README.md` e o `readme.md` do enunciado são o mesmo ficheiro.)

## Como correr

```
pip install -r requirements.txt
cd simulator
python savi_box_simulator.py --batch configs/level_*.json      # gera as cenas em simulator/scenes
cd ..
python main.py --scene simulator/scenes/level_4 --num_boxes 3                          # escreve result.json
python main.py --scene simulator/scenes/level_4 --num_boxes 3 --evaluate --visualize   # + erros e janelas 3D
python evaluate.py simulator/scenes/level_*                                            # tabelas em results/results.md
python evaluate.py simulator/scenes/level_* --gt_registration                          # isola segmentação + ajuste
```

Opções do `main.py`: `--method bbox` (usa a bounding box no result.json), `--no_color`, `--no_loop_closure`, `--output`.
Além do `result.json`, são gravadas `top_view.png` (caixas vistas de cima) e `optimization.png` (evolução da otimização).

**Nota:** foi feita uma alteração de uma linha no simulador (`cv2.imwrite` → `cv2.imencode(...).tofile(...)`), porque
no Windows o OpenCV não escreve ficheiros em caminhos com acentos (como "Trabalho prático") e as imagens RGB não eram gravadas.

## Estrutura do código

| Ficheiro | Conteúdo |
|---|---|
| `main.py` | Programa principal (pipeline completo, escrita do `result.json`) |
| `evaluate.py` | Corre várias cenas e gera as tabelas de erros dos dois métodos |
| `savi/scene_io.py` | Leitura das cenas, "imagem de pontos 3D" de cada câmara, escrita do resultado |
| `savi/geometry.py` | Transformações, Kabsch, distância com sinal a uma caixa, cantos das caixas |
| `savi/registration.py` | Parte 1: ASIFT + RANSAC + ICP + grafo de poses + agregação |
| `savi/segmentation.py` | Parte 2: plano da mesa, referencial da mesa, recorte, DBSCAN, cor |
| `savi/bbox_method.py` | Parte 3: método base (bounding box orientada) |
| `savi/fitting.py` | Parte 3: estimativa inicial e ajuste por otimização (`least_squares`) |
| `savi/evaluation.py` | Erros das câmaras, precisão/recall, erros das caixas e IoU 3D |
| `savi/visualization.py` | Janelas Open3D (evolução da otimização, etc.) e figuras matplotlib |

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

## Parte 2 — Segmentação

1. As poses das câmaras dão o centro do anel de câmaras (por cima da mesa) e a direção "cima".
2. RANSAC repetido (`segment_plane`) dentro do anel: dos planos horizontais grandes, escolhe-se o **mais alto**
   (o chão pode ter mais pontos que a mesa, mas está por baixo; os topos das caixas são pequenos demais).
   Fica a maior componente ligada, reajusta-se o plano por SVD e o retângulo de área mínima do tampo dá o
   **referencial da mesa** (origem no centro, x ao longo do comprimento, z para cima) e os seus limites.
3. Ficam os pontos entre 8 mm e 60 cm acima da mesa e dentro dos limites; voxel de 4 mm e remoção de outliers.
4. **DBSCAN** (eps = 1.5 cm) separa os objetos. Se há menos clusters que caixas, as caixas extra vão para o
   cluster com maior área por caixa.
5. **Cor:** dentro de um cluster com várias caixas, k-means na **cromaticidade normalizada**
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
componente ligada). O eixo mais alinhado com o z da mesa passa a ser a altura, e a base é posta na mesa.

### Método principal: otimização
Parâmetros por caixa: `x, y, yaw, length, width, height` (referencial da mesa). Resíduos (`least_squares`, perda `soft_l1`):

1. **Dados → modelo:** distância de cada ponto à face mais próxima **com orientação compatível com a sua normal**,
   na caixa que melhor o explica. A associação (argmin) é recalculada **em cada avaliação** da função objetivo.
2. **Modelo → dados:** amostras nas faces visíveis de cada caixa têm de ter pontos medidos por perto
   (amostras dentro ou a < 2 cm de outra caixa não contam: são faces de contacto, invisíveis).
3. **Sobreposição:** raiz da área de interseção entre cada par de caixas (`cv2.rotatedRectangleIntersection`).
4. **Cor (Nível 4):** cada caixa tem uma cor; o ponto escolhe a caixa que minimiza
   `d_geom² + (λ·d_cor)²`, com λ = σ_geom / σ_cor (cada erro dividido pelo seu desvio padrão — Parte 06, Ex. 3).

**Estimativa inicial:** retângulo de área mínima de cada cluster/grupo de cor. Para caixas da mesma cor em contacto:
pontos do topo agrupados por altura (DBSCAN 3D) e, se faltarem grupos, divisão do topo em duas partes, repetida até haver
tantas partes como caixas. Há dois métodos de divisão:
- **k-means no plano:** divide a área ao meio, sem olhar ao contorno;
- **degraus do contorno:** percorre o topo ao longo do seu eixo principal, em fatias de 5 mm, e corta onde os limites
  laterais mudam de repente, que é onde acaba uma caixa e começa a outra.

**Multi-start:** o programa corre a otimização a partir das duas estimativas e fica com a que acaba com **menor custo**.
Os custos são comparáveis, porque a função objetivo e os pontos são os mesmos.
Numa cena de teste com 3 caixas da mesma altura e cor em fila, o k-means pôs as fronteiras até 11 cm fora do sítio e a
otimização ficou presa num mínimo local (IoU 0.21–0.60, custo 4810). A partir dos degraus, o IoU passou a 0.945–0.991
(custo 2280).

Resultados do sistema completo (registo estimado) nas cenas fornecidas, com ruído.
Tabelas completas (erro por câmara, por caixa, precisão/recall) em [results/results.md](results/results.md),
figuras em `results/` (`*_top_view.png` e `*_optimization.png`).

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
- Na deteção da mesa, o chão tinha mais pontos que o tampo — escolhe-se o plano grande mais alto.
- Mínimos locais e ambiguidades da função objetivo nas caixas em contacto — ver Parte 3.
