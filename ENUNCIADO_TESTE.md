# Teste — Dia da apresentação (simulação n.º 2)

Tem **20 minutos** a partir do momento em que começa.

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador, de vários níveis, e **não têm**
`ground_truth.json` nem `config.json`. O número de caixas de cada cena é:

| cena | caixas |
|---|---|
| cena_1 | 2 |
| cena_2 | 1 |
| cena_3 | 2 |
| cena_4 | 3 |
| cena_5 | 3 |
| cena_6 | 3 |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, do código e de um `result.json` por cena,
   em `resultados_teste/<cena>/result.json` (por exemplo `resultados_teste/cena_1/result.json`);
4. Ao fim dos 20 minutos, parar. O que for enviado depois não conta.

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.

Bom trabalho!
