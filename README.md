# Agrupamento fuzzy de clientes de cartão de crédito

Aplicação de **Fuzzy C-Means** (implementação própria em NumPy) à base
[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
(Kaggle): 8.950 clientes e 17 variáveis de comportamento de uso do cartão.

Cada cliente recebe um **grau de pertinência** a cada grupo, em vez de um rótulo único.

## Como rodar

```bash
pip install -r requirements.txt
python fuzzy_cartao.py                          # versão de terminal, salva em ./resultados
python -m streamlit run app_fuzzy_cartao.py     # app interativo
```

## Estrutura

| Arquivo | Função |
|---|---|
| `fcm.py` | pré-processamento, Fuzzy C-Means, índices (PC, PE, Xie-Beni, silhouette), comparação com K-Means |
| `fuzzy_cartao.py` | protótipo de terminal (CSV, TXT e PNG em `resultados/`) |
| `app_fuzzy_cartao.py` | app Streamlit (5 abas) |
| `data/CC_GENERAL.csv` | base original |

## Metodologia

1. `CUST_ID` é descartado. `MINIMUM_PAYMENTS` (313 faltantes) e `CREDIT_LIMIT` (1) são imputados pela mediana.
2. `CASH_ADVANCE_FREQUENCY` tem valores até 1,5; as frequências são limitadas a 1.
3. `log(1 + x)` nas variáveis assimétricas (valores monetários e contagens) e depois z-score.
4. Fuzzy C-Means com várias inicializações, mantendo a de menor função objetivo.
5. Escolha de c (2 a 8) por Silhouette e Xie-Beni; PC e PE são mostrados como apoio.
6. Clientes de fronteira: pertinência máxima abaixo do limiar (padrão 60%).
7. Comparação com K-Means (silhouette e Adjusted Rand Index).

## Resultados (m = 1,5; semente 42; 5 inicializações)

Índices por número de grupos:

| c | PC | PE | Xie-Beni | Silhouette |
|---|---|---|---|---|
| 2 | 0,745 | 0,401 | 0,725 | 0,243 |
| 3 | 0,624 | 0,655 | 1,088 | 0,225 |
| 4 | 0,550 | 0,848 | 1,171 | 0,210 |
| 5 | 0,527 | 0,951 | 0,824 | 0,218 |
| 6 | 0,512 | 1,029 | 0,657 | 0,216 |
| 7 | 0,476 | 1,127 | 1,180 | 0,176 |
| 8 | 0,458 | 1,200 | 1,078 | 0,166 |

- A sugestão automática (média dos rankings de Silhouette e Xie-Beni) escolhe **c = 2**:
  clientes que compram muito × clientes que compram pouco (12,6% na fronteira).
- Com **c = 4**: grupos de 2.591, 2.421, 2.029 e 1.909 clientes; 40,1% na fronteira (limiar de 60%).
  Silhouette 0,210 (K-Means: 0,219) e concordância com o K-Means (ARI) de 0,686.
- Os grupos se sobrepõem bastante (silhouette baixo), o que justifica a abordagem fuzzy.

## Achado importante: o fuzzificador

Com o valor clássico **m = 2** e as 17 variáveis, o Fuzzy C-Means **colapsa** nesta base:
os centróides ficam muito próximos da média geral, ~85% dos clientes têm pertinência máxima
abaixo de 60% e o índice de Xie-Beni explode para c ≥ 3. É o efeito conhecido do FCM em
dimensão alta. Com **m = 1,5** os grupos são interpretáveis, por isso esse é o padrão.
