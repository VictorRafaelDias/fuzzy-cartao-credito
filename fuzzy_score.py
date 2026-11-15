"""Sistema de inferência fuzzy (estilo Mamdani) para um score de dependência do crédito.

Fluxo clássico da lógica fuzzy:
  1. Fuzzificação: cada valor de entrada vira graus de pertinência a "baixa", "média" e "alta".
  2. Regras SE-ENTÃO: combinam as entradas (E = mínimo) e dizem qual a dependência resultante.
  3. Agregação: o conjunto de saída de cada regra é "cortado" na força da regra; juntam-se todos (máximo).
  4. Defuzzificação: o centro de gravidade (centróide) da área resultante é o score (0 a 100).

As curvas e as regras abaixo são definidas por critério de especialista (não são aprendidas dos
dados) e não foram validadas contra inadimplência real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TERMOS = ("baixa", "média", "alta")

# Entradas: nome interno -> descrição. `pontos` são os cortes (q1, q2, q3) das curvas:
# "baixa" vale 1 até q1 e cai a 0 em q2; "média" é um triângulo q1-q2-q3; "alta" sobe de q2 a q3.
ENTRADAS = {
    "uso": {
        "rotulo": "Utilização do limite",
        "formula": "Saldo ÷ Limite de crédito",
        "pontos": (0.10, 0.40, 0.80),
        "baixa": "usa pouco do limite",
        "alta": "usa quase todo o limite",
        "ajuda": "Quanto do limite do cartão está comprometido. 100% = limite todo usado.",
    },
    "pagamento": {
        "rotulo": "Taxa de pagamento integral",
        "formula": "Percentual das faturas pagas por inteiro (PRC_FULL_PAYMENT)",
        "pontos": (0.05, 0.30, 0.70),
        "baixa": "quase nunca paga a fatura inteira",
        "alta": "quase sempre paga a fatura inteira",
        "ajuda": "Em que parte das faturas o cliente pagou o valor total (e não só o mínimo).",
    },
    "saque": {
        "rotulo": "Frequência de saque",
        "formula": "Frequência de saque em dinheiro no crédito (CASH_ADVANCE_FREQUENCY)",
        "pontos": (0.05, 0.20, 0.45),
        "baixa": "quase nunca saca dinheiro no crédito",
        "alta": "saca dinheiro no crédito com frequência",
        "ajuda": "Índice de 0% (nunca saca) a 100% (saca sempre), calculado pela própria base.",
    },
    "compras": {
        "rotulo": "Frequência de compras",
        "formula": "Frequência de compras (PURCHASES_FREQUENCY)",
        "pontos": (0.15, 0.50, 0.85),
        "baixa": "quase não compra com o cartão",
        "alta": "compra com o cartão o tempo todo",
        "ajuda": "Índice de 0% (nunca compra) a 100% (compra sempre), calculado pela própria base.",
    },
}

PONTOS_PADRAO = {k: tuple(v["pontos"]) for k, v in ENTRADAS.items()}

CLASSES = ["Baixa dependência", "Média dependência", "Alta dependência"]
COR_CLASSE = ["#2ca02c", "#eeca3b", "#d62728"]
COR_TERMO = {"baixa": "#2ca02c", "média": "#d62728", "alta": "#1f77b4"}

# Conjuntos da saída (score 0-100): mesma forma das entradas.
SAIDA_PONTOS = (20.0, 50.0, 80.0)
SAIDA_ROTULO = "Dependência do crédito"

# Regras: (condições, consequente). Condições = [(entrada, termo), ...] ligadas por E.
REGRAS = [
    ([("saque", "alta"), ("pagamento", "baixa")], "alta"),
    ([("uso", "alta"), ("pagamento", "baixa")], "alta"),
    ([("saque", "alta"), ("compras", "baixa")], "alta"),
    ([("saque", "média"), ("uso", "alta")], "alta"),
    ([("uso", "média"), ("pagamento", "média")], "média"),
    ([("saque", "média"), ("pagamento", "média")], "média"),
    ([("compras", "alta"), ("pagamento", "média")], "média"),
    ([("uso", "alta"), ("pagamento", "média")], "média"),
    ([("saque", "baixa"), ("pagamento", "baixa"), ("uso", "média")], "média"),
    ([("saque", "baixa"), ("pagamento", "alta")], "baixa"),
    ([("uso", "baixa"), ("saque", "baixa")], "baixa"),
    ([("compras", "alta"), ("pagamento", "alta")], "baixa"),
    ([("uso", "baixa"), ("pagamento", "alta")], "baixa"),
]


# Raciocínio de cada regra (mesma ordem de REGRAS).
REGRAS_PORQUE = [
    "Saca dinheiro com frequência e não quita a fatura: financia-se com o cartão.",
    "Limite quase todo usado e fatura paga só em parte: carrega dívida.",
    "Usa o cartão só para sacar, quase não compra: é empréstimo, não meio de pagamento.",
    "Saca de vez em quando e já está com o limite apertado.",
    "Meio-termo no uso do limite e no pagamento da fatura.",
    "Saca às vezes e paga a fatura em parte.",
    "Compra bastante, mas paga só parte da fatura: pode estar financiando as compras.",
    "Limite bastante usado, mas ainda paga parte da fatura.",
    "Não saca, mas também não quita a fatura e tem saldo em aberto.",
    "Não saca e costuma quitar a fatura: usa como meio de pagamento.",
    "Usa pouco do limite e não saca: baixa necessidade de crédito.",
    "Compra bastante e paga a fatura inteira: usa como meio de pagamento.",
    "Pouco saldo em aberto e paga a fatura inteira.",
]

# Eixo horizontal de cada entrada (para os gráficos).
EIXOS = {
    "uso": "% do limite usado",
    "pagamento": "% das faturas quitadas",
    "saque": "índice de saque (%)",
    "compras": "índice de compras (%)",
}


def texto_cond(entrada: str, termo: str) -> str:
    return f"{ENTRADAS[entrada]['rotulo']} é {termo.upper()}"


def texto_condicao(i: int) -> str:
    cond, _ = REGRAS[i]
    return " E ".join(f"{ENTRADAS[e]['rotulo'].lower()} é {t.upper()}" for e, t in cond)


def texto_regra(i: int) -> str:
    return f"SE {texto_condicao(i)} ENTÃO dependência é {REGRAS[i][1].upper()}"


def valores_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as 4 entradas (todas entre 0 e 1) a partir da base."""
    limite = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median()).replace(0, np.nan)
    uso = (df["BALANCE"] / limite).fillna(1.0).clip(0, 1)
    return pd.DataFrame({
        "uso": uso,
        "pagamento": df["PRC_FULL_PAYMENT"].fillna(0).clip(0, 1),
        "saque": df["CASH_ADVANCE_FREQUENCY"].fillna(0).clip(0, 1),
        "compras": df["PURCHASES_FREQUENCY"].fillna(0).clip(0, 1),
    })


def graus(x, pontos) -> np.ndarray:
    """Graus de pertinência (baixa, média, alta) de cada valor de x. Retorna (n, 3)."""
    q1, q2, q3 = pontos
    x = np.asarray(x, dtype=float)
    return np.column_stack([
        np.interp(x, [q1, q2], [1.0, 0.0]),
        np.interp(x, [q1, q2, q3], [0.0, 1.0, 0.0], left=0.0, right=0.0),
        np.interp(x, [q2, q3], [0.0, 1.0]),
    ])


def conjuntos_saida(grade: np.ndarray) -> np.ndarray:
    """Conjuntos de saída (3, len(grade))."""
    return graus(grade, SAIDA_PONTOS).T


def inferir(valores: pd.DataFrame, com_curva: bool = False, pontos: dict | None = None) -> dict:
    """Roda o sistema fuzzy para todos os clientes de `valores` (saída de valores_entrada).
    `pontos` permite trocar os cortes das curvas de entrada (chave -> (q1, q2, q3))."""
    pontos = {**PONTOS_PADRAO, **(pontos or {})}
    mu = {k: graus(valores[k].to_numpy(), pontos[k]) for k in ENTRADAS}
    forca = np.column_stack([
        np.min([mu[e][:, TERMOS.index(t)] for e, t in cond], axis=0) for cond, _ in REGRAS
    ])
    grade = np.linspace(0, 100, 201)
    saida = conjuntos_saida(grade)
    agg = np.zeros((len(valores), len(grade)))
    cortes = []
    for r, (_, cons) in enumerate(REGRAS):
        corte = np.minimum(forca[:, [r]], saida[TERMOS.index(cons)][None, :])
        agg = np.maximum(agg, corte)
        if com_curva:
            cortes.append(corte)
    area = agg.sum(axis=1)
    score = np.where(area > 1e-9, (agg * grade).sum(axis=1) / np.maximum(area, 1e-9), 50.0)
    pert_saida = graus(score, SAIDA_PONTOS)
    resultado = {
        "mu": mu, "forca": forca, "score": score,
        "pert_saida": pert_saida, "classe": pert_saida.argmax(axis=1),
        "grade": grade, "saida": saida,
    }
    if com_curva:
        resultado["agregada"] = agg
        resultado["cortes"] = cortes
    return resultado
