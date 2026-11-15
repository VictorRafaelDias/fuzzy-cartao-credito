"""Fuzzy C-Means (implementação própria em NumPy) + índices de validação.

Usado tanto pelo script de terminal (`fuzzy_cartao.py`) quanto pelo app
Streamlit (`app_fuzzy_cartao.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

EPS = 1e-10

# Rótulos em português para as 17 variáveis da base.
ROTULOS = {
    "BALANCE": "Saldo",
    "BALANCE_FREQUENCY": "Freq. de atualização do saldo",
    "PURCHASES": "Compras",
    "ONEOFF_PURCHASES": "Compras à vista",
    "INSTALLMENTS_PURCHASES": "Compras parceladas",
    "CASH_ADVANCE": "Saque no crédito",
    "PURCHASES_FREQUENCY": "Freq. de compras",
    "ONEOFF_PURCHASES_FREQUENCY": "Freq. de compras à vista",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "Freq. de compras parceladas",
    "CASH_ADVANCE_FREQUENCY": "Freq. de saque",
    "CASH_ADVANCE_TRX": "Nº de saques",
    "PURCHASES_TRX": "Nº de compras",
    "CREDIT_LIMIT": "Limite de crédito",
    "PAYMENTS": "Pagamentos",
    "MINIMUM_PAYMENTS": "Pagamento mínimo",
    "PRC_FULL_PAYMENT": "% da fatura paga integral",
    "TENURE": "Tempo de cartão (meses)",
}


def rotulo(coluna: str) -> str:
    return ROTULOS.get(coluna, coluna)


# --------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------
def preparar_dados(df: pd.DataFrame, colunas: list[str], usar_log: bool = True) -> np.ndarray:
    """Imputa faltantes (mediana), corrige frequências > 1, aplica log1p nas
    variáveis assimétricas e padroniza (z-score)."""
    X = df[colunas].astype(float).copy()
    X = X.fillna(X.median())
    for col in colunas:
        if "FREQUENCY" in col:
            X[col] = X[col].clip(upper=1.0)  # há valores 1.5 na base (erro de registro)
    if usar_log:
        for col in colunas:
            if X[col].min() >= 0 and X[col].max() > 1:
                X[col] = np.log1p(X[col])
    desvio = X.std(ddof=0).replace(0, 1.0)
    return ((X - X.mean()) / desvio).to_numpy()


# --------------------------------------------------------------------------
# Fuzzy C-Means
# --------------------------------------------------------------------------
@dataclass
class ResultadoFCM:
    U: np.ndarray        # (n, c) graus de pertinência
    centros: np.ndarray  # (c, p) centróides no espaço padronizado
    J: float             # função objetivo
    n_iter: int
    m: float

    @property
    def rotulos(self) -> np.ndarray:
        return self.U.argmax(axis=1)


def _dist2(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    d2 = (X**2).sum(1)[:, None] - 2 * X @ V.T + (V**2).sum(1)[None, :]
    return np.maximum(d2, EPS)


def _atualizar_U(d2: np.ndarray, m: float) -> np.ndarray:
    # divide pelo menor valor da linha para evitar overflow quando m é próximo de 1
    inv = (d2 / d2.min(axis=1, keepdims=True)) ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def _centros(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    Um = U**m
    return (Um.T @ X) / Um.sum(axis=0)[:, None]


def fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300,
        tol: float = 1e-5, seed: int = 0) -> ResultadoFCM:
    rng = np.random.default_rng(seed)
    U = rng.random((X.shape[0], c))
    U /= U.sum(axis=1, keepdims=True)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        V = _centros(X, U, m)
        U_novo = _atualizar_U(_dist2(X, V), m)
        delta = np.abs(U_novo - U).max()
        U = U_novo
        if delta < tol:
            break
    V = _centros(X, U, m)
    J = float(((U**m) * _dist2(X, V)).sum())
    return ResultadoFCM(U=U, centros=V, J=J, n_iter=n_iter, m=m)


def fcm_multi(X: np.ndarray, c: int, m: float = 2.0, n_init: int = 5,
              seed: int = 42) -> ResultadoFCM:
    """Roda várias inicializações e mantém a de menor função objetivo.
    Os grupos são reordenados do maior para o menor (tamanho), p/ rótulos estáveis."""
    melhor = min((fcm(X, c, m, seed=seed + i) for i in range(n_init)), key=lambda r: r.J)
    ordem = np.argsort(-np.bincount(melhor.rotulos, minlength=c))
    return ResultadoFCM(U=melhor.U[:, ordem], centros=melhor.centros[ordem],
                        J=melhor.J, n_iter=melhor.n_iter, m=m)


# --------------------------------------------------------------------------
# Índices de validação
# --------------------------------------------------------------------------
def indices(X: np.ndarray, res: ResultadoFCM, seed: int = 42,
            amostra_sil: int = 3000) -> dict:
    U, V, m = res.U, res.centros, res.m
    n, c = U.shape
    pc = float((U**2).sum() / n)                                   # Partition Coefficient
    pe = float(-(U * np.log(np.maximum(U, EPS))).sum() / n)         # Partition Entropy
    d_centros = _dist2(V, V)
    np.fill_diagonal(d_centros, np.inf)
    xb = float(((U**m) * _dist2(X, V)).sum() / (n * d_centros.min()))  # Xie-Beni
    rot = res.rotulos
    sil = float("nan")
    if len(np.unique(rot)) > 1:
        sil = float(silhouette_score(X, rot, sample_size=min(amostra_sil, n), random_state=seed))
    return {"c": c, "J": res.J, "PC": pc, "PE": pe, "XB": xb, "Silhouette": sil}


def varredura(X: np.ndarray, m: float, faixa=range(2, 9), n_init: int = 5,
              seed: int = 42) -> pd.DataFrame:
    return pd.DataFrame([indices(X, fcm_multi(X, c, m, n_init, seed), seed) for c in faixa])


def sugerir_c(tabela: pd.DataFrame) -> int:
    """Sugere c pela média dos rankings de Silhouette (maior) e Xie-Beni (menor).
    PC e PE ficam de fora porque melhoram quase sempre quando c diminui."""
    r_sil = tabela["Silhouette"].rank(ascending=False)
    r_xb = tabela["XB"].rank(ascending=True)
    pontos = (r_sil + r_xb) / 2
    return int(tabela.loc[pontos.idxmin(), "c"])


def comparar_kmeans(X: np.ndarray, c: int, rotulos_fcm: np.ndarray, seed: int = 42,
                    amostra_sil: int = 3000) -> dict:
    km = KMeans(n_clusters=c, n_init=10, random_state=seed).fit(X)
    n = X.shape[0]
    return {
        "silhouette_kmeans": float(silhouette_score(X, km.labels_, sample_size=min(amostra_sil, n), random_state=seed)),
        "silhouette_fcm": float(silhouette_score(X, rotulos_fcm, sample_size=min(amostra_sil, n), random_state=seed)),
        "ari": float(adjusted_rand_score(km.labels_, rotulos_fcm)),
    }


# --------------------------------------------------------------------------
# Interpretação
# --------------------------------------------------------------------------
def perfil_original(df: pd.DataFrame, colunas: list[str], rotulos: np.ndarray) -> pd.DataFrame:
    """Média (na escala original) de cada variável por grupo."""
    perfil = df[colunas].groupby(rotulos).mean()
    perfil.index = [f"Grupo {i + 1}" for i in perfil.index]
    return perfil


def descrever_grupos(centros: np.ndarray, colunas: list[str], top: int = 3) -> list[list[tuple[str, float]]]:
    """Para cada grupo, as `top` variáveis mais distantes da média (em desvios-padrão)."""
    saida = []
    for v in centros:
        idx = np.argsort(-np.abs(v))[:top]
        saida.append([(rotulo(colunas[i]), float(v[i])) for i in idx])
    return saida


def fmt_pct(x: float, casas: int = 1) -> str:
    return f"{x * 100:.{casas}f}%".replace(".", ",")


# --------------------------------------------------------------------------
# Texto explicativo (nomes e frases automáticas para cada grupo)
# --------------------------------------------------------------------------
DESCRICOES = {
    "BALANCE": "Saldo disponível na conta do cartão no período.",
    "BALANCE_FREQUENCY": "Com que frequência o saldo é atualizado (0 a 100%).",
    "PURCHASES": "Valor total gasto em compras.",
    "ONEOFF_PURCHASES": "Valor gasto em compras feitas de uma só vez (à vista).",
    "INSTALLMENTS_PURCHASES": "Valor gasto em compras parceladas.",
    "CASH_ADVANCE": "Dinheiro sacado usando o limite do cartão de crédito.",
    "PURCHASES_FREQUENCY": "Com que frequência o cliente faz compras (0 a 100%).",
    "ONEOFF_PURCHASES_FREQUENCY": "Com que frequência faz compras à vista (0 a 100%).",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "Com que frequência faz compras parceladas (0 a 100%).",
    "CASH_ADVANCE_FREQUENCY": "Com que frequência saca dinheiro no crédito (0 a 100%).",
    "CASH_ADVANCE_TRX": "Número de saques no crédito.",
    "PURCHASES_TRX": "Número de compras realizadas.",
    "CREDIT_LIMIT": "Limite de crédito do cartão.",
    "PAYMENTS": "Valor total pago pelo cliente.",
    "MINIMUM_PAYMENTS": "Valor dos pagamentos mínimos feitos.",
    "PRC_FULL_PAYMENT": "Percentual de vezes em que a fatura foi paga por inteiro.",
    "TENURE": "Há quantos meses o cliente tem o cartão.",
}


def eh_proporcao(coluna: str) -> bool:
    """Variáveis que vão de 0 a 1 e devem ser exibidas como porcentagem."""
    return "FREQUENCY" in coluna or coluna == "PRC_FULL_PAYMENT"


def formatar_valor(coluna: str, valor: float) -> str:
    if eh_proporcao(coluna):
        return fmt_pct(valor, 0)
    if abs(valor) >= 100:
        return f"{valor:,.0f}".replace(",", ".")
    return f"{valor:.1f}".replace(".", ",")


def nomear_grupo(centro: np.ndarray, colunas: list[str]) -> str:
    """Dá um nome ao grupo a partir do centróide (em desvios-padrão)."""
    z = {col: float(v) for col, v in zip(colunas, centro)}
    g = lambda col: z.get(col, 0.0)  # variável fora da seleção conta como 0

    if g("CASH_ADVANCE") > 0.5 and g("PURCHASES") < -0.5:
        return "Dependentes de saque"
    if g("PURCHASES") > 0.6 and g("PURCHASES_TRX") > 0.6 and g("ONEOFF_PURCHASES") > 0.5:
        return "Compradores intensos"
    if g("INSTALLMENTS_PURCHASES") > 0.5 and g("PURCHASES_FREQUENCY") > 0.4:
        return "Compradores que parcelam"
    if g("PURCHASES_FREQUENCY") < -0.3 and g("CASH_ADVANCE_FREQUENCY") < -0.3:
        return "Pouco ativos"
    # sem regra especial: usa a característica mais marcante
    col, val = max(z.items(), key=lambda kv: abs(kv[1]))
    return f"{rotulo(col)} {'alto' if val > 0 else 'baixo'}"


_CLAUSULAS = [
    # (variável, limiar, texto se alto, texto se baixo)
    ("PURCHASES", 0.5, "gastam muito em compras", "gastam pouco em compras"),
    ("PURCHASES_FREQUENCY", 0.5, "compram com frequência", "raramente compram"),
    ("ONEOFF_PURCHASES", 0.5, "fazem compras grandes à vista", "quase não compram à vista"),
    ("INSTALLMENTS_PURCHASES", 0.5, "parcelam bastante", "quase não parcelam"),
    ("CASH_ADVANCE", 0.5, "sacam muito dinheiro no crédito", "quase não sacam dinheiro no crédito"),
    ("BALANCE", 0.5, "têm saldo alto", "têm saldo baixo"),
    ("CREDIT_LIMIT", 0.4, "têm limite alto", "têm limite baixo"),
    ("PAYMENTS", 0.4, "pagam valores altos", "pagam valores baixos"),
    ("PRC_FULL_PAYMENT", 0.4, "pagam a fatura inteira com mais frequência", "raramente pagam a fatura inteira"),
    ("BALANCE_FREQUENCY", 0.4, "movimentam a conta com regularidade", "movimentam pouco a conta"),
]


def frase_grupo(centro: np.ndarray, colunas: list[str]) -> str:
    """Frase em português: o que os clientes deste grupo costumam fazer."""
    z = {col: float(v) for col, v in zip(colunas, centro)}
    partes = []
    for col, lim, alto, baixo in _CLAUSULAS:
        if col in z and z[col] >= lim:
            partes.append(alto)
        elif col in z and z[col] <= -lim:
            partes.append(baixo)
    if not partes:
        return "Clientes com comportamento próximo da média geral, sem uma característica muito marcante."
    return "Clientes que " + ", ".join(partes[:-1]) + (" e " if len(partes) > 1 else "") + partes[-1] + "."


def principais_diferencas(centro: np.ndarray, colunas: list[str], medias_grupo: pd.Series,
                          media_geral: pd.Series, top: int = 5) -> pd.DataFrame:
    """Variáveis que mais separam o grupo da média geral, em unidades originais."""
    idx = np.argsort(-np.abs(centro))[:top]
    linhas = []
    for i in idx:
        col = colunas[i]
        g, t = float(medias_grupo[col]), float(media_geral[col])
        linhas.append({
            "Variável": rotulo(col),
            "Neste grupo": formatar_valor(col, g),
            "Média geral": formatar_valor(col, t),
            "Em relação à média": ("acima" if centro[i] > 0 else "abaixo")
                                   + (f" ({g / t:.1f}×)".replace(".", ",") if t > 0 else ""),
        })
    return pd.DataFrame(linhas)
