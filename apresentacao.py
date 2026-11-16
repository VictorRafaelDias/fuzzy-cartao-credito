"""Apresentação passo a passo: lógica fuzzy aplicada a clientes de cartão de crédito.

Cada aba é um tópico: problema, dados, conceito, fuzzificação, regras, score, resultados e conclusões.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import fcm
import fuzzy_score as fs

ARQUIVO = Path(__file__).parent / "data" / "CC_GENERAL.csv"
CINZA = "#8b8f98"
COR_MD = {"baixa": "green", "média": "red", "alta": "blue"}  # cores do texto colorido do Streamlit
USADAS_NO_SCORE = {
    "BALANCE": "Utilização do limite (saldo ÷ limite)",
    "CREDIT_LIMIT": "Utilização do limite (saldo ÷ limite)",
    "PRC_FULL_PAYMENT": "Taxa de pagamento integral",
    "CASH_ADVANCE_FREQUENCY": "Frequência de saque",
    "PURCHASES_FREQUENCY": "Frequência de compras",
}
DEFINICAO_CLASSES = {
    "Baixa dependência": "Usa o cartão como **meio de pagamento**: usa pouco do limite, costuma pagar a fatura inteira e quase não saca dinheiro.",
    "Média dependência": "**Perfil misto**: usa uma parte do limite, paga a fatura em parte e às vezes saca dinheiro.",
    "Alta dependência": "Usa o cartão como **empréstimo**: limite quase todo comprometido, paga só parte da fatura e/ou saca dinheiro com frequência.",
}


def pct(x: float, casas: int = 0) -> str:
    return fcm.fmt_pct(x, casas)


def br(x: float, casas: int = 0) -> str:
    """Número no padrão brasileiro: milhar com ponto, decimal com vírgula."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# --------------------------------------------------------------------------
# Dados e cálculos (com cache)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar() -> pd.DataFrame:
    return pd.read_csv(ARQUIVO)


@st.cache_data(show_spinner=False)
def calcular(pontos_t: tuple) -> tuple[pd.DataFrame, dict]:
    """Entradas do score e resultado do sistema fuzzy para toda a base."""
    pontos = dict(zip(fs.ENTRADAS, pontos_t))
    valores = fs.valores_entrada(carregar())
    return valores, fs.inferir(valores, pontos=pontos)


df = carregar()
ids = df["CUST_ID"].tolist()
colunas_num = [c for c in df.columns if c != "CUST_ID"]

# cliente de exemplo inicial: um caso em que várias regras disparam e o score fica no meio
_, R0 = calcular(tuple(fs.PONTOS_PADRAO.values()))
_cand = np.flatnonzero((R0["forca"] > 0.2).sum(axis=1) >= 3)
i_padrao = int(_cand[np.argmin(np.abs(R0["score"][_cand] - 60))]) if len(_cand) else 0


# --------------------------------------------------------------------------
# Gráfico das curvas de uma entrada (baixa / média / alta)
# --------------------------------------------------------------------------
def grafico_termos(pontos3, chave: str, valor: float, graus3: np.ndarray, altura: int = 210,
                   rotulo_x: str | None = None) -> alt.Chart:
    """Curvas baixa/média/alta de uma entrada, com a faixa dominante do cliente em destaque."""
    q1, q2, q3 = pontos3
    vert = fcm.vertices_termos({"q1": q1, "q2": q2, "q3": q3, "xmin": 0.0, "xmax": 1.0})
    x_enc = alt.X("x:Q", title=rotulo_x or fs.EIXOS[chave], scale=alt.Scale(domain=[0, 1], nice=False),
                  axis=alt.Axis(values=[0, q1, q2, q3, 1.0], format="%", labelFontSize=10))
    y_enc = alt.Y("grau:Q", title="pertinência (0 a 1)", scale=alt.Scale(domain=[0, 1.05]),
                  axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
    grau_t = dict(zip(fs.TERMOS, graus3))
    dominante = max(grau_t, key=grau_t.get)
    fundo = alt.Chart(vert[vert["termo"] != dominante]).mark_line(
        strokeWidth=1.5, color=CINZA, opacity=0.6).encode(x=x_enc, y=y_enc, detail="termo:N")
    frente = alt.Chart(vert[vert["termo"] == dominante]).mark_line(
        strokeWidth=5, color=fs.COR_TERMO[dominante]).encode(x=x_enc, y=y_enc)
    regua = alt.Chart(pd.DataFrame({"x": [valor]})).mark_rule(
        strokeDash=[5, 4], color="#c9ccd1", strokeWidth=1.5).encode(x=x_enc)
    pts = pd.DataFrame({"x": [valor] * 3, "grau": list(grau_t.values()), "termo": list(grau_t)})
    pts["cor"] = [fs.COR_TERMO[tm] if tm == dominante else CINZA for tm in pts["termo"]]
    marcas = alt.Chart(pts).mark_point(filled=True, size=90, opacity=1).encode(
        x=x_enc, y=y_enc, color=alt.Color("cor:N", scale=None),
        tooltip=["termo", alt.Tooltip("grau:Q", format=".2f")])
    return (fundo + frente + regua + marcas).properties(height=altura)


# --------------------------------------------------------------------------
# Barra lateral: cliente e parâmetros
# --------------------------------------------------------------------------
def resetar_pontos() -> None:
    for chave, padrao in fs.PONTOS_PADRAO.items():
        for n in range(3):
            st.session_state[f"p_{chave}_{n}"] = int(round(padrao[n] * 100))


with st.sidebar:
    st.header("Cliente de exemplo")
    cliente = st.selectbox(
        "Cliente", ids, index=i_padrao, key="cliente",
        help="As abas 4, 5 e 6 mostram o cálculo do score para este cliente. Troque e tudo se atualiza.",
    )
    st.header("Parâmetros das curvas")
    st.caption(
        "Cada entrada tem três pontos de corte que desenham as curvas baixa, média e alta. Mexa nos controles e "
        "veja as curvas, o score e os resultados mudarem."
    )
    pontos: dict[str, tuple] = {}
    rotulos_corte = ("Fim da faixa BAIXA (100% até aqui)", "Pico da faixa MÉDIA", "Início da faixa ALTA (100% a partir daqui)")
    for chave, e in fs.ENTRADAS.items():
        with st.expander(e["rotulo"]):
            lidos = []
            for n, nome in enumerate(rotulos_corte):
                kk = f"p_{chave}_{n}"
                if kk not in st.session_state:
                    st.session_state[kk] = int(round(fs.PONTOS_PADRAO[chave][n] * 100))
                lidos.append(st.slider(nome, 0, 100, key=kk, format="%d%%") / 100)
            if lidos[0] < lidos[1] < lidos[2]:
                pontos[chave] = tuple(lidos)
            else:
                st.warning("Os três pontos precisam estar em ordem crescente. Usando os valores originais.")
                pontos[chave] = fs.PONTOS_PADRAO[chave]
    st.button("Voltar aos valores originais", on_click=resetar_pontos, icon=":material/restart_alt:")

pontos_t = tuple(pontos[k] for k in fs.ENTRADAS)
V, R = calcular(pontos_t)
k_cli = ids.index(cliente)
det = fs.inferir(V.iloc[[k_cli]], com_curva=True, pontos=pontos)
score_k = float(det["score"][0])
cls_k = int(det["classe"][0])
forca_k = det["forca"][0]
limite_k = float(df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median()).iloc[k_cli])
score_todos, classe_todos = R["score"], R["classe"]

# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("Lógica fuzzy aplicada a clientes de cartão de crédito")
st.caption(
    "Apresentação passo a passo: cada aba é um tópico. Base: Credit Card Dataset for Clustering (Kaggle). "
    "Problema: identificar quais clientes usam o cartão como empréstimo (dependência do crédito)."
)

(t0, t1, t2, t3, t4, t5, t6, t7) = st.tabs([
    ":material/help: 1 · O problema",
    ":material/database: 2 · Os dados",
    ":material/lightbulb: 3 · Lógica fuzzy",
    ":material/tune: 4 · Fuzzificação",
    ":material/rule: 5 · Regras",
    ":material/speed: 6 · Score",
    ":material/analytics: 7 · Resultados",
    ":material/flag: 8 · Conclusões",
])

# ==========================================================================
# 1. O problema
# ==========================================================================
with t0:
    with st.container(border=True):
        st.subheader("Qual problema estamos resolvendo?")
        st.markdown(
            "Um cartão de crédito pode ser usado de duas formas bem diferentes:\n\n"
            "- como **meio de pagamento**: a pessoa compra e paga a fatura inteira no vencimento;\n"
            "- como **empréstimo**: a pessoa deixa o saldo crescer, paga só parte da fatura, às vezes saca dinheiro no crédito "
            "e vive com o limite quase todo usado.\n\n"
            "Chamamos essa segunda situação de **dependência do crédito do cartão**. A pergunta deste trabalho é:"
        )
        st.info(
            "**Dado o comportamento de uso do cartão de um cliente, o quanto ele depende do crédito do cartão? "
            "Uma nota de 0 (não depende) a 100 (depende muito).**",
            icon=":material/target:",
        )
        st.markdown(
            "**Para que serve essa nota?** Um banco poderia usá-la para acompanhar os clientes que mais dependem do crédito "
            "(maior chance de problemas para pagar), oferecer renegociação ou um empréstimo mais barato, ou rever limites. "
            "Atenção: isso é uma **hipótese de uso**. Esta base não informa quem de fato deixou de pagar, então o score "
            "**não prevê inadimplência**. Ele organiza o comportamento observado segundo um critério claro."
        )
    with st.container(border=True):
        st.subheader("Por que usar lógica fuzzy aqui?")
        st.markdown(
            "\"Depender do crédito\" não tem uma fronteira nítida. Ninguém passa de \"não depende\" para \"depende\" de um dia para o outro, "
            "e os critérios que um analista usaria são **vagos, em linguagem natural**: *\"usa muito do limite\"*, *\"quase nunca paga a "
            "fatura inteira\"*, *\"saca dinheiro com frequência\"*. A lógica fuzzy foi criada justamente para transformar frases assim "
            "em contas, sem precisar de uma linha de corte rígida."
        )
    with st.container(border=True):
        st.subheader("O que significa baixa, média e alta dependência?")
        st.dataframe(pd.DataFrame([
            {"Classe": nome, "Score aproximado": faixa, "Como é o cliente": texto_c.replace("**", "")}
            for (nome, texto_c), faixa in zip(DEFINICAO_CLASSES.items(), ["até cerca de 35", "de 35 a 65", "acima de 65"])
        ]), hide_index=True)
        st.caption("As faixas do score são aproximadas porque as classes também são fuzzy: perto de 35 ou de 65 o cliente é um pouco de duas classes.")
    with st.container(border=True):
        st.subheader("O modelo em uma visão")
        c1, c2, c3 = st.columns(3)
        with c1, st.container(border=True):
            st.markdown("**4 ENTRADAS** (o que olhamos no cliente)")
            st.markdown("- Utilização do limite\n- Taxa de pagamento integral\n- Frequência de saque\n- Frequência de compras")
            st.caption("Aba 4")
        with c2, st.container(border=True):
            st.markdown("**13 REGRAS** (como interpretamos)")
            st.markdown("Frases do tipo *SE saca com frequência E quase nunca paga a fatura inteira, ENTÃO a dependência é ALTA*.")
            st.caption("Aba 5")
        with c3, st.container(border=True):
            st.markdown("**1 SAÍDA** (o resultado)")
            st.markdown("O **score de dependência do crédito**, de 0 a 100, e a classe: baixa, média ou alta.")
            st.caption("Aba 6")

# ==========================================================================
# 2. Os dados
# ==========================================================================
with t1:
    with st.container(border=True):
        st.subheader("De onde vêm os dados")
        st.markdown(
            "A base é a [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), "
            "do Kaggle. Cada **linha é um cliente** de cartão de crédito e cada **coluna é uma característica do "
            "comportamento de uso** do cartão: quanto comprou, quanto sacou em dinheiro, qual o limite, como paga a fatura etc."
        )
        st.markdown(
            "A base **não diz se o cliente é bom ou mau pagador**. Não existe uma resposta certa para conferir. "
            "Por isso a lógica fuzzy entra como forma de **construir um critério**: nós definimos como interpretar "
            "o comportamento, e o programa aplica esse critério a todos os clientes."
        )
        faltantes = int(df[colunas_num].isna().sum().sum())
        with st.container(horizontal=True):
            st.metric("Clientes (linhas)", f"{len(df):,}".replace(",", "."), border=True)
            st.metric("Características (colunas)", len(colunas_num), border=True,
                      help="17 colunas numéricas, além do código do cliente (CUST_ID).")
            st.metric("Valores faltantes", faltantes, border=True,
                      help="Células vazias: 313 em Pagamento mínimo e 1 em Limite de crédito.")
            st.metric("Clientes repetidos", int(df["CUST_ID"].duplicated().sum()), border=True)

    with st.container(border=True):
        st.subheader("As colunas")
        st.markdown(
            "Cada linha da tabela é uma coluna da base. **Valores em dinheiro** são acumulados no período coberto pela base "
            "(não são mensais) e a moeda não é informada. **Índices de 0 a 100%** (as frequências) são calculados pela própria "
            "base: 0% = nunca, 100% = sempre. Eles **não são contagens**."
        )
        linhas = []
        for col in colunas_num:
            s = df[col]
            linhas.append({
                "Coluna": fcm.rotulo(col),
                "Nome original": col,
                "O que mede": fcm.DESCRICOES.get(col, ""),
                "Unidade": fcm.UNIDADES.get(col, ""),
                "Mínimo": fcm.formatar_valor(col, s.min()),
                "Mediana": fcm.formatar_valor(col, s.median()),
                "Média": fcm.formatar_valor(col, s.mean()),
                "Máximo": fcm.formatar_valor(col, s.max()),
                "% de zeros": pct((s == 0).mean()),
                "Faltantes": int(s.isna().sum()),
                "Usada no score": USADAS_NO_SCORE.get(col, "-"),
            })
        st.dataframe(pd.DataFrame(linhas), hide_index=True, height=430)
        st.caption(
            "A última coluna mostra quais características alimentam o score fuzzy. A escolha delas é explicada na aba 4."
        )

    with st.container(border=True):
        st.subheader("Como são os valores?")
        st.markdown(
            "Escolha uma coluna para ver como os clientes se distribuem. Nas colunas em dinheiro, a maioria dos clientes tem "
            "valores baixos e poucos têm valores enormes. Por isso a média fica bem acima da mediana."
        )
        esq, dir_ = st.columns([2, 1])
        with esq:
            var = st.selectbox("Coluna", colunas_num, format_func=fcm.rotulo_longo, key="var_dist")
            usar_log = st.toggle("Ver em escala logarítmica, log(1 + valor)", value=False, key="log_dist",
                                 help="Comprime os valores grandes para enxergar melhor o corpo da distribuição.")
        serie = df[var].dropna()
        if "FREQUENCY" in var:
            serie = serie.clip(upper=1.0)
        grafico_v = pd.DataFrame({"valor": np.log1p(serie) if usar_log else serie})
        eixo = f"log(1 + {fcm.rotulo(var).lower()})" if usar_log else fcm.rotulo(var)
        st.altair_chart(
            alt.Chart(grafico_v).mark_bar(color="#4c78a8").encode(
                x=alt.X("valor:Q", bin=alt.Bin(maxbins=40), title=eixo),
                y=alt.Y("count():Q", title="clientes"),
            ).properties(height=260),
            width="stretch",
        )
        with dir_:
            st.metric("Média", fcm.formatar_valor(var, serie.mean()), border=True)
            st.metric("Mediana (cliente do meio)", fcm.formatar_valor(var, serie.median()), border=True)
            st.metric("Clientes com valor zero", pct((serie == 0).mean()), border=True)

    with st.container(border=True):
        st.subheader("Correlação: quais colunas andam juntas?")
        st.markdown(
            "A **correlação** mede o quanto duas colunas sobem e descem juntas. Vai de **−1 a +1**: "
            "perto de **+1**, quando uma sobe a outra também sobe; perto de **−1**, quando uma sobe a outra cai; "
            "perto de **0**, uma não diz nada sobre a outra. (Correlação não prova que uma causa a outra.)"
        )
        corr = df[colunas_num].corr()
        nomes = [fcm.rotulo(c) for c in colunas_num]
        longo = corr.set_axis(nomes, axis=0).set_axis(nomes, axis=1).reset_index(names="A").melt(
            id_vars="A", var_name="B", value_name="r")
        base_c = alt.Chart(longo).encode(
            x=alt.X("B:N", sort=nomes, title=None, axis=alt.Axis(labelAngle=-60, labelFontSize=10, labelLimit=260)),
            y=alt.Y("A:N", sort=nomes, title=None, axis=alt.Axis(labelFontSize=10, labelLimit=260)),
        )
        mapa = base_c.mark_rect().encode(
            color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="correlação"),
            tooltip=["A", "B", alt.Tooltip("r:Q", format="+.2f")])
        texto = base_c.mark_text(fontSize=8).encode(
            text=alt.Text("r:Q", format="+.1f"),
            color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value("black")))
        st.altair_chart((mapa + texto).properties(height=560), width="stretch")
        st.caption("Azul = andam juntas; vermelho = uma sobe quando a outra cai. A diagonal é sempre +1 (cada coluna com ela mesma).")

        pares = []
        for i in range(len(colunas_num)):
            for j in range(i + 1, len(colunas_num)):
                pares.append((colunas_num[i], colunas_num[j], float(corr.iloc[i, j])))
        pares.sort(key=lambda p: -abs(p[2]))
        st.markdown("##### Os pares mais fortes")
        st.dataframe(pd.DataFrame([{
            "Coluna A": fcm.rotulo(a), "Coluna B": fcm.rotulo(b), "Correlação": f"{r:+.2f}".replace(".", ","),
            "Leitura": ("andam juntas: quem tem mais de uma tende a ter mais da outra" if r > 0
                        else "uma sobe quando a outra cai"),
        } for a, b, r in pares[:8]]), hide_index=True)
        st.markdown(
            "**O que isso nos diz:** várias colunas medem o mesmo comportamento (por exemplo, compras, compras à vista "
            "e número de compras andam juntas). Usar todas repetiria a mesma informação. Por isso o score usa só 4 entradas, "
            "uma para cada comportamento relevante. Como elas foram escolhidas está na aba 4."
        )

    with st.container(border=True):
        st.subheader("Limpeza feita")
        freq_cols = [c for c in colunas_num if "FREQUENCY" in c]
        acima = int(sum((df[c] > 1).sum() for c in freq_cols))
        st.markdown(
            "- **Faltantes:** 313 clientes sem *pagamento mínimo* e 1 sem *limite de crédito*. O limite faltante é preenchido com a mediana. "
            "O pagamento mínimo não entra no score.\n"
            f"- **Índices acima de 100%:** {acima} valores de frequência passam de 1 (100%), o que não faz sentido para um índice. "
            "Eles são limitados a 100%.\n"
            "- **Utilização do limite:** calculada como saldo ÷ limite e limitada a 100%."
        )

# ==========================================================================
# 3. Lógica fuzzy
# ==========================================================================
with t2:
    with st.container(border=True):
        st.subheader("O problema da lógica \"sim ou não\"")
        st.markdown(
            "Uma regra tradicional exige uma **linha de corte**. Exemplo: \"usar mais de 60% do limite é uso ALTO\". "
            "Com 59% de uso o cliente não é alto; com 61% ele é. Dois clientes quase iguais ficam em classes opostas, "
            "e a mudança é brusca. Na vida real a transição é gradual."
        )
        st.markdown(
            "A **lógica fuzzy** (nebulosa) resolve isso trocando \"sim ou não\" por **graus de pertinência entre 0 e 1**. "
            "Um valor pode ser, ao mesmo tempo, *um pouco médio e bastante alto*. É o mesmo raciocínio da curva clássica de "
            "estatura (baixa, média, alta), só que aplicada ao uso do cartão."
        )

    with st.container(border=True):
        st.subheader("Experimente: utilização do limite")
        uso_demo = st.slider("Percentual do limite usado pelo cliente", 0, 100, 45, key="uso_demo", format="%d%%") / 100
        g_demo = fs.graus([uso_demo], pontos["uso"])[0]
        crisp = "baixa" if uso_demo < 0.25 else "média" if uso_demo < 0.60 else "alta"
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Regra tradicional** (faixas fixas: abaixo de 25% = baixa, de 25% a 60% = média, acima de 60% = alta)")
            st.metric("Classificação", crisp.upper(), border=True)
            st.markdown("**Lógica fuzzy** (graus de pertinência)")
            with st.container(horizontal=True):
                for nome_t, g in zip(fs.TERMOS, g_demo):
                    st.metric(nome_t.capitalize(), pct(g), border=True)
        with c2:
            st.altair_chart(grafico_termos(pontos["uso"], "uso", uso_demo, g_demo, 260), width="stretch")
        st.caption(
            "Mova o controle devagar perto de 60%: a regra tradicional troca de média para alta de uma só vez. "
            "No fuzzy, a pertinência a \"média\" cai e a \"alta\" sobe aos poucos."
        )

    with st.container(border=True):
        st.subheader("Como o score é construído: 4 passos")
        cols4 = st.columns(4)
        passos = [
            ("1 · Fuzzificação", "Cada informação do cliente vira graus de **baixa, média e alta**.", "Aba 4"),
            ("2 · Regras", "Regras **SE-ENTÃO** combinam as informações e dizem qual a dependência resultante.", "Aba 5"),
            ("3 · Agregação", "Cada regra ativa contribui com uma **área**; juntamos todas.", "Aba 6"),
            ("4 · Defuzzificação", "O **centro de gravidade** da área é um número de 0 a 100: o **score**.", "Aba 6"),
        ]
        for caixa, (titulo, texto_p, aba) in zip(cols4, passos):
            with caixa, st.container(border=True):
                st.markdown(f"**{titulo}**")
                st.markdown(texto_p)
                st.caption(aba)
        st.markdown(
            "O **score** vai de 0 (usa o cartão como meio de pagamento) a 100 (usa o cartão como empréstimo), como definido na aba 1. "
            "Depois, o próprio score é fuzzificado em **baixa, média e alta dependência**, e é assim que o cliente é classificado (aba 7)."
        )

# ==========================================================================
# 4. Fuzzificação
# ==========================================================================
with t3:
    with st.container(border=True):
        st.subheader("Como escolhemos as 4 entradas?")
        st.markdown(
            "**Não foi sorteio, e também não foram \"as 4 colunas mais correlacionadas\".** O critério foi:\n\n"
            "1. **Escolher os comportamentos que indicam usar o cartão como empréstimo**: quanto do limite está em uso, "
            "se a fatura é quitada, se saca dinheiro no crédito e se usa o cartão para comprar.\n"
            "2. **Usar uma coluna por comportamento.** A correlação (aba 2) mostra que várias colunas medem o mesmo "
            "comportamento. Usar todas só repetiria a informação e daria peso demais a ele.\n"
            "3. **Preferir colunas fáceis de interpretar e comparáveis entre clientes**: índices de 0 a 100% (frequência) "
            "em vez de valores em dinheiro, que dependem do tamanho do limite de cada um."
        )
        familias = [
            ("Quanto do limite está em uso", ["BALANCE", "CREDIT_LIMIT"], "Utilização do limite = saldo ÷ limite",
             "Saldo sozinho engana (R$ 2.000 é muito para limite de 2.500 e pouco para 20.000). A razão é comparável."),
            ("Como paga a fatura", ["PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT"], "Taxa de pagamento integral",
             "É a única que diz diretamente se o cliente quita a fatura. Os valores pagos dependem do tamanho da compra."),
            ("Saque de dinheiro no crédito", ["CASH_ADVANCE", "CASH_ADVANCE_TRX", "CASH_ADVANCE_FREQUENCY"], "Frequência de saque",
             "Já em escala 0 a 100% e muito ligada às outras duas colunas de saque."),
            ("Uso do cartão para comprar", ["PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "PURCHASES_TRX",
                                            "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
                                            "PURCHASES_INSTALLMENTS_FREQUENCY"], "Frequência de compras",
             "Resume as 7 colunas de compras, que andam juntas, em um índice de 0 a 100%."),
        ]
        corr_abs = df[colunas_num].corr().abs()
        linhas_f = []
        for comport, cols, escolhida, motivo in familias:
            sub = corr_abs.loc[cols, cols].to_numpy()
            media_c = float((sub.sum() - len(cols)) / (len(cols) * (len(cols) - 1)))
            linhas_f.append({
                "Comportamento": comport,
                "Colunas da base relacionadas": ", ".join(fcm.rotulo(c) for c in cols),
                "Correlação média entre elas": f"{media_c:.2f}".replace(".", ","),
                "Entrada escolhida": escolhida,
                "Por que esta": motivo,
            })
        st.dataframe(pd.DataFrame(linhas_f), hide_index=True)
        st.caption(
            "Correlação média: média dos valores absolutos das correlações entre as colunas do grupo (0 = nada a ver, 1 = idênticas). "
            "Nos grupos de compras e de saque ela é alta, então uma coluna representa bem as outras. No grupo de pagamento é "
            "baixa: ali a escolha vem do critério de negócio (quitar ou não a fatura), não da correlação."
        )
        st.markdown(
            "**Colunas que ficaram de fora:** *Freq. de atualização do saldo* e *Tempo de cartão* quase não se relacionam com usar o "
            "cartão como empréstimo. Os demais valores em dinheiro e contagens foram resumidos pelas entradas escolhidas."
        )

    with st.container(border=True):
        st.subheader("Passo 1 · Fuzzificação: transformar cada entrada em graus")
        st.markdown(
            "Para cada entrada definimos **três curvas** (baixa, média, alta) com **três pontos de corte**:\n\n"
            "- **baixa** vale 100% até o primeiro ponto e cai a 0% no segundo;\n"
            "- **média** tem o pico (100%) no segundo ponto e vale 0% no primeiro e no terceiro;\n"
            "- **alta** sobe de 0% no segundo ponto até 100% no terceiro e se mantém.\n\n"
            "Estes pontos são os **parâmetros** da fuzzificação e podem ser alterados na barra lateral."
        )
        st.dataframe(pd.DataFrame([{
            "Entrada": e["rotulo"],
            "Como é calculada": e["formula"],
            "Baixa até": pct(pontos[k][0]),
            "Pico da média": pct(pontos[k][1]),
            "Alta a partir de": pct(pontos[k][2]),
            "BAIXA quando o cliente...": e["baixa"],
            "ALTA quando o cliente...": e["alta"],
        } for k, e in fs.ENTRADAS.items()]), hide_index=True)

        with st.expander("Por que estes pontos de corte?"):
            st.markdown(
                "Os valores padrão foram escolhidos olhando como os clientes se distribuem (percentis) e o bom senso de "
                "um analista. Frequências e pagamento têm muitos zeros, então \"baixa\" cobre uma faixa pequena e "
                "\"alta\" começa antes do que o intuitivo."
            )
            linhas = []
            for k, e in fs.ENTRADAS.items():
                s = V[k]
                linhas.append({
                    "Entrada": e["rotulo"], "% de clientes com zero": pct((s == 0).mean()),
                    "Percentil 25": pct(s.quantile(.25)), "Mediana": pct(s.median()),
                    "Percentil 75": pct(s.quantile(.75)), "Percentil 90": pct(s.quantile(.90)),
                    "Cortes escolhidos": " / ".join(pct(x) for x in pontos[k]),
                })
            st.dataframe(pd.DataFrame(linhas), hide_index=True)

    with st.container(border=True):
        st.subheader(f"Exemplo: cliente {cliente}")
        st.markdown(
            f"Saldo de **{fcm.formatar_valor('BALANCE', float(df['BALANCE'].iloc[k_cli]))}** para um limite de "
            f"**{fcm.formatar_valor('CREDIT_LIMIT', limite_k)}** (utilização de {pct(V['uso'].iloc[k_cli])}), "
            f"frequência de compras de **{pct(V['compras'].iloc[k_cli])}**, frequência de saque de **{pct(V['saque'].iloc[k_cli])}** "
            f"e **{pct(V['pagamento'].iloc[k_cli])}** das faturas pagas por inteiro."
        )
        st.markdown(
            "Em cada gráfico, a **linha tracejada** marca o valor do cliente e os pontos mostram o grau em cada curva. "
            "**A faixa em que ele mais se encaixa fica colorida e grossa; as outras ficam cinza.** Acima de cada gráfico, "
            "a frase resume o resultado."
        )
        cols_g = st.columns(2)
        linhas_g = []
        for n, (chave, e) in enumerate(fs.ENTRADAS.items()):
            g3 = det["mu"][chave][0]
            dom = fs.TERMOS[int(np.argmax(g3))]
            with cols_g[n % 2], st.container(border=True):
                st.markdown(
                    f"**{e['rotulo']}** · o cliente está em **{pct(V[chave].iloc[k_cli])}** "
                    f"→ faixa :{COR_MD[dom]}[**{dom.upper()}** ({pct(g3.max())})]"
                )
                st.altair_chart(grafico_termos(pontos[chave], chave, float(V[chave].iloc[k_cli]), g3), width="stretch")
            linhas_g.append({"Entrada": e["rotulo"], "Valor do cliente": pct(V[chave].iloc[k_cli]),
                             "Grau em BAIXA": pct(g3[0]), "Grau em MÉDIA": pct(g3[1]), "Grau em ALTA": pct(g3[2])})
        st.markdown("##### Resultado da fuzzificação deste cliente")
        st.dataframe(pd.DataFrame(linhas_g), hide_index=True)
        st.caption(
            "Estes 12 números (4 entradas × 3 faixas) são o que as regras vão usar no próximo passo (aba 5). "
            "Muitas vezes o cliente fica entre duas faixas, com um grau em cada."
        )

# ==========================================================================
# 5. Regras
# ==========================================================================
with t4:
    with st.container(border=True):
        st.subheader(f"De onde partimos: cliente {cliente}")
        st.markdown("Estes são os graus calculados na aba 4 (fuzzificação). As regras abaixo usam exatamente estes números.")
        st.dataframe(pd.DataFrame([{
            "Entrada": e["rotulo"], "Valor do cliente": pct(V[k].iloc[k_cli]),
            "Grau em BAIXA": pct(det["mu"][k][0][0]), "Grau em MÉDIA": pct(det["mu"][k][0][1]),
            "Grau em ALTA": pct(det["mu"][k][0][2]),
        } for k, e in fs.ENTRADAS.items()]), hide_index=True)

    with st.container(border=True):
        st.subheader("Passo 2 · As regras SE-ENTÃO")
        st.markdown(
            "As regras são o **conhecimento do especialista** escrito em frases. Cada regra tem de **uma a três condições** "
            "(SE, E, E) e uma **conclusão** sobre a dependência do crédito (ENTÃO). Cada condição é do tipo *entrada é FAIXA* "
            "(por exemplo, *Frequência de saque é ALTA*). A coluna **Raciocínio** explica a ideia de cada regra."
        )
        st.markdown(
            "**O que é a \"força\" de uma regra?** É a resposta para: *\"o quanto esta regra vale para este cliente?\"* "
            "Entre parênteses em cada condição está o **grau** que o cliente tem naquela faixa (veja a tabela acima). "
            "A **força** é o **menor** desses graus. É como uma corrente: ela é tão forte quanto o elo mais fraco. "
            "Se uma condição não se aplica ao cliente (0%), a regra inteira vale 0% e não influencia o resultado."
        )
        col_forca = f"Força para o cliente {cliente}"
        cols_cond = ["SE (condição 1)", "E (condição 2)", "E (condição 3)"]
        filas, graus_cond = [], []
        for i, (cond, cons) in enumerate(fs.REGRAS):
            gs = [float(det["mu"][e][0][fs.TERMOS.index(t)]) for e, t in cond]
            cel = [f"{fs.texto_cond(e, t)} ({pct(g)})" for (e, t), g in zip(cond, gs)]
            cel += ["-"] * (3 - len(cel))
            graus_cond.append(gs + [None] * (3 - len(gs)))
            filas.append({
                "Nº": i + 1, cols_cond[0]: cel[0], cols_cond[1]: cel[1], cols_cond[2]: cel[2],
                "ENTÃO a dependência é": cons.upper(), "Raciocínio": fs.REGRAS_PORQUE[i],
                col_forca: forca_k[i],
            })
        tabela_r = pd.DataFrame(filas)
        cores = pd.DataFrame("", index=tabela_r.index, columns=tabela_r.columns)
        for i, gs in enumerate(graus_cond):
            for j, g in enumerate(gs):
                if g is not None and g > 0:  # condição que o cliente tem: azul, mais forte = maior grau
                    cores.loc[i, cols_cond[j]] = f"background-color: rgba(41, 151, 255, {0.18 + 0.62 * g:.2f})"
            if forca_k[i] > 0:  # regra ativa: a conclusão ganha a cor da classe
                hexa = fs.COR_CLASSE[fs.TERMOS.index(fs.REGRAS[i][1])].lstrip("#")
                rgb = ", ".join(str(int(hexa[n:n + 2], 16)) for n in (0, 2, 4))
                cores.loc[i, "ENTÃO a dependência é"] = f"background-color: rgba({rgb}, 0.45)"
        st.dataframe(
            tabela_r.style.apply(lambda _: cores, axis=None), hide_index=True,
            column_config={
                "Nº": st.column_config.NumberColumn(width="small"),
                "SE (condição 1)": st.column_config.TextColumn(width="large"),
                "E (condição 2)": st.column_config.TextColumn(width="large"),
                "E (condição 3)": st.column_config.TextColumn(width="large"),
                "Raciocínio": st.column_config.TextColumn(width="large"),
                f"Força para o cliente {cliente}": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="percent"),
            },
        )
        st.caption(
            "**Cores:** uma condição fica **azul** quando o cliente a tem (quanto mais escuro, maior o grau). Condição sem cor = o cliente "
            "não tem aquela faixa (grau 0%). Quando a regra tem força, a conclusão (ENTÃO) ganha a cor da classe "
            "(verde = baixa, amarelo = média, vermelho = alta). "
            f"{int((forca_k > 0).sum())} das {len(fs.REGRAS)} regras têm alguma força para este cliente."
        )

    with st.container(border=True):
        st.subheader("Quais regras mais influenciam o resultado deste cliente?")
        st.markdown(
            "Cada barra é a **força de uma regra** (do mesmo jeito que na tabela). Quanto **maior a barra**, mais aquela regra "
            "vale para este cliente e **mais ela puxa o score** para a conclusão dela. A **cor** mostra a conclusão: "
            "verde = baixa dependência, amarelo = média, vermelho = alta."
        )
        barras = pd.DataFrame({
            "Regra": [f"Regra {i + 1}" for i in range(len(fs.REGRAS))],
            "Força": forca_k,
            "Conclusão da regra": [f"dependência {cons}" for _, cons in fs.REGRAS],
        })
        st.altair_chart(
            alt.Chart(barras).mark_bar().encode(
                y=alt.Y("Regra:N", sort=list(barras["Regra"]), title=None),
                x=alt.X("Força:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="força da regra para este cliente"),
                color=alt.Color("Conclusão da regra:N", scale=alt.Scale(
                    domain=["dependência baixa", "dependência média", "dependência alta"], range=fs.COR_CLASSE)),
                tooltip=["Regra", "Conclusão da regra", alt.Tooltip("Força:Q", format=".0%")],
            ).properties(height=330),
            width="stretch",
        )

        st.markdown("##### Exemplo de cálculo das duas regras mais fortes")
        for r_i in np.argsort(-forca_k)[:2]:
            if forca_k[r_i] <= 0:
                continue
            cond, cons = fs.REGRAS[r_i]
            partes = [f"{fs.texto_cond(e, t)} (grau {pct(det['mu'][e][0][fs.TERMOS.index(t)])})" for e, t in cond]
            st.markdown(
                f"- **Regra {r_i + 1}** ({fs.REGRAS_PORQUE[r_i].lower()}) Condições: {'; '.join(partes)}. "
                f"Força = o menor grau = **{pct(forca_k[r_i])}**. Conclusão: dependência **{cons.upper()}**."
            )

    with st.container(border=True):
        st.subheader("Comparar com outros clientes")
        st.markdown(
            "Acima analisamos um cliente por vez (o escolhido na barra lateral). Aqui você coloca **vários clientes lado a lado**: "
            "cada coluna é um cliente e cada linha uma regra, com a **força** da regra para aquele cliente (azul mais forte = "
            "regra mais forte). Assim dá para ver quais regras valem para cada perfil."
        )
        i_max, i_min = int(np.argmax(score_todos)), int(np.argmin(score_todos))
        padrao_cmp = list(dict.fromkeys([cliente, ids[i_max], ids[i_min]]))
        escolhidos = st.multiselect(
            "Clientes para comparar", ids, default=padrao_cmp, max_selections=6, key="comparar",
            help="Começa com o cliente escolhido, o de maior score e o de menor score. Digite um código para buscar outro.",
        )
        if escolhidos:
            ks = [ids.index(x) for x in escolhidos]
            det_c = fs.inferir(V.iloc[ks], pontos=pontos)
            resumo_c = pd.DataFrame({
                "Cliente": escolhidos,
                "Score": [f"{x:.0f}" for x in det_c["score"]],
                "Classe": [fs.CLASSES[i] for i in det_c["classe"]],
                "Utilização do limite": [pct(V["uso"].iloc[k]) for k in ks],
                "Pagamento integral": [pct(V["pagamento"].iloc[k]) for k in ks],
                "Freq. de saque": [pct(V["saque"].iloc[k]) for k in ks],
                "Freq. de compras": [pct(V["compras"].iloc[k]) for k in ks],
            })
            st.dataframe(resumo_c, hide_index=True)
            matriz = pd.DataFrame(
                det_c["forca"].T, columns=escolhidos,
                index=[f"Regra {i + 1} → {cons.upper()}" for i, (_, cons) in enumerate(fs.REGRAS)],
            )
            matriz.insert(0, "Condições da regra", [fs.texto_condicao(i) for i in range(len(fs.REGRAS))])
            cores_m = pd.DataFrame("", index=matriz.index, columns=matriz.columns)
            for c_ in escolhidos:
                cores_m[c_] = [f"background-color: rgba(41, 151, 255, {0.55 * v:.2f})" if v > 0 else "" for v in matriz[c_]]
            st.dataframe(
                matriz.style.apply(lambda _: cores_m, axis=None).format({c_: "{:.0%}" for c_ in escolhidos}),
                column_config={"Condições da regra": st.column_config.TextColumn(width="large")},
            )
            st.caption("Cada número é a força da regra para o cliente da coluna. Célula sem cor = a regra não se aplica a ele (0%).")

# ==========================================================================
# 6. Score
# ==========================================================================
with t5:
    forca_cons = [
        max([float(forca_k[i]) for i, (_, cons) in enumerate(fs.REGRAS) if cons == fs.TERMOS[n]] or [0.0])
        for n in range(3)
    ]
    grade = det["grade"]
    with st.container(border=True):
        st.subheader("Passo 3 · Agregação: juntar as regras em três conclusões")
        st.markdown(
            f"As 13 regras de cada cliente terminam em apenas **três conclusões**: dependência baixa, média ou alta. "
            "Para cada conclusão vale a **regra mais forte** que aponta para ela (o \"OU\" fuzzy é o **máximo**). "
            f"Para o cliente **{cliente}**:"
        )
        with st.container(horizontal=True):
            for cl, v in zip(fs.CLASSES, forca_cons):
                st.metric(f"Força de \"{cl.lower()}\"", pct(v), border=True)
        st.markdown(
            "Cada conclusão é uma **região na escala de 0 a 100** (o contorno cinza). Quanto mais forte a conclusão, mais "
            "a região é preenchida (de baixo para cima). Uma conclusão com força 60% preenche a região só até 60% de altura: "
            "ela é \"cortada\" nessa altura."
        )
        cols3 = st.columns(3)
        for j, (caixa, cl) in enumerate(zip(cols3, fs.CLASSES)):
            with caixa, st.container(border=True):
                st.markdown(f"**{cl}** · força {pct(forca_cons[j])}")
                contorno = pd.DataFrame({"x": grade, "grau": det["saida"][j]})
                corte = pd.DataFrame({"x": grade, "grau": np.minimum(forca_cons[j], det["saida"][j])})
                xs = alt.X("x:Q", title="score (0 a 100)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=[0, 50, 100]))
                ys = alt.Y("grau:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
                st.altair_chart(
                    (alt.Chart(corte).mark_area(color=fs.COR_CLASSE[j], opacity=0.7).encode(x=xs, y=ys)
                     + alt.Chart(contorno).mark_line(color=CINZA, strokeWidth=1.5).encode(x=xs, y=ys)).properties(height=150),
                    width="stretch")
        st.markdown(
            "**Agregação** é juntar as três regiões preenchidas em uma só figura: a **área laranja** do gráfico abaixo."
        )
        st.markdown("##### Como ler este gráfico")
        st.markdown(
            "- **Eixo horizontal (de 0 a 100):** os **scores possíveis**. Cada ponto da linha é um score que o cliente poderia ter.\n"
            "- **Eixo vertical, a \"altura\" (de 0 a 1):** o **quanto as regras apoiam aquele score**. 0 = nenhuma regra defende esse score; "
            "1 = apoio total.\n"
            "- **Contornos cinzas e os nomes no alto do gráfico (baixa, média, alta dependência):** são as três regiões da escala. "
            "Os nomes no topo só dizem **qual contorno é qual**: a região \"baixa\" fica à esquerda (scores baixos), "
            "a \"alta\" à direita (scores altos).\n"
            "- **Área laranja:** o resultado das regras deste cliente. Onde ela é **alta**, as regras defendem aquele score; "
            "onde é **zero**, ninguém defende."
        )
        contornos = pd.DataFrame({
            "x": np.tile(grade, 3), "grau": det["saida"].ravel(), "faixa": np.repeat(fs.CLASSES, len(grade)),
        })
        area = pd.DataFrame({"x": grade, "grau": det["agregada"][0]})
        rotulos_x = pd.DataFrame({
            "x": [10, 50, 90], "grau": [1.1] * 3,
            "txt": ["região BAIXA dependência", "região MÉDIA dependência", "região ALTA dependência"],
        })
        x_pico = float(grade[int(np.argmax(det["agregada"][0]))])
        rotulo_area = pd.DataFrame({"x": [min(max(x_pico, 22), 78)], "grau": [float(det["agregada"][0].max()) / 2],
                                    "txt": ["área laranja = resultado das regras"]})
        x_s = alt.X("x:Q", title="score possível (0 a 100)", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=list(range(0, 101, 10))))
        y_s = alt.Y("grau:Q", title="altura = apoio das regras (0 a 1)", scale=alt.Scale(domain=[0, 1.2]),
                    axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
        g_cont = alt.Chart(contornos).mark_line(color=CINZA, strokeWidth=1.5).encode(x=x_s, y=y_s, detail="faixa:N")
        g_area = alt.Chart(area).mark_area(color="#f58518", opacity=0.65).encode(x=x_s, y=y_s)
        g_txt = alt.Chart(rotulos_x).mark_text(color=CINZA, fontSize=12).encode(x=x_s, y=y_s, text="txt:N")
        g_txt_area = alt.Chart(rotulo_area).mark_text(color="white", fontSize=12, fontWeight="bold").encode(
            x=x_s, y=y_s, text="txt:N")
        st.altair_chart((g_cont + g_area + g_txt + g_txt_area).properties(height=280), width="stretch")

    with st.container(border=True):
        st.subheader("Passo 4 · Defuzzificação: da área ao score")
        agg = det["agregada"][0]
        st.markdown(
            "A área laranja é uma **figura**, mas queremos **um número**: o score. Para isso usamos uma **média**, com um detalhe: "
            "**cada score possível conta na proporção da sua altura**. Um score que as regras defendem muito (altura grande) puxa a média "
            "com força; um score que ninguém defende (altura zero) não puxa nada."
        )
        st.markdown(
            "É o mesmo que perguntar: *se a área laranja fosse uma placa de madeira, em que ponto ela ficaria em equilíbrio, como numa "
            "gangorra?* Esse ponto é o **centro de gravidade** e é o score. A conta é:\n\n"
            "**score = soma de (posição × altura) ÷ soma das alturas**\n\n"
            "onde **posição** é o score possível (o número no eixo horizontal) e **altura** é o apoio das regras naquela posição "
            "(a altura da área laranja ali)."
        )
        st.markdown("##### A conta passo a passo (de 10 em 10 pontos)")
        posicoes = np.arange(0, 101, 10)
        alturas = np.interp(posicoes, grade, agg)
        tabela_conta = pd.DataFrame({
            "Posição (score possível)": posicoes,
            "Altura (apoio das regras)": [br(h, 2) for h in alturas],
            "Posição × altura": [br(p_ * h, 1) for p_, h in zip(posicoes, alturas)],
        })
        total = pd.DataFrame({
            "Posição (score possível)": ["TOTAL"],
            "Altura (apoio das regras)": [br(alturas.sum(), 2)],
            "Posição × altura": [br((posicoes * alturas).sum(), 1)],
        })
        st.dataframe(pd.concat([tabela_conta.astype(str), total.astype(str)]), hide_index=True)
        aprox = float((posicoes * alturas).sum() / alturas.sum()) if alturas.sum() > 0 else 50.0
        soma_ponderada = float((grade * agg).sum())
        soma_alturas = float(agg.sum())
        st.markdown(
            f"**Score aproximado** = {br((posicoes * alturas).sum(), 1)} ÷ {br(alturas.sum(), 2)} = **{br(aprox, 1)}**. "
            f"O programa faz a mesma conta com 201 posições (de 0,5 em 0,5), mais precisa: "
            f"{br(soma_ponderada, 1)} ÷ {br(soma_alturas, 1)} = **{br(score_k, 1)}**."
        )
        g_reg = alt.Chart(pd.DataFrame({"x": [score_k]})).mark_rule(color="#e45756", strokeWidth=3).encode(x=x_s)
        g_reg_txt = alt.Chart(pd.DataFrame({"x": [score_k], "grau": [1.17], "txt": [f"score = {br(score_k, 1)}"]})).mark_text(
            color="#e45756", fontSize=13, fontWeight="bold", dx=48).encode(x=x_s, y=y_s, text="txt:N")
        st.altair_chart((g_cont + g_area + g_reg + g_reg_txt).properties(height=280), width="stretch")
        st.caption(
            "A linha vermelha é o ponto de equilíbrio da área laranja: o score do cliente. Se as regras defendessem só scores altos, "
            "a área ficaria à direita e a linha também; se defendessem só scores baixos, iria para a esquerda."
        )

        pk = det["pert_saida"][0]
        top2 = [int(i) for i in np.argsort(-forca_k)[:2] if forca_k[i] > 0]
        motivos = "; ".join(f"regra {i + 1} ({fs.REGRAS_PORQUE[i].rstrip('.').lower()})" for i in top2)
        st.success(
            f"**Resultado para o cliente {cliente}: score {score_k:.0f}, {fs.CLASSES[cls_k].lower()}.** "
            f"As regras que mais pesaram foram: {motivos}. "
            f"Em graus: {pct(pk[0])} baixa, {pct(pk[1])} média e {pct(pk[2])} alta dependência. "
            "O próprio resultado é fuzzy: o cliente pode pertencer a duas classes ao mesmo tempo.",
            icon=":material/check_circle:",
        )
        with st.container(horizontal=True):
            st.metric(f"Score do cliente {cliente}", f"{score_k:.0f}", border=True)
            st.metric("Classe (maior pertinência)", fs.CLASSES[cls_k], border=True)
            for nome_cl, valor_cl in zip(fs.CLASSES, pk):
                st.metric(f"Pertinência: {nome_cl.lower()}", pct(valor_cl), border=True)
        st.caption(
            "O score é um ponto de equilíbrio, por isso raramente chega a 0 ou 100 "
            f"(nesta base vai de {score_todos.min():.0f} a {score_todos.max():.0f}). A classe é a faixa em que o score tem a maior pertinência."
        )

# ==========================================================================
# 7. Resultados
# ==========================================================================
with t6:
    entre = float((R["pert_saida"].max(axis=1) < 0.6).mean())
    with st.container(border=True):
        st.subheader("O que cada classe significa")
        for nome_cl, texto_c in DEFINICAO_CLASSES.items():
            st.markdown(f"- **{nome_cl}:** {texto_c}")
        st.caption("Lembrete: esta é uma classificação de comportamento de uso, definida pelas nossas regras. Não é uma previsão de inadimplência.")

    with st.container(border=True):
        st.subheader("Quantos clientes em cada classe?")
        cont = np.bincount(classe_todos, minlength=3)
        df_cls = pd.DataFrame({
            "Classe": fs.CLASSES, "Clientes": cont, "Percentual": cont / cont.sum(),
        })
        df_cls["Rótulo"] = [f"{pct(p, 1)} ({br(n)} clientes)" for p, n in zip(df_cls["Percentual"], df_cls["Clientes"])]
        esc_cls = alt.Scale(domain=fs.CLASSES, range=fs.COR_CLASSE)
        barras_c = alt.Chart(df_cls).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Classe:N", sort=fs.CLASSES, title=None),
            x=alt.X("Percentual:Q", axis=alt.Axis(format="%"), title="% dos clientes", scale=alt.Scale(domain=[0, max(0.6, float(df_cls["Percentual"].max()) + 0.15)])),
            color=alt.Color("Classe:N", scale=esc_cls, legend=None),
            tooltip=["Classe", "Clientes", alt.Tooltip("Percentual:Q", format=".1%")],
        )
        rotulos_c = barras_c.mark_text(align="left", dx=6, fontSize=13).encode(text="Rótulo:N", color=alt.value("#c9ccd1"))
        st.altair_chart((barras_c + rotulos_c).properties(height=170), width="stretch")
        st.caption(
            f"A classe com mais clientes é **{fs.CLASSES[int(np.argmax(cont))].lower()}** ({pct(cont.max() / cont.sum(), 1)}). "
            "Os percentuais mudam se você alterar os pontos de corte na barra lateral."
        )

    with st.container(border=True):
        st.subheader("O score aplicado a todos os clientes")
        st.markdown(
            "Rodando o mesmo processo (fuzzificação, regras, agregação, centro de gravidade) para os 8.950 clientes, "
            "cada um recebe um score e uma classe. **Os parâmetros da barra lateral valem aqui**: mude os cortes das curvas "
            "e veja a distribuição se mover."
        )
        with st.container(horizontal=True):
            st.metric("Clientes", f"{len(df):,}".replace(",", "."), border=True)
            st.metric("Score médio", f"{score_todos.mean():.0f}", border=True)
            st.metric("Clientes entre duas classes", pct(entre), border=True,
                      help="Clientes cuja maior pertinência às classes é menor que 60%: estão na transição.")
        hist = alt.Chart(pd.DataFrame({"score": score_todos})).mark_bar(color="#4c78a8").encode(
            x=alt.X("score:Q", bin=alt.Bin(step=2), title="score de dependência do crédito"),
            y=alt.Y("count():Q", title="clientes"))
        cortes = alt.Chart(pd.DataFrame({"x": [35, 65]})).mark_rule(color="#e45756", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((hist + cortes).properties(height=260), width="stretch")
        st.caption("Linhas tracejadas em 35 e 65: onde a classe muda (cruzamento entre as faixas baixa/média e média/alta).")

    base_cl = df.assign(_classe=np.array(fs.CLASSES)[classe_todos], _uso=V["uso"], _pag=V["pagamento"],
                        _saq=V["saque"], _comp=V["compras"])
    resumo = base_cl.groupby("_classe").agg(
        clientes=("CUST_ID", "size"), uso=("_uso", "mean"), pag=("_pag", "mean"), saq=("_saq", "mean"),
        comp=("_comp", "mean"), saldo=("BALANCE", "mean"), saque_valor=("CASH_ADVANCE", "mean"),
        compras_valor=("PURCHASES", "mean"),
    ).reindex(fs.CLASSES)
    with st.container(border=True):
        st.subheader("Perfil de cada classe")
        st.dataframe(pd.DataFrame({
            "Classe": fs.CLASSES,
            "Clientes": [f"{int(x):,}".replace(",", ".") for x in resumo["clientes"]],
            "% da base": [pct(x / len(df), 1) for x in resumo["clientes"]],
            "Utilização do limite (média)": [pct(x) for x in resumo["uso"]],
            "Faturas pagas por inteiro (média)": [pct(x) for x in resumo["pag"]],
            "Frequência de saque (média)": [pct(x) for x in resumo["saq"]],
            "Frequência de compras (média)": [pct(x) for x in resumo["comp"]],
            "Saldo médio (valor)": [fcm.formatar_valor("BALANCE", x) for x in resumo["saldo"]],
            "Saque médio (valor no período)": [fcm.formatar_valor("CASH_ADVANCE", x) for x in resumo["saque_valor"]],
            "Compras médias (valor no período)": [fcm.formatar_valor("PURCHASES", x) for x in resumo["compras_valor"]],
        }), hide_index=True)
        st.markdown(
            "**Leitura:** a classe de alta dependência usa a maior parte do limite, quase nunca paga a fatura inteira e saca "
            "dinheiro com frequência. A de baixa dependência faz o contrário."
        )

    with st.container(border=True):
        st.subheader("Clientes, um a um")
        filtro = st.multiselect("Classes", fs.CLASSES, default=fs.CLASSES, key="filtro_classes")
        tabela_c = pd.DataFrame({
            "Cliente": df["CUST_ID"], "Score": score_todos, "Classe": np.array(fs.CLASSES)[classe_todos],
            "Pertinência baixa": R["pert_saida"][:, 0], "Pertinência média": R["pert_saida"][:, 1],
            "Pertinência alta": R["pert_saida"][:, 2],
            "Utilização do limite": V["uso"], "Pagamento integral": V["pagamento"],
            "Freq. de saque": V["saque"], "Freq. de compras": V["compras"],
        })
        tabela_c = tabela_c[tabela_c["Classe"].isin(filtro)].sort_values("Score", ascending=False)
        st.caption(f"{len(tabela_c):,} clientes".replace(",", "."))
        st.dataframe(
            tabela_c, hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.0f"),
                **{c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="percent")
                   for c in ["Pertinência baixa", "Pertinência média", "Pertinência alta"]},
                **{c: st.column_config.NumberColumn(c, format="percent")
                   for c in ["Utilização do limite", "Pagamento integral", "Freq. de saque", "Freq. de compras"]},
            },
        )
        st.download_button("Baixar CSV", tabela_c.to_csv(index=False).encode("utf-8"),
                           file_name="score_fuzzy.csv", mime="text/csv", icon=":material/download:")

# ==========================================================================
# 8. Conclusões
# ==========================================================================
with t7:
    n_cls = resumo["clientes"].to_numpy()
    alta, baixa = fs.CLASSES[2], fs.CLASSES[0]
    outras = base_cl.groupby("_classe")[["PAYMENTS", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "MINIMUM_PAYMENTS"]].mean().reindex(fs.CLASSES)

    def razao(col: str) -> float:
        return float(outras.loc[alta, col] / max(outras.loc[baixa, col], 1e-9))

    with st.container(border=True):
        st.subheader("Resposta ao problema")
        st.markdown(
            f"A pergunta era: **quais clientes dependem do crédito do cartão, e o quanto?** Com a lógica fuzzy chegamos a:\n\n"
            f"- **{pct(n_cls[2] / len(df))} dos clientes ({br(n_cls[2])}) têm alta dependência**: em média usam "
            f"{pct(resumo.loc[alta, 'uso'])} do limite, pagam por inteiro só {pct(resumo.loc[alta, 'pag'])} das faturas "
            f"e têm {pct(resumo.loc[alta, 'saq'])} de índice de saque.\n"
            f"- **{pct(n_cls[0] / len(df))} ({br(n_cls[0])}) têm baixa dependência**: usam {pct(resumo.loc[baixa, 'uso'])} do limite, "
            f"pagam por inteiro {pct(resumo.loc[baixa, 'pag'])} das faturas e quase não sacam ({pct(resumo.loc[baixa, 'saq'])}).\n"
            f"- **{pct(n_cls[1] / len(df))} ({br(n_cls[1])}) ficam na classe média**, e {pct(entre)} de todos os clientes estão "
            "**na zona de transição entre duas classes**."
        )
        st.markdown(
            "**Na prática:** os clientes de alta dependência seriam os primeiros a acompanhar (renegociação, ajuste de limite, oferta de "
            "um crédito mais barato que o saque no cartão). Os de baixa dependência usam o cartão como meio de pagamento, "
            "e os de transição são os que mais merecem um olhar caso a caso."
        )
    with st.container(border=True):
        st.subheader("O score faz sentido? Um teste com colunas que não usamos")
        st.markdown(
            "Se o score captura algo real, ele também deve diferenciar as classes em **colunas que não entraram no cálculo**. "
            f"Comparando a classe de alta com a de baixa dependência: saca **{br(razao('CASH_ADVANCE'), 1)}×** mais dinheiro (valor), "
            f"compra **{br(1 / razao('PURCHASES'), 1)}×** menos em valor total, e tem limite médio de "
            f"{fcm.formatar_valor('CREDIT_LIMIT', outras.loc[alta, 'CREDIT_LIMIT'])} contra {fcm.formatar_valor('CREDIT_LIMIT', outras.loc[baixa, 'CREDIT_LIMIT'])}. "
            "Isso é coerente com o perfil de quem usa o cartão como empréstimo, mas é apenas um teste de consistência: "
            "a base não tem informação de inadimplência para confirmar o risco."
        )
        st.dataframe(pd.DataFrame({
            "Classe": fs.CLASSES,
            "Pagamentos (valor no período)": [fcm.formatar_valor("PAYMENTS", x) for x in outras["PAYMENTS"]],
            "Compras (valor no período)": [fcm.formatar_valor("PURCHASES", x) for x in outras["PURCHASES"]],
            "Saque (valor no período)": [fcm.formatar_valor("CASH_ADVANCE", x) for x in outras["CASH_ADVANCE"]],
            "Limite de crédito": [fcm.formatar_valor("CREDIT_LIMIT", x) for x in outras["CREDIT_LIMIT"]],
            "Pagamento mínimo (valor)": [fcm.formatar_valor("MINIMUM_PAYMENTS", x) for x in outras["MINIMUM_PAYMENTS"]],
        }), hide_index=True)
    with st.container(border=True):
        st.subheader("O que a lógica fuzzy trouxe")
        st.markdown(
            "- **Transição gradual:** dois clientes parecidos recebem scores parecidos, sem saltos bruscos. Por isso é possível dizer "
            f"que {pct(entre)} dos clientes estão entre duas classes, em vez de forçá-los em uma.\n"
            "- **Explicabilidade:** para qualquer cliente dá para mostrar por que ele recebeu aquele score (abas 4 a 6): quais graus, "
            "quais regras e como o número final foi calculado.\n"
            "- **Conhecimento de especialista:** o critério é escrito em linguagem natural, sem precisar de rótulos nos dados."
        )
    with st.container(border=True):
        st.subheader("Limites")
        st.markdown(
            "- **Critério subjetivo:** os pontos de corte e as regras são escolha nossa. Outras pessoas escolheriam diferente "
            "(a barra lateral permite testar isso).\n"
            "- **Sem validação externa:** a base não tem informação de inadimplência, então o score não foi comparado com a realidade.\n"
            "- **Período e moeda:** a documentação da base não deixa claro se os valores acumulados cobrem 6 ou 12 meses, e a moeda não é informada.\n"
            "- **Escala do score:** como é um centro de gravidade, o score fica entre cerca de 18 e 82, e não entre 0 e 100."
        )
    with st.container(border=True):
        st.subheader("E o agrupamento em grupos?")
        st.markdown(
            "Existe uma segunda técnica fuzzy, o **agrupamento Fuzzy C-Means**, em que um algoritmo procura grupos nos dados e dá a "
            "cada cliente um percentual por grupo. Ela é uma técnica reconhecida, mas **é diferente da lógica fuzzy de regras**: "
            "não há curvas nem regras escritas por nós, e o número de grupos tem de ser escolhido (aqui foi por índices "
            "estatísticos, e o resultado foi uma divisão fraca em 2 grupos). Por isso ficou como **apêndice**, "
            "e o foco desta apresentação é o sistema de regras."
        )
        try:
            st.page_link("apendice_fcm.py", label="Abrir o apêndice: agrupamento Fuzzy C-Means", icon=":material/bubble_chart:")
        except Exception:  # fora do menu de páginas (execução isolada)
            st.caption("Apêndice disponível no menu lateral de páginas.")
