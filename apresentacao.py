"""Apresentação passo a passo: lógica fuzzy aplicada a clientes de cartão de crédito.

Cada aba é um tópico: dados, conceito, fuzzificação, regras, score, resultados e conclusões.
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
USADAS_NO_SCORE = {
    "BALANCE": "Utilização do limite (saldo ÷ limite)",
    "CREDIT_LIMIT": "Utilização do limite (saldo ÷ limite)",
    "PRC_FULL_PAYMENT": "Taxa de pagamento integral",
    "CASH_ADVANCE_FREQUENCY": "Frequência de saque",
    "PURCHASES_FREQUENCY": "Frequência de compras",
}


def pct(x: float, casas: int = 0) -> str:
    return fcm.fmt_pct(x, casas)


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
def grafico_termos(pontos3, valor: float | None, graus3: np.ndarray | None, titulo: str = "",
                   altura: int = 210) -> alt.Chart:
    """Curvas baixa/média/alta. Com `valor`, marca o cliente e destaca a faixa dominante."""
    q1, q2, q3 = pontos3
    vert = fcm.vertices_termos({"q1": q1, "q2": q2, "q3": q3, "xmin": 0.0, "xmax": 1.0})
    x_enc = alt.X("x:Q", title="valor da entrada", scale=alt.Scale(domain=[0, 1], nice=False),
                  axis=alt.Axis(values=[0, q1, q2, q3, 1.0], format="%", labelFontSize=10))
    y_enc = alt.Y("grau:Q", title="grau de pertinência", scale=alt.Scale(domain=[0, 1.05]),
                  axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
    if valor is None:
        linhas = alt.Chart(vert).mark_line(strokeWidth=3).encode(
            x=x_enc, y=y_enc, color=alt.Color("termo:N", legend=alt.Legend(title="faixa"),
                                              scale=alt.Scale(domain=list(fs.COR_TERMO), range=list(fs.COR_TERMO.values()))))
        return linhas.properties(height=altura, title=titulo)
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
    tit = alt.TitleParams(titulo, fontSize=13, anchor="start", color=fs.COR_TERMO[dominante])
    return (fundo + frente + regua + marcas).properties(height=altura, title=tit)


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
        help="Os passos 3, 4 e 5 mostram o cálculo para este cliente. Troque e tudo se atualiza.",
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
limite_k = float(df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median()).iloc[k_cli])
score_todos, classe_todos = R["score"], R["classe"]

# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("Lógica fuzzy aplicada a clientes de cartão de crédito")
st.caption(
    "Apresentação passo a passo: cada aba é um tópico. Base: Credit Card Dataset for Clustering (Kaggle). "
    "Objetivo: usar a lógica fuzzy para dar a cada cliente um score de dependência do crédito."
)

(t1, t2, t3, t4, t5, t6, t7) = st.tabs([
    ":material/database: 1 · Os dados",
    ":material/lightbulb: 2 · Lógica fuzzy",
    ":material/tune: 3 · Fuzzificação",
    ":material/rule: 4 · Regras",
    ":material/speed: 5 · Score",
    ":material/analytics: 6 · Resultados",
    ":material/flag: 7 · Conclusões",
])

# ==========================================================================
# 1. Os dados
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
            "A última coluna mostra quais características alimentam o score fuzzy. As outras ficaram de fora para o "
            "critério ficar simples e explicável (veja a correlação abaixo)."
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
            "**Por que isso importa para o score:** várias colunas medem a mesma coisa (por exemplo, compras, compras à vista "
            "e número de compras andam juntas). Em vez de usar as 17, escolhemos **uma medida de cada comportamento**: "
            "quanto do limite está usado, se paga a fatura inteira, se saca dinheiro e se compra. "
            "Nos dados, saldo alto anda junto com saque (correlação +0,50) e contra o pagamento integral (−0,32), "
            "o que sustenta tratar esses sinais como indícios de dependência do crédito."
        )

    with st.container(border=True):
        st.subheader("Limpeza feita")
        freq_cols = [c for c in colunas_num if "FREQUENCY" in c]
        acima = int(sum((df[c] > 1).sum() for c in freq_cols))
        st.markdown(
            f"- **Faltantes:** 313 clientes sem *pagamento mínimo* e 1 sem *limite de crédito*. O limite faltante é preenchido com a mediana. "
            "O pagamento mínimo não entra no score.\n"
            f"- **Índices acima de 100%:** {acima} valores de frequência passam de 1 (100%), o que não faz sentido para um índice. "
            "Eles são limitados a 100%.\n"
            "- **Utilização do limite:** calculada como saldo ÷ limite e limitada a 100%."
        )

# ==========================================================================
# 2. Lógica fuzzy
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
            st.altair_chart(grafico_termos(pontos["uso"], uso_demo, g_demo, f"{pct(uso_demo)} do limite usado", 260),
                            width="stretch")
        st.caption(
            "Mova o controle devagar perto de 60%: a regra tradicional troca de média para alta de uma só vez. "
            "No fuzzy, a pertinência a \"média\" cai e a \"alta\" sobe aos poucos."
        )

    with st.container(border=True):
        st.subheader("Como o score é construído: 4 passos")
        cols4 = st.columns(4)
        passos = [
            ("1 · Fuzzificação", "Cada informação do cliente vira graus de **baixa, média e alta**.", "Aba 3"),
            ("2 · Regras", "Regras **SE-ENTÃO** combinam as informações e dizem qual a dependência resultante.", "Aba 4"),
            ("3 · Agregação", "Cada regra ativa contribui com uma **área**; juntamos todas.", "Aba 5"),
            ("4 · Defuzzificação", "O **centro de gravidade** da área é um número de 0 a 100: o **score**.", "Aba 5"),
        ]
        for caixa, (titulo, texto_p, aba) in zip(cols4, passos):
            with caixa, st.container(border=True):
                st.markdown(f"**{titulo}**")
                st.markdown(texto_p)
                st.caption(aba)
        st.markdown(
            "O **score** vai de 0 (usa o cartão como meio de pagamento: usa pouco do limite, paga a fatura inteira, não saca) "
            "a 100 (usa o cartão como empréstimo: limite quase todo usado, paga só parte da fatura, saca dinheiro). "
            "Depois, o próprio score é fuzzificado em **baixa, média e alta dependência**, e é assim que o cliente é classificado (aba 6)."
        )

# ==========================================================================
# 3. Fuzzificação
# ==========================================================================
with t3:
    with st.container(border=True):
        st.subheader("Passo 1 · Fuzzificação: as 4 entradas")
        st.markdown(
            "São quatro informações sobre o cliente. Para cada uma, definimos **três curvas** (baixa, média, alta) "
            "com **três pontos de corte**:\n\n"
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
        st.subheader(f"Cliente {cliente}")
        st.markdown(
            f"Saldo de **{fcm.formatar_valor('BALANCE', float(df['BALANCE'].iloc[k_cli]))}** para um limite de "
            f"**{fcm.formatar_valor('CREDIT_LIMIT', limite_k)}** (utilização de {pct(V['uso'].iloc[k_cli])}), "
            f"frequência de compras de **{pct(V['compras'].iloc[k_cli])}**, frequência de saque de **{pct(V['saque'].iloc[k_cli])}** "
            f"e **{pct(V['pagamento'].iloc[k_cli])}** das faturas pagas por inteiro."
        )
        st.markdown(
            "Em cada gráfico, a linha tracejada marca onde o cliente está. **A faixa em que ele mais se encaixa fica colorida e "
            "grossa; as outras ficam cinza.**"
        )
        cols_g = st.columns(2)
        linhas_g = []
        for n, (chave, e) in enumerate(fs.ENTRADAS.items()):
            g3 = det["mu"][chave][0]
            dom = fs.TERMOS[int(np.argmax(g3))]
            titulo = f"{e['rotulo']}: {pct(V[chave].iloc[k_cli])} → {dom} ({pct(g3.max())})"
            with cols_g[n % 2]:
                st.altair_chart(grafico_termos(pontos[chave], float(V[chave].iloc[k_cli]), g3, titulo), width="stretch")
            linhas_g.append({"Entrada": e["rotulo"], "Valor do cliente": pct(V[chave].iloc[k_cli]),
                             "Baixa": pct(g3[0]), "Média": pct(g3[1]), "Alta": pct(g3[2])})
        st.markdown("##### Resultado da fuzzificação deste cliente")
        st.dataframe(pd.DataFrame(linhas_g), hide_index=True)
        st.caption(
            "Estes 12 números (4 entradas × 3 faixas) são o que as regras vão usar no próximo passo. "
            "Note que as três faixas de uma mesma entrada nunca somam mais que 100%, e muitas vezes o cliente fica "
            "entre duas faixas."
        )

# ==========================================================================
# 4. Regras
# ==========================================================================
with t4:
    forca_k = det["forca"][0]
    with st.container(border=True):
        st.subheader("Passo 2 · Regras SE-ENTÃO")
        st.markdown(
            "As regras são o **conhecimento do especialista** escrito em forma de frases. Cada regra junta duas ou três "
            "condições e diz qual deve ser a dependência do crédito. Exemplo: *SE a frequência de saque é ALTA E a taxa de "
            "pagamento integral é BAIXA, ENTÃO a dependência é ALTA*.\n\n"
            "Como cada condição é uma **faixa com grau entre 0 e 1**, a regra também ganha um grau: a **força**. "
            "O \"E\" fuzzy é o **mínimo** dos graus, como numa corrente que é tão forte quanto o elo mais fraco. "
            "Força 0% = a regra não se aplica a este cliente."
        )
        tabela_regras = pd.DataFrame({
            "Nº": list(range(1, len(fs.REGRAS) + 1)),
            "SE (condições)": [fs.texto_condicao(i) for i in range(len(fs.REGRAS))],
            "ENTÃO dependência é": [cons.upper() for _, cons in fs.REGRAS],
            f"Força para o cliente {cliente}": forca_k,
        })
        st.dataframe(
            tabela_regras, hide_index=True,
            column_config={
                "Nº": st.column_config.NumberColumn(width="small"),
                "SE (condições)": st.column_config.TextColumn(width="large"),
                f"Força para o cliente {cliente}": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="percent"),
            },
        )
        st.caption(f"{int((forca_k > 0).sum())} das {len(fs.REGRAS)} regras se aplicam a este cliente.")

    with st.container(border=True):
        st.subheader("Quais regras pesam para este cliente?")
        barras = pd.DataFrame({
            "Regra": [f"Regra {i + 1}" for i in range(len(fs.REGRAS))],
            "Força": forca_k,
            "Resultado": [f"dependência {cons}" for _, cons in fs.REGRAS],
        })
        st.altair_chart(
            alt.Chart(barras).mark_bar().encode(
                y=alt.Y("Regra:N", sort=list(barras["Regra"]), title=None),
                x=alt.X("Força:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("Resultado:N", scale=alt.Scale(
                    domain=["dependência baixa", "dependência média", "dependência alta"],
                    range=fs.COR_CLASSE)),
                tooltip=["Regra", "Resultado", alt.Tooltip("Força:Q", format=".0%")],
            ).properties(height=330),
            width="stretch",
        )
        st.caption("A cor mostra a conclusão de cada regra: verde = baixa dependência, amarelo = média, vermelho = alta.")

        st.markdown("##### Exemplo de cálculo das duas regras mais fortes")
        ordem = np.argsort(-forca_k)[:2]
        for r_i in ordem:
            if forca_k[r_i] <= 0:
                continue
            cond, cons = fs.REGRAS[r_i]
            partes = []
            for e, t in cond:
                partes.append(f"{fs.ENTRADAS[e]['rotulo'].lower()} {t.upper()} ({pct(det['mu'][e][0][fs.TERMOS.index(t)])})")
            st.markdown(
                f"- **Regra {r_i + 1}:** {' E '.join(partes)}. Força = menor valor = **{pct(forca_k[r_i])}**. "
                f"Conclusão: dependência **{cons.upper()}**."
            )

# ==========================================================================
# 5. Score
# ==========================================================================
with t5:
    with st.container(border=True):
        st.subheader("Passo 3 · Agregação")
        st.markdown(
            "Cada regra aponta para uma conclusão: dependência baixa, média ou alta. Para cada conclusão, vale a **regra mais forte** "
            "que aponta para ela (o \"OU\" fuzzy é o **máximo**):"
        )
        forca_cons = {
            cl: max([float(forca_k[i]) for i, (_, cons) in enumerate(fs.REGRAS) if cons == fs.TERMOS[n]] or [0.0])
            for n, cl in enumerate(fs.CLASSES)
        }
        with st.container(horizontal=True):
            for cl, v in forca_cons.items():
                st.metric(f"Força para {cl.lower()}", pct(v), border=True)
        st.markdown(
            "Cada conclusão é uma região da escala de 0 a 100 (contornos cinzas no gráfico). Uma conclusão com força 60%, por exemplo, "
            "preenche essa região só até a altura de 60%. Juntando as três, obtém-se a **área laranja**."
        )

    with st.container(border=True):
        st.subheader("Passo 4 · Defuzzificação: o score")
        st.markdown(
            "Para transformar a área em **um número**, usa-se o **centro de gravidade** (o ponto onde a área se equilibra): "
            "score = soma de (posição × altura) ÷ soma das alturas. Em vez de fazer a conta, basta ver onde a **linha vermelha** cai."
        )
        grade = det["grade"]
        contornos = pd.DataFrame({
            "x": np.tile(grade, 3), "grau": det["saida"].ravel(), "faixa": np.repeat(fs.CLASSES, len(grade)),
        })
        area = pd.DataFrame({"x": grade, "grau": det["agregada"][0]})
        rotulos_x = pd.DataFrame({"x": [8, 50, 92], "grau": [1.08] * 3, "txt": ["baixa", "média", "alta"]})
        x_s = alt.X("x:Q", title="score de dependência do crédito (0 a 100)", scale=alt.Scale(domain=[0, 100]))
        y_s = alt.Y("grau:Q", title="grau de pertinência", scale=alt.Scale(domain=[0, 1.15]),
                    axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
        g_cont = alt.Chart(contornos).mark_line(color=CINZA, strokeWidth=1.5).encode(x=x_s, y=y_s, detail="faixa:N")
        g_area = alt.Chart(area).mark_area(color="#f58518", opacity=0.65).encode(x=x_s, y=y_s)
        g_reg = alt.Chart(pd.DataFrame({"x": [score_k]})).mark_rule(color="#e45756", strokeWidth=3).encode(x=x_s)
        g_txt = alt.Chart(rotulos_x).mark_text(color=CINZA, fontSize=12).encode(x=x_s, y=y_s, text="txt:N")
        st.altair_chart((g_cont + g_area + g_reg + g_txt).properties(height=300), width="stretch")
        pk = det["pert_saida"][0]
        with st.container(horizontal=True):
            st.metric(f"Score do cliente {cliente}", f"{score_k:.0f}", border=True)
            st.metric("Classe (maior pertinência)", fs.CLASSES[cls_k], border=True)
        st.markdown("##### O score também é fuzzy")
        with st.container(horizontal=True):
            for nome_cl, valor_cl in zip(fs.CLASSES, pk):
                st.metric(nome_cl, pct(valor_cl), border=True)
        st.caption(
            "O score é um ponto de equilíbrio, por isso raramente chega a 0 ou 100 "
            f"(nesta base vai de {score_todos.min():.0f} a {score_todos.max():.0f}). A classe é a faixa em que o score tem "
            "maior pertinência. Um score de 62, por exemplo, é um pouco \"média\" e mais \"alta\": o cliente pode pertencer a duas classes ao mesmo tempo."
        )

# ==========================================================================
# 6. Resultados
# ==========================================================================
with t6:
    entre = float((R["pert_saida"].max(axis=1) < 0.6).mean())
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

    with st.container(border=True):
        st.subheader("Perfil de cada classe")
        base_cl = df.assign(_classe=np.array(fs.CLASSES)[classe_todos], _uso=V["uso"], _pag=V["pagamento"],
                            _saq=V["saque"], _comp=V["compras"])
        resumo = base_cl.groupby("_classe").agg(
            clientes=("CUST_ID", "size"), uso=("_uso", "mean"), pag=("_pag", "mean"), saq=("_saq", "mean"),
            comp=("_comp", "mean"), saldo=("BALANCE", "mean"), saque_valor=("CASH_ADVANCE", "mean"),
            compras_valor=("PURCHASES", "mean"),
        ).reindex(fs.CLASSES)
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
            "dinheiro com frequência. A de baixa dependência faz o contrário. Isso mostra que o critério "
            "definido pelas regras separa comportamentos diferentes. Mas é uma **consistência** com a nossa intuição, não uma validação "
            "contra inadimplência."
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
# 7. Conclusões
# ==========================================================================
with t7:
    with st.container(border=True):
        st.subheader("O que foi feito")
        st.markdown(
            "1. **Dados:** 8.950 clientes de cartão, 17 características de comportamento, sem rótulo de bom ou mau pagador.\n"
            "2. **Conceito:** a lógica fuzzy troca o \"sim ou não\" por graus de pertinência entre 0 e 1.\n"
            "3. **Fuzzificação:** 4 informações viram graus de baixa, média e alta por meio de curvas com 3 pontos de corte.\n"
            "4. **Regras:** 13 regras SE-ENTÃO combinam as informações (E = mínimo).\n"
            "5. **Score:** as regras ativas formam uma área; seu centro de gravidade é o score de 0 a 100.\n"
            "6. **Resultados:** todos os clientes receberam score e classe, e muitos ficam entre duas classes."
        )
    with st.container(border=True):
        st.subheader("O que a lógica fuzzy trouxe")
        st.markdown(
            "- **Transição gradual:** dois clientes parecidos recebem scores parecidos, sem saltos bruscos.\n"
            "- **Explicabilidade:** para qualquer cliente dá para mostrar exatamente por que ele recebeu aquele score (abas 3 a 5).\n"
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
