"""App Streamlit: agrupamento fuzzy (Fuzzy C-Means) de clientes de cartão de crédito.

Rodar:  python -m streamlit run app_fuzzy_cartao.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

import fcm
import fuzzy_score as fs

def dec(x: float, casas: int = 1) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


ARQUIVO = Path(__file__).parent / "data" / "CC_GENERAL.csv"
FAIXA_C = range(2, 9)

st.set_page_config(
    page_title="Lógica fuzzy · Cartões de crédito",
    page_icon=":material/credit_card:",
    layout="wide",
)


# --------------------------------------------------------------------------
# Dados e cálculos (com cache)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar() -> pd.DataFrame:
    return pd.read_csv(ARQUIVO)


@st.cache_data(show_spinner="Calculando índices de validação para c = 2 a 8...", max_entries=10)
def varredura_cache(colunas: tuple, m: float, n_init: int, seed: int, usar_log: bool) -> pd.DataFrame:
    X = fcm.preparar_dados(carregar(), list(colunas), usar_log)
    return fcm.varredura(X, m, FAIXA_C, n_init, seed)


@st.cache_data(show_spinner="Ajustando o Fuzzy C-Means...", max_entries=10)
def ajuste_cache(colunas: tuple, c: int, m: float, n_init: int, seed: int, usar_log: bool) -> dict:
    X = fcm.preparar_dados(carregar(), list(colunas), usar_log)
    res = fcm.fcm_multi(X, c, m, n_init, seed)
    pca = PCA(n_components=3).fit(X)
    return {
        "U": res.U,
        "centros": res.centros,
        "J": res.J,
        "n_iter": res.n_iter,
        "indices": fcm.indices(X, res, seed),
        "kmeans": fcm.comparar_kmeans(X, c, res.rotulos, seed),
        "pca": pca.transform(X),
        "var_pca": pca.explained_variance_ratio_,
        "pc1_pesos": pca.components_[0],
    }


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
df = carregar()
todas = [c for c in df.columns if c != "CUST_ID"]

with st.sidebar:
    st.header("Agrupamento (Fuzzy C-Means)")
    st.caption(
        "Estes controles valem só para as abas de agrupamento (Qualidade, Grupos, Pertinência, Curvas dos "
        "grupos, Tabela e Interpretação). A aba **Score fuzzy** tem suas próprias regras."
    )
    colunas = st.multiselect(
        "Variáveis usadas no agrupamento",
        options=todas,
        default=todas,
        format_func=fcm.rotulo,
        help="As características de cada cliente que o programa compara para decidir quem é parecido com quem. "
             "Tirar variáveis muda os grupos.",
    )
    m = st.slider(
        "Fuzzificador (m)", 1.1, 3.0, 1.5, 0.1,
        help="Controla o quanto os grupos se 'misturam'. Perto de 1, cada cliente fica quase 100% em um só "
             "grupo (parecido com um agrupamento comum). Valores maiores dividem a pertinência entre vários "
             "grupos. Padrão: 1,5. Nesta base, m = 2 faz todos os grupos ficarem iguais (colapsam).",
    )
    modo_c = st.segmented_control(
        "Nº de grupos", ["Automático", "Manual"], default="Automático",
        help="Automático: o programa testa de 2 a 8 grupos e escolhe o melhor segundo Silhouette e Xie-Beni "
             "(veja a aba Qualidade). Manual: você escolhe o número no controle abaixo.",
    )
    c_manual = st.slider("c (manual)", 2, 8, 4, disabled=(modo_c != "Manual"),
                         help="Número de grupos quando o modo é Manual.")
    limiar = st.slider(
        "Limiar de fronteira", 0.40, 0.90, 0.60, 0.05,
        help="Se a MAIOR pertinência de um cliente for menor que este valor, ele é 'de fronteira' (indeciso "
             "entre grupos). Exemplo com 60%: um cliente com 55% no Grupo 1 e 45% no Grupo 2 é de fronteira; "
             "um com 90% e 10% não é.",
    )
    usar_log = st.toggle(
        "Aplicar log(1+x) nas variáveis assimétricas", value=True,
        help="Valores em dinheiro têm poucos clientes com quantias enormes. O log comprime essa escala para "
             "que esses poucos não dominem o agrupamento.",
    )
    with st.expander("Avançado"):
        n_init = st.number_input(
            "Inicializações", 1, 20, 5,
            help="Quantas vezes o algoritmo roda com centros iniciais sorteados diferentes. Fica o melhor resultado.")
        seed = st.number_input("Semente", 0, 9999, 42,
                               help="Número que fixa o sorteio, para o resultado ser sempre o mesmo.")

if len(colunas) < 2:
    st.warning("Escolha pelo menos 2 variáveis na barra lateral.", icon=":material/warning:")
    st.stop()

# Cabeçalho aparece antes dos cálculos pesados.
st.title("Lógica fuzzy aplicada a clientes de cartão de crédito")
st.caption(
    f"{len(df):,} clientes · base: Credit Card Dataset for Clustering (Kaggle). ".replace(",", ".")
    + "Duas técnicas: (1) um score calculado por regras fuzzy e (2) um agrupamento Fuzzy C-Means."
)

colunas_t = tuple(colunas)
tabela = varredura_cache(colunas_t, m, int(n_init), int(seed), usar_log)
c_sugerido = fcm.sugerir_c(tabela)
c = c_manual if modo_c == "Manual" else c_sugerido
r = ajuste_cache(colunas_t, c, m, int(n_init), int(seed), usar_log)

U: np.ndarray = r["U"]
rotulos = U.argmax(axis=1)
pert_max = U.max(axis=1)
fronteira = pert_max < limiar
nomes_grupo = [f"Grupo {i + 1}" for i in range(c)]
grupo_txt = np.array(nomes_grupo)[rotulos]
tamanhos = np.bincount(rotulos, minlength=c)

dados = pd.DataFrame({
    "CUST_ID": df["CUST_ID"],
    "Grupo": grupo_txt,
    "Pertinência máxima": pert_max,
    "Fronteira": fronteira,
})
for i, nome in enumerate(nomes_grupo):
    dados[f"Pertinência {nome}"] = U[:, i]

escala_cor = alt.Scale(domain=nomes_grupo, scheme="tableau10")

# nome e frase automáticos de cada grupo (usados em várias abas)
perfis = [fcm.nomear_grupo(v, colunas) for v in r["centros"]]
for nome_perfil in set(perfis):
    if perfis.count(nome_perfil) > 1:  # nomes repetidos ganham numeração
        seq = iter(range(1, perfis.count(nome_perfil) + 1))
        perfis = [f"{p} ({next(seq)})" if p == nome_perfil else p for p in perfis]
frases = [fcm.frase_grupo(v, colunas) for v in r["centros"]]

tab_score, tab_geral, tab_grupos, tab_pert, tab_curvas, tab_tabela, tab_interp = st.tabs([
    ":material/speed: Score fuzzy",
    ":material/dashboard: Qualidade",
    ":material/bubble_chart: Grupos",
    ":material/blur_on: Pertinência",
    ":material/show_chart: Curvas dos grupos",
    ":material/table: Tabela",
    ":material/lightbulb: Interpretação",
])

# --------------------------------------------------------------------------
# Score fuzzy (sistema de inferência por regras)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def score_base() -> tuple[pd.DataFrame, dict]:
    valores = fs.valores_entrada(carregar())
    return valores, fs.inferir(valores)


CLASSES = ["Baixa dependência", "Média dependência", "Alta dependência"]
COR_CLASSE = ["#2ca02c", "#eeca3b", "#d62728"]
COR_TERMO = {"baixa": "#2ca02c", "média": "#d62728", "alta": "#1f77b4"}
CINZA = "#8b8f98"


def grafico_termos(pontos, valor: float, graus3: np.ndarray, titulo: str) -> alt.Chart:
    """Curvas baixa/média/alta de uma entrada, com a faixa dominante do cliente em destaque."""
    q1, q2, q3 = pontos
    t = {"q1": q1, "q2": q2, "q3": q3, "xmin": 0.0, "xmax": 1.0}
    vert = fcm.vertices_termos(t)
    grau_t = dict(zip(fs.TERMOS, graus3))
    dominante = max(grau_t, key=grau_t.get)
    x_enc = alt.X("x:Q", title="valor da entrada", scale=alt.Scale(domain=[0, 1], nice=False),
                  axis=alt.Axis(values=[0, q1, q2, q3, 1.0], format="%", labelFontSize=10))
    y_enc = alt.Y("grau:Q", title="grau de pertinência", scale=alt.Scale(domain=[0, 1.05]),
                  axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))
    fundo = alt.Chart(vert[vert["termo"] != dominante]).mark_line(
        strokeWidth=1.5, color=CINZA, opacity=0.6).encode(x=x_enc, y=y_enc, detail="termo:N")
    frente = alt.Chart(vert[vert["termo"] == dominante]).mark_line(
        strokeWidth=5, color=COR_TERMO[dominante]).encode(x=x_enc, y=y_enc)
    regua = alt.Chart(pd.DataFrame({"x": [valor]})).mark_rule(
        strokeDash=[5, 4], color="#c9ccd1", strokeWidth=1.5).encode(x=x_enc)
    pts = pd.DataFrame({"x": [valor] * 3, "grau": list(grau_t.values()), "termo": list(grau_t)})
    pts["cor"] = [COR_TERMO[tm] if tm == dominante else CINZA for tm in pts["termo"]]
    pontos_g = alt.Chart(pts).mark_point(filled=True, size=90, opacity=1).encode(
        x=x_enc, y=y_enc, color=alt.Color("cor:N", scale=None),
        tooltip=["termo", alt.Tooltip("grau:Q", format=".2f")])
    tit = alt.TitleParams(titulo, fontSize=13, anchor="start", color=COR_TERMO[dominante])
    return (fundo + frente + regua + pontos_g).properties(height=210, title=tit)


with tab_score:
    V, R = score_base()
    score_todos, classe_todos = R["score"], R["classe"]

    # ---------------- O que é ----------------
    with st.container(border=True):
        st.subheader("O que é este score?")
        st.markdown(
            "Aqui a lógica fuzzy é usada do jeito clássico, em quatro passos. **Nada é sorteado nem aprendido: "
            "as curvas e as regras são definidas por nós, como faria um analista de crédito**, e o programa "
            "só aplica essas regras a cada cliente para chegar a **uma nota de 0 a 100: a dependência do crédito**.\n\n"
            "- **0** = usa o cartão como meio de pagamento: usa pouco do limite, paga a fatura inteira, não saca dinheiro.\n"
            "- **100** = usa o cartão como empréstimo: limite quase todo usado, só paga parte da fatura, saca dinheiro.\n\n"
            "1. **Fuzzificação:** cada informação do cliente vira graus entre 0 e 1 de \"baixa\", \"média\" e \"alta\".\n"
            "2. **Regras SE-ENTÃO:** combinam essas informações. Ex.: *SE a frequência de saque é ALTA E o pagamento "
            "integral é BAIXO, ENTÃO a dependência é ALTA*.\n"
            "3. **Agregação:** cada regra que \"dispara\" contribui com uma área, e juntamos todas.\n"
            "4. **Defuzzificação:** o centro de gravidade dessa área é o **score**."
        )
        st.caption(
            "Importante: este score é uma proposta de critério, não foi validado com dados de inadimplência. "
            "Serve para mostrar a lógica fuzzy funcionando de ponta a ponta."
        )
        st.markdown("##### As 4 informações usadas (entradas)")
        st.dataframe(
            pd.DataFrame([{
                "Entrada": e["rotulo"],
                "Como é calculada": e["formula"],
                "O que significa": e["ajuda"],
                "BAIXA quando o cliente...": e["baixa"],
                "ALTA quando o cliente...": e["alta"],
            } for e in fs.ENTRADAS.values()]),
            hide_index=True,
        )

    # ---------------- Escolha do cliente ----------------
    n_ativas = (R["forca"] > 0.2).sum(axis=1)
    candidatos = np.flatnonzero(n_ativas >= 3)
    i_padrao = int(candidatos[np.argmin(np.abs(score_todos[candidatos] - 60))]) if len(candidatos) else 0
    ids_s = df["CUST_ID"].tolist()
    cli_s = st.selectbox("Cliente de exemplo", ids_s, index=i_padrao, key="cli_score",
                         help="Troque o cliente e todos os passos abaixo se atualizam.")
    k = ids_s.index(cli_s)
    um = V.iloc[[k]]
    det = fs.inferir(um, com_curva=True)
    score_k = float(det["score"][0])
    cls_k = int(det["classe"][0])
    limite_k = float(df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median()).iloc[k])
    st.markdown(
        f"**Cliente {cli_s}:** saldo {fcm.formatar_valor('BALANCE', float(df['BALANCE'].iloc[k]))} "
        f"para um limite de {fcm.formatar_valor('CREDIT_LIMIT', limite_k)}, "
        f"frequência de compras {fcm.fmt_pct(V['compras'].iloc[k], 0)}, "
        f"frequência de saque {fcm.fmt_pct(V['saque'].iloc[k], 0)}, "
        f"faturas pagas por inteiro {fcm.fmt_pct(V['pagamento'].iloc[k], 0)}."
    )

    # ---------------- Passo 1 ----------------
    with st.container(border=True):
        st.subheader("Passo 1 · Fuzzificação")
        st.markdown(
            "Cada gráfico é uma informação do cliente. As três curvas (**baixa, média, alta**) dizem quanto aquele "
            "valor pertence a cada faixa. A linha tracejada marca o cliente. **A faixa em que ele mais se encaixa fica "
            "colorida e grossa; as outras ficam cinza.** Onde duas curvas se cruzam, o valor é um pouco de cada."
        )
        cols_g = st.columns(2)
        for n, (chave, e) in enumerate(fs.ENTRADAS.items()):
            g3 = det["mu"][chave][0]
            dom = fs.TERMOS[int(np.argmax(g3))]
            titulo = f"{e['rotulo']}: {fcm.fmt_pct(V[chave].iloc[k], 0)} → {dom} ({fcm.fmt_pct(g3.max(), 0)})"
            with cols_g[n % 2]:
                st.altair_chart(grafico_termos(e["pontos"], float(V[chave].iloc[k]), g3, titulo), width="stretch")
        st.caption(
            "Os números do eixo são os pontos de corte escolhidos: \"baixa\" vale 100% até o primeiro, "
            "\"média\" tem o pico no segundo e \"alta\" chega a 100% no terceiro."
        )

    # ---------------- Passo 2 ----------------
    with st.container(border=True):
        st.subheader("Passo 2 · Regras SE-ENTÃO")
        st.markdown(
            "Cada regra combina duas ou três condições. A **força** da regra é o **menor** grau entre as condições "
            "(o \"E\" fuzzy é o mínimo: a corrente é tão forte quanto o elo mais fraco). Força 0% = a regra não se "
            "aplica a este cliente; quanto maior, mais ela pesa."
        )
        forca_k = det["forca"][0]
        tabela_regras = pd.DataFrame({
            "SE (condições)": [fs.texto_condicao(i) for i in range(len(fs.REGRAS))],
            "ENTÃO dependência é": [cons.upper() for _, cons in fs.REGRAS],
            "Força neste cliente": forca_k,
        }).sort_values("Força neste cliente", ascending=False)
        st.dataframe(
            tabela_regras, hide_index=True,
            column_config={
                "SE (condições)": st.column_config.TextColumn(width="large"),
                "Força neste cliente": st.column_config.ProgressColumn(
                    "Força neste cliente", min_value=0.0, max_value=1.0, format="percent"),
            },
        )
        st.caption(f"{int((forca_k > 0).sum())} de {len(fs.REGRAS)} regras se aplicam a este cliente.")

    # ---------------- Passo 3 ----------------
    with st.container(border=True):
        st.subheader("Passo 3 · Agregação e defuzzificação (o score)")
        st.markdown(
            "O resultado de cada regra é uma região da escala 0-100 (**baixa, média ou alta dependência**, contornos "
            "cinzas). A regra \"corta\" essa região na altura da sua força. Juntando os cortes de todas as regras "
            "obtém-se a **área laranja**. O **score é o ponto de equilíbrio (centro de gravidade) dessa área**."
        )
        grade = det["grade"]
        contornos = pd.DataFrame({
            "x": np.tile(grade, 3), "grau": det["saida"].ravel(),
            "faixa": np.repeat(CLASSES, len(grade)),
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
            st.metric("Score do cliente", f"{score_k:.0f}", border=True)
            st.metric("Classe (maior pertinência)", CLASSES[cls_k], border=True)
            for nome_cl, valor_cl in zip(CLASSES, pk):
                st.metric(nome_cl, fcm.fmt_pct(valor_cl, 0), border=True)
        st.caption(
            "O score é calculado como centro de gravidade, por isso raramente chega a 0 ou 100 (nesta base vai "
            f"de {score_todos.min():.0f} a {score_todos.max():.0f}). A classe é a faixa em que o score tem a maior "
            "pertinência: score 62, por exemplo, é um pouco \"média\" e mais \"alta\"."
        )

    # ---------------- Base inteira ----------------
    with st.container(border=True):
        st.subheader("O mesmo score para toda a base")
        st.markdown(
            "Aplicando as mesmas regras aos 8.950 clientes, cada um recebe um score. Agrupando pela classe, "
            "temos grupos definidos **pelas regras** (diferente do Fuzzy C-Means, em que os grupos saem dos dados)."
        )
        tab_hist = pd.DataFrame({"score": score_todos})
        h = alt.Chart(tab_hist).mark_bar(color="#4c78a8").encode(
            x=alt.X("score:Q", bin=alt.Bin(step=2), title="score de dependência do crédito"),
            y=alt.Y("count():Q", title="clientes"))
        cortes_cl = alt.Chart(pd.DataFrame({"x": [35, 65]})).mark_rule(color="#e45756", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((h + cortes_cl).properties(height=240), width="stretch")
        st.caption("Linhas tracejadas em 35 e 65: onde a classe muda (pontos de cruzamento entre as faixas).")

        base_cl = df.assign(
            _classe=np.array(CLASSES)[classe_todos], _uso=V["uso"], _pag=V["pagamento"],
            _saq=V["saque"], _comp=V["compras"])
        resumo_cl = base_cl.groupby("_classe").agg(
            clientes=("CUST_ID", "size"),
            uso=("_uso", "mean"), pag=("_pag", "mean"), saq=("_saq", "mean"), comp=("_comp", "mean"),
            saldo=("BALANCE", "mean"), saque_valor=("CASH_ADVANCE", "mean"), compras_valor=("PURCHASES", "mean"),
        ).reindex(CLASSES)
        st.dataframe(pd.DataFrame({
            "Classe": CLASSES,
            "Clientes": [f"{int(x):,}".replace(",", ".") for x in resumo_cl["clientes"]],
            "% da base": [fcm.fmt_pct(x / len(df)) for x in resumo_cl["clientes"]],
            "Utilização do limite (média)": [fcm.fmt_pct(x, 0) for x in resumo_cl["uso"]],
            "Faturas pagas por inteiro (média)": [fcm.fmt_pct(x, 0) for x in resumo_cl["pag"]],
            "Frequência de saque (média)": [fcm.fmt_pct(x, 0) for x in resumo_cl["saq"]],
            "Frequência de compras (média)": [fcm.fmt_pct(x, 0) for x in resumo_cl["comp"]],
            "Saldo médio (valor)": [fcm.formatar_valor("BALANCE", x) for x in resumo_cl["saldo"]],
            "Saque médio (valor no período)": [fcm.formatar_valor("CASH_ADVANCE", x) for x in resumo_cl["saque_valor"]],
            "Compras médias (valor no período)": [fcm.formatar_valor("PURCHASES", x) for x in resumo_cl["compras_valor"]],
        }), hide_index=True)

        st.markdown("##### Os grupos do Fuzzy C-Means batem com este score?")
        cruz = pd.crosstab(pd.Series(grupo_txt, name="Grupo"), pd.Series(np.array(CLASSES)[classe_todos], name="Classe"),
                           normalize="index").reindex(columns=CLASSES, fill_value=0)
        cruz.insert(0, "Perfil do grupo", [perfis[i] for i in range(c)])
        cruz.insert(1, "Score médio", [float(score_todos[rotulos == i].mean()) for i in range(c)])
        st.dataframe(
            cruz.reset_index(), hide_index=True,
            column_config={
                "Score médio": st.column_config.NumberColumn(format="%.0f"),
                **{cl: st.column_config.ProgressColumn(f"% em {cl.lower()}", min_value=0.0, max_value=1.0, format="percent")
                   for cl in CLASSES},
            },
        )
        st.caption(
            "Cada linha é um grupo do agrupamento Fuzzy C-Means (aba Interpretação); as colunas mostram que parte "
            "dele cai em cada classe do score. Se os dois métodos concordam, o grupo de \"saque\" deve ter score "
            "médio mais alto que o de \"compradores\". É um teste de consistência, não uma prova."
        )

# --------------------------------------------------------------------------
# Qualidade do agrupamento
# --------------------------------------------------------------------------
with tab_geral:
    with st.container(horizontal=True):
        st.metric("Clientes", f"{len(df):,}".replace(",", "."), border=True)
        st.metric("Grupos (c)", c, delta="sugerido" if c == c_sugerido else f"sugerido: {c_sugerido}",
                  delta_color="off", border=True,
                  help="Quantos grupos o agrupamento formou. 'Sugerido' = o número que o programa escolheu sozinho.")
        st.metric("Silhouette", f"{r['indices']['Silhouette']:.3f}", border=True,
                  help="De -1 a 1. Mede o quanto cada cliente está mais perto do próprio grupo do que do grupo vizinho. "
                       "Acima de 0,5 = grupos bem definidos; 0,25 a 0,5 = razoáveis; abaixo disso = grupos sobrepostos.")
        st.metric("Coef. de partição", f"{r['indices']['PC']:.3f}", border=True,
                  help=f"De {1 / c:.2f} a 1. Perto de 1 = a maioria dos clientes pertence quase 100% a um só grupo; "
                       f"perto de {1 / c:.2f} = tudo misturado.")
        st.metric("Clientes de fronteira", fcm.fmt_pct(fronteira.mean()),
                  delta=f"{int(fronteira.sum()):,} clientes".replace(",", "."), delta_color="off", border=True,
                  help="Clientes cuja maior pertinência é menor que o limiar escolhido na barra lateral: "
                       "estão 'entre' grupos.")

    sil = r["indices"]["Silhouette"]
    leitura = ("bem definidos" if sil > 0.5 else "razoáveis" if sil >= 0.25 else "pouco definidos (os grupos se sobrepõem bastante)")
    st.info(
        f"**Em resumo:** com {c} grupos, a Silhouette é {dec(sil, 3)}, o que indica grupos **{leitura}**. "
        f"{fcm.fmt_pct(fronteira.mean(), 0)} dos clientes ficam na fronteira entre grupos. Isso é comum em dados de "
        "comportamento de clientes: eles formam um contínuo, não \"ilhas\" separadas. É justamente por isso que a lógica "
        "fuzzy ajuda: em vez de forçar o cliente em um grupo, ela mostra o quanto ele pertence a cada um.",
        icon=":material/info:",
    )

    with st.container(border=True):
        st.subheader("O que significa cada índice")
        st.markdown(
            "O Fuzzy C-Means precisa que você diga **quantos grupos** formar. Estes índices servem para **julgar a qualidade "
            "do agrupamento** e ajudar a escolher esse número. Os gráficos abaixo mostram cada índice para 2 a 8 grupos."
        )
        st.dataframe(pd.DataFrame([
            {"Índice": "Silhouette", "Melhor quando": "maior",
             "Em português": "Para cada cliente, compara a distância até os colegas do próprio grupo com a distância até o grupo vizinho. "
                             "Vai de -1 a 1: perto de 1 = bem encaixado; perto de 0 = na divisa entre dois grupos; negativo = provavelmente no grupo errado."},
            {"Índice": "Xie-Beni (XB)", "Melhor quando": "menor",
             "Em português": "Razão entre o quão espalhados estão os clientes dentro dos grupos e a menor distância entre dois centros. "
                             "Valor baixo = grupos compactos e bem afastados uns dos outros."},
            {"Índice": "Coeficiente de partição (PC)", "Melhor quando": "maior",
             "Em português": "Média dos quadrados das pertinências. Vale 1 quando todo cliente pertence 100% a um grupo e 1/c quando está totalmente "
                             "dividido entre todos. Quase sempre é maior com menos grupos."},
            {"Índice": "Entropia da partição (PE)", "Melhor quando": "menor",
             "Em português": "Mede a \"confusão\" das pertinências: 0 = ninguém dividido entre grupos; cresce quanto mais os clientes "
                             "ficam divididos. Também favorece poucos grupos."},
        ]), hide_index=True)

    with st.container(border=True):
        st.subheader("Índices de validação por número de grupos")
        st.caption(
            "A sugestão automática usa a média dos rankings de **Silhouette** (maior é melhor) e "
            "**Xie-Beni** (menor é melhor). PC e PE melhoram quase sempre quando o número de grupos diminui, então "
            "são mostrados só como apoio. A linha vermelha marca o número de grupos em uso."
        )
        definicoes = [
            ("PC", "Coeficiente de partição (↑ melhor)"),
            ("PE", "Entropia da partição (↓ melhor)"),
            ("XB", "Xie-Beni (↓ melhor)"),
            ("Silhouette", "Silhouette (↑ melhor)"),
        ]
        graficos = []
        for col, titulo in definicoes:
            base = alt.Chart(tabela).encode(x=alt.X("c:O", title="nº de grupos"))
            linha = base.mark_line(point=True).encode(y=alt.Y(f"{col}:Q", title=None, scale=alt.Scale(zero=False)))
            marca = alt.Chart(pd.DataFrame({"c": [c]})).mark_rule(color="red", strokeDash=[4, 3]).encode(x="c:O")
            graficos.append((linha + marca).properties(title=titulo, height=200))
        linha1, linha2 = st.columns(2), st.columns(2)
        for caixa, graf in zip(linha1 + linha2, graficos):
            with caixa:
                st.altair_chart(graf)
        st.dataframe(tabela.round(3), hide_index=True)

    if m >= 2.0:
        st.warning(
            f"Com m = {dec(m)} e {len(colunas)} variáveis, a pertinência tende a ficar quase igual "
            "para todos os grupos (centros se aproximam da média geral). Tente m entre 1,3 e 1,6.",
            icon=":material/info:",
        )

# --------------------------------------------------------------------------
# Grupos
# --------------------------------------------------------------------------
with tab_grupos:
    pca_xy = r["pca"]
    var_pca = r["var_pca"]
    dados_pca = dados.assign(PC1=pca_xy[:, 0], PC2=pca_xy[:, 1], PC3=pca_xy[:, 2])

    with st.container(border=True):
        st.subheader("Quem é cada grupo")
        st.markdown(
            "O programa só numera os grupos (Grupo 1, Grupo 2...). Os **nomes e frases abaixo são gerados "
            "automaticamente** olhando as características médias de cada grupo. Detalhes na aba **Interpretação**."
        )
        st.dataframe(pd.DataFrame({
            "Grupo": nomes_grupo,
            "Perfil": perfis,
            "Em uma frase": frases,
            "Clientes": [f"{int(t):,}".replace(",", ".") for t in tamanhos],
        }), hide_index=True)

    esq, dir_ = st.columns([3, 2])

    with esq, st.container(border=True):
        st.subheader("Mapa dos grupos (PCA)")
        st.caption(
            "Cada ponto é um cliente, colorido pelo grupo em que tem maior pertinência. Como cada cliente tem "
            f"{len(colunas)} características, o PCA \"achata\" tudo em 2 eixos (PC1 e PC2) que preservam o máximo das "
            f"diferenças entre clientes, só para podermos desenhar (juntos explicam {fcm.fmt_pct(var_pca[0] + var_pca[1], 0)} "
            "da variação). **Pontos mais claros = clientes indecisos** (pertinência máxima baixa), geralmente na divisa entre cores."
        )
        pontos = alt.Chart(dados_pca).mark_circle(size=22).encode(
            x=alt.X("PC1:Q"),
            y=alt.Y("PC2:Q"),
            color=alt.Color("Grupo:N", scale=escala_cor),
            opacity=alt.Opacity("Pertinência máxima:Q", scale=alt.Scale(domain=[1 / c, 1], range=[0.15, 0.9]), legend=None),
            tooltip=["CUST_ID", "Grupo", alt.Tooltip("Pertinência máxima:Q", format=".0%")],
        ).properties(height=420).interactive()
        st.altair_chart(pontos)

    with dir_, st.container(border=True):
        st.subheader("Tamanho dos grupos")
        st.caption("**Nítidos:** clientes com pertinência máxima acima do limiar. **Fronteira:** abaixo do limiar (indecisos).")
        tam = pd.DataFrame({"Grupo": nomes_grupo, "Clientes": tamanhos,
                            "Fronteira": [int(fronteira[rotulos == i].sum()) for i in range(c)]})
        tam["Nítidos"] = tam["Clientes"] - tam["Fronteira"]
        longo = tam.melt(id_vars="Grupo", value_vars=["Nítidos", "Fronteira"], var_name="Tipo", value_name="n")
        st.altair_chart(
            alt.Chart(longo).mark_bar().encode(
                x=alt.X("Grupo:N", title=None, sort=nomes_grupo, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("n:Q", title="clientes"),
                color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Nítidos", "Fronteira"], range=["#4c78a8", "#e45756"])),
                tooltip=["Grupo", "Tipo", "n"],
            ).properties(height=420)
        )

    with st.container(border=True):
        st.subheader("Perfil dos grupos")
        st.caption(
            "Cada coluna é um grupo e cada linha uma característica. O número mostra **quantos desvios-padrão a média do "
            "grupo está acima (+, azul) ou abaixo (−, vermelho) da média de todos os clientes**. Exemplo: +1,2 em "
            "\"Saque no crédito\" quer dizer que o grupo saca bem mais que o cliente típico; −0,8 em \"Compras\" quer "
            "dizer que compra bem menos. Perto de 0 = igual à média. As unidades de cada variável estão entre parênteses."
        )
        centros = pd.DataFrame(r["centros"], columns=[fcm.rotulo_longo(x) for x in colunas], index=nomes_grupo)
        longo = centros.reset_index(names="Grupo").melt(id_vars="Grupo", var_name="Variável", value_name="z")
        lim = float(max(1.0, np.abs(longo["z"]).max()))
        mapa = alt.Chart(longo).mark_rect().encode(
            x=alt.X("Grupo:N", sort=nomes_grupo, axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("Variável:N", sort=[fcm.rotulo_longo(x) for x in colunas], title=None),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="redblue", domain=[-lim, lim]), title="desv.-padrão"),
            tooltip=["Grupo", "Variável", alt.Tooltip("z:Q", format="+.2f")],
        )
        texto = alt.Chart(longo).mark_text(fontSize=11).encode(
            x=alt.X("Grupo:N", sort=nomes_grupo), y=alt.Y("Variável:N", sort=[fcm.rotulo_longo(x) for x in colunas]),
            text=alt.Text("z:Q", format="+.1f"),
            color=alt.condition(f"abs(datum.z) > {lim * 0.55}", alt.value("white"), alt.value("black")),
        )
        st.altair_chart((mapa + texto).properties(height=max(260, 24 * len(colunas))))

# --------------------------------------------------------------------------
# Pertinência
# --------------------------------------------------------------------------
with tab_pert:
    with st.container(border=True):
        st.subheader("O que é pertinência?")
        st.markdown(
            f"No Fuzzy C-Means, cada cliente recebe **{c} percentuais, um por grupo, que somam 100%**. Esse percentual é a "
            "**pertinência**: o quanto o cliente se parece com o grupo (mais perto do centro do grupo = maior). Ex.: "
            "70% no Grupo 1 e 30% no Grupo 2 → ele fica no Grupo 1, mas com uma parte de Grupo 2. Esta aba mostra como essas "
            "pertinências se distribuem entre os 8.950 clientes."
        )
    esq, dir_ = st.columns(2)
    with esq, st.container(border=True):
        st.subheader("Quão definidos são os clientes?")
        st.caption(
            "Cada barra conta **quantos clientes** têm determinada pertinência máxima (o maior dos percentuais dele). "
            f"Barras à direita (perto de 100%) = clientes bem definidos. Barras à esquerda do limiar vermelho ({fcm.fmt_pct(limiar, 0)}) = "
            f"clientes de fronteira: {fcm.fmt_pct(fronteira.mean(), 0)} da base."
        )
        hist = alt.Chart(dados).mark_bar().encode(
            x=alt.X("Pertinência máxima:Q", bin=alt.Bin(maxbins=30), title="pertinência máxima"),
            y=alt.Y("count():Q", title="clientes"),
        )
        regra = alt.Chart(pd.DataFrame({"x": [limiar]})).mark_rule(color="red", strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((hist + regra).properties(height=320))

    with dir_, st.container(border=True):
        st.subheader("Pertinência ao longo do eixo PC1")
        pesos = pd.Series(r["pc1_pesos"], index=[fcm.rotulo(x) for x in colunas])
        if pesos.mean() < 0:  # o sinal do eixo é arbitrário: deixa o PC1 crescer com a intensidade de uso
            pesos = -pesos
        altos = ", ".join(pesos.sort_values(ascending=False).index[:3])
        baixos = ", ".join(pesos.sort_values().index[:2])
        st.caption(
            "**PC1** é um eixo resumo que combina as variáveis e separa ao máximo os clientes. Aqui ele cresce principalmente com: "
            f"{altos} (e decresce com {baixos}). Cada linha mostra a pertinência média de um grupo para clientes em cada posição "
            "do eixo: onde a linha de um grupo está no alto, esse grupo domina; onde duas linhas se cruzam, é a fronteira entre eles."
        )
        pc1 = r["pca"][:, 0]
        faixas = pd.cut(pc1, bins=40)
        tabela_mf = pd.DataFrame(U, columns=nomes_grupo)
        tabela_mf["PC1"] = pd.Series(pc1).groupby(faixas, observed=True).transform("mean").to_numpy()
        medias = tabela_mf.groupby("PC1", observed=True).mean().reset_index().melt(
            id_vars="PC1", var_name="Grupo", value_name="Pertinência")
        st.altair_chart(
            alt.Chart(medias).mark_line(point=True).encode(
                x=alt.X("PC1:Q"), y=alt.Y("Pertinência:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("Grupo:N", scale=escala_cor),
            ).properties(height=320)
        )

    with st.container(border=True):
        st.subheader("Clientes mais indecisos")
        st.caption(
            "Lista dos clientes cuja maior pertinência é a mais baixa, isto é, que estão mais no meio do caminho entre grupos. "
            "Cada linha é um cliente; cada célula é o percentual dele em um grupo (mais escuro = maior)."
        )
        n_amb = st.slider("Quantos clientes listar", 10, 60, 25, 5)
        amb_idx = np.argsort(pert_max)[:n_amb]
        amb = pd.DataFrame(U[amb_idx], columns=nomes_grupo)
        amb["Cliente"] = df["CUST_ID"].to_numpy()[amb_idx]
        amb_longo = amb.melt(id_vars="Cliente", var_name="Grupo", value_name="Pertinência")
        base = alt.Chart(amb_longo).encode(
            x=alt.X("Grupo:N", sort=nomes_grupo, axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("Cliente:N", sort=amb["Cliente"].tolist(), title=None),
        )
        mapa = base.mark_rect().encode(color=alt.Color("Pertinência:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                                       legend=alt.Legend(format="%")))
        texto = base.mark_text(fontSize=10).encode(
            text=alt.Text("Pertinência:Q", format=".0%"),
            color=alt.condition(alt.datum["Pertinência"] > 0.55, alt.value("white"), alt.value("black")),
        )
        st.altair_chart((mapa + texto).properties(height=max(250, 18 * n_amb)))

    with st.container(border=True):
        st.subheader("Consultar um cliente")
        st.caption("Escolha um cliente para ver a pertinência dele a cada grupo e os valores que levaram a isso.")
        cliente = st.selectbox("Cliente", df["CUST_ID"], index=0)
        i = int(df.index[df["CUST_ID"] == cliente][0])
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Grupo mais provável", grupo_txt[i], delta=f"{fcm.fmt_pct(pert_max[i])} de pertinência",
                      delta_color="off", border=True)
            if fronteira[i]:
                st.info("Cliente de fronteira: pertence a mais de um grupo.", icon=":material/call_split:")
            pert_cli = pd.DataFrame({"Grupo": nomes_grupo, "Pertinência": U[i]})
            st.altair_chart(
                alt.Chart(pert_cli).mark_bar().encode(
                    x=alt.X("Pertinência:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("Grupo:N", sort=nomes_grupo, title=None),
                    color=alt.Color("Grupo:N", scale=escala_cor, legend=None),
                    tooltip=["Grupo", alt.Tooltip("Pertinência:Q", format=".1%")],
                ).properties(height=40 * c + 30)
            )
        with c2:
            media_todos = df[colunas].mean()
            valores = pd.DataFrame({
                "Variável": [fcm.rotulo_longo(x) for x in colunas],
                "Cliente": [fcm.formatar_valor(x, df.loc[i, x]) if pd.notna(df.loc[i, x]) else "sem dado" for x in colunas],
                "Média geral": [fcm.formatar_valor(x, media_todos[x]) for x in colunas],
            })
            st.dataframe(valores, hide_index=True)

# --------------------------------------------------------------------------
# Tabela
# --------------------------------------------------------------------------
with tab_tabela:
    with st.container(horizontal=True):
        filtro_grupos = st.multiselect("Grupos", nomes_grupo, default=nomes_grupo)
        so_fronteira = st.toggle("Só clientes de fronteira", value=False)
    vis = dados[dados["Grupo"].isin(filtro_grupos)]
    if so_fronteira:
        vis = vis[vis["Fronteira"]]
    vis = vis.sort_values("Pertinência máxima")
    colunas_pert = {
        f"Pertinência {nome}": st.column_config.ProgressColumn(f"Pertinência {nome}", min_value=0.0, max_value=1.0, format="%.2f")
        for nome in nomes_grupo
    }
    st.caption(f"{len(vis):,} clientes".replace(",", "."))
    st.dataframe(
        vis, hide_index=True,
        column_config={**colunas_pert,
                       "Pertinência máxima": st.column_config.NumberColumn(format="%.3f")},
    )
    st.download_button(
        "Baixar CSV", vis.to_csv(index=False).encode("utf-8"),
        file_name="pertinencias_fuzzy.csv", mime="text/csv", icon=":material/download:",
    )

# --------------------------------------------------------------------------
# Interpretação
# --------------------------------------------------------------------------
with tab_interp:
    milhar = lambda n: f"{int(n):,}".replace(",", ".")
    medias_orig = fcm.perfil_original(df, colunas, rotulos).reindex(nomes_grupo)
    media_geral = df[colunas].mean()
    centros_z = r["centros"]

    # ---------------- 0. As duas técnicas ----------------
    with st.container(border=True):
        st.subheader("Resumo: duas técnicas de lógica fuzzy neste app")
        st.dataframe(pd.DataFrame([
            {"": "Pergunta que responde", "Score fuzzy (aba Score fuzzy)": "Qual a dependência do crédito deste cliente, de 0 a 100?",
             "Agrupamento Fuzzy C-Means (demais abas)": "Quais tipos de cliente existem e o quanto este cliente pertence a cada tipo?"},
            {"": "Quem define as curvas", "Score fuzzy (aba Score fuzzy)": "Nós (critério de analista): baixa / média / alta em 4 informações",
             "Agrupamento Fuzzy C-Means (demais abas)": "Ninguém: nascem da distância até o centro de cada grupo"},
            {"": "Quem define as regras", "Score fuzzy (aba Score fuzzy)": "Nós: regras SE-ENTÃO escritas à mão",
             "Agrupamento Fuzzy C-Means (demais abas)": "Não há regras: o algoritmo descobre os grupos olhando os dados"},
            {"": "Resultado", "Score fuzzy (aba Score fuzzy)": "Um score por cliente, e uma classe (baixa, média ou alta dependência)",
             "Agrupamento Fuzzy C-Means (demais abas)": f"Um percentual por grupo para cada cliente ({c} percentuais que somam 100%)"},
        ]), hide_index=True)
        st.caption("As duas técnicas são independentes. A última tabela da aba Score fuzzy compara os resultados.")

    # ---------------- 1. O que o programa faz ----------------
    with st.container(border=True):
        st.subheader("O que o Fuzzy C-Means faz")
        st.markdown(
            f"A base tem **{milhar(len(df))} clientes de cartão de crédito**, cada um descrito por "
            f"**{len(colunas)} números** (quanto compra, quanto saca em dinheiro, limite, pagamentos...). "
            f"O programa **separa os clientes em {c} grupos de comportamento parecido**, sem que ninguém "
            "diga antes quais são os grupos: ele descobre sozinho olhando os dados."
        )
        st.markdown("##### O que é a lógica *fuzzy* (nebulosa)")
        st.markdown(
            "Num agrupamento comum, cada cliente cai em **um único grupo** (sim ou não). "
            "No *fuzzy*, cada cliente recebe um **grau de pertinência** a **todos** os grupos, de 0% a 100%, "
            "somando 100%. Exemplo: 70% no Grupo 1 e 30% no Grupo 2 quer dizer que o cliente se parece com os "
            "dois, mas mais com o Grupo 1. Isso é mais realista, porque ninguém é 100% \"comprador\" ou 100% "
            "\"sacador\": existe gente no meio do caminho."
        )
        st.markdown("##### Passo a passo")
        st.markdown(
            "1. **Limpeza:** valores faltantes viram a mediana da coluna, e frequências acima de 100% são "
            "corrigidas para 100%.\n"
            "2. **Mesma escala:** aplica-se log nos valores em dinheiro e depois se padroniza tudo. Sem isso, "
            "números grandes (milhares) esmagariam os pequenos (porcentagens de 0 a 1).\n"
            f"3. **Chute inicial:** sorteiam-se {c} pontos centrais, um por grupo (os *centros*).\n"
            "4. **Pertinência:** para cada cliente mede-se a distância até cada centro. "
            "Quanto mais perto de um centro, maior a pertinência àquele grupo.\n"
            "5. **Ajuste dos centros:** cada centro é recalculado como a média de todos os clientes, "
            "**ponderada pela pertinência** (quem pertence mais ao grupo puxa mais o centro).\n"
            f"6. **Repetição:** os passos 4 e 5 se repetem até estabilizar (aqui, {r['n_iter']} rodadas na "
            f"melhor tentativa). O processo é refeito {int(n_init)} vezes com sorteios diferentes e fica a "
            "melhor tentativa.\n"
            f"7. **Resultado:** o grupo do cliente é aquele em que ele tem **maior %**. Se essa maior % ficar "
            f"abaixo de {fcm.fmt_pct(limiar, 0)}, o cliente é considerado de **fronteira**."
        )
        st.caption(
            f"O parâmetro m = {dec(m)} controla o quanto a pertinência é \"borrada\": perto de 1, os clientes "
            "ficam quase 100% em um só grupo; valores maiores deixam tudo mais misturado."
        )

    # ---------------- 2. Quem são os grupos ----------------
    st.subheader("Quem são os grupos")
    resumo = pd.DataFrame({
        "Grupo": nomes_grupo,
        "Perfil": perfis,
        "Clientes": [milhar(t) for t in tamanhos],
        "% da base": [fcm.fmt_pct(t / len(df)) for t in tamanhos],
        "Em uma frase": frases,
    })
    st.dataframe(resumo, hide_index=True)

    for i, nome in enumerate(nomes_grupo):
        no_grupo = rotulos == i
        nitidos = int((~fronteira[no_grupo]).sum())
        with st.container(border=True):
            st.markdown(f"### {nome} · {perfis[i]}")
            st.markdown(f"**{frases[i]}**")
            st.markdown(
                f"{milhar(tamanhos[i])} clientes ({fcm.fmt_pct(tamanhos[i] / len(df))} da base). "
                f"{fcm.fmt_pct(nitidos / max(tamanhos[i], 1), 0)} deles têm pertinência acima de "
                f"{fcm.fmt_pct(limiar, 0)} (clientes \"nítidos\" deste grupo)."
            )
            st.markdown("**O que levou o programa a formar este grupo** (as variáveis em que ele mais se afasta da média geral):")
            st.dataframe(
                fcm.principais_diferencas(centros_z[i], colunas, medias_orig.loc[nome], media_geral),
                hide_index=True,
            )
            idx_tipico = np.flatnonzero(no_grupo)[np.argmax(U[no_grupo, i])]
            st.caption(
                f"Cliente mais típico do grupo: {df['CUST_ID'].iloc[idx_tipico]} "
                f"({fcm.fmt_pct(U[idx_tipico, i], 0)} de pertinência)."
            )

    # ---------------- 3. Por que cada cliente cai em um grupo ----------------
    st.subheader("Por que um cliente cai em um grupo? Dois exemplos")

    def exemplo(i: int, titulo: str, explicacao: str) -> None:
        a, b = np.argsort(-U[i])[:2]
        with st.container(border=True):
            st.markdown(
                f"**{titulo}: {df['CUST_ID'].iloc[i]}** — {fcm.fmt_pct(U[i, a], 0)} no {nomes_grupo[a]} "
                f"({perfis[a]}) e {fcm.fmt_pct(U[i, b], 0)} no {nomes_grupo[b]} ({perfis[b]})."
            )
            decisivas = np.argsort(-np.abs(centros_z[a] - centros_z[b]))[:4]
            st.dataframe(
                pd.DataFrame({
                    "Variável": [fcm.rotulo(colunas[j]) for j in decisivas],
                    "Este cliente": [fcm.formatar_valor(colunas[j], df[colunas[j]].iloc[i]) for j in decisivas],
                    f"Média do {nomes_grupo[a]}": [fcm.formatar_valor(colunas[j], medias_orig.loc[nomes_grupo[a], colunas[j]]) for j in decisivas],
                    f"Média do {nomes_grupo[b]}": [fcm.formatar_valor(colunas[j], medias_orig.loc[nomes_grupo[b], colunas[j]]) for j in decisivas],
                }),
                hide_index=True,
            )
            st.caption(explicacao)

    i_claro, i_duvida = int(np.argmax(pert_max)), int(np.argmin(pert_max))
    exemplo(i_claro, "Cliente bem definido",
            "Os valores do cliente estão muito perto das médias do grupo em que ele foi colocado e longe das do outro: "
            "por isso a pertinência é quase total.")
    exemplo(i_duvida, "Cliente de fronteira",
            "Os valores ficam no meio do caminho entre os dois grupos. Por isso a pertinência se divide, "
            "e o programa só o coloca no grupo de maior %. Este é o tipo de informação que um "
            "agrupamento comum esconde.")

    with st.expander("O que significa cada variável (e em que unidade)"):
        st.dataframe(
            pd.DataFrame({
                "Variável": [fcm.rotulo(x) for x in colunas],
                "Unidade": [fcm.UNIDADES.get(x, "") for x in colunas],
                "O que mede": [fcm.DESCRICOES.get(x, "") for x in colunas],
            }),
            hide_index=True,
        )
        st.caption(
            "Valores em dinheiro estão na moeda original da base (não informada) e são **acumulados no período coberto "
            "pela base**, não por mês. A documentação da base não é conclusiva sobre a duração desse período (6 ou 12 meses). "
            "As variáveis de frequência são índices de 0% a 100% calculados pela própria base, e não contagens."
        )

    # ---------------- 4. Comparação com K-Means ----------------
    st.subheader("Fuzzy C-Means × K-Means")
    k = r["kmeans"]
    with st.container(horizontal=True):
        st.metric("Silhouette (fuzzy)", f"{k['silhouette_fcm']:.3f}", border=True)
        st.metric("Silhouette (K-Means)", f"{k['silhouette_kmeans']:.3f}", border=True)
        st.metric("Concordância (ARI)", f"{k['ari']:.3f}", border=True,
                  help="1 = os dois métodos formam exatamente os mesmos grupos.")
    st.caption(
        "O K-Means é o agrupamento \"comum\": põe cada cliente em um único grupo. "
        f"Com m = {dec(m)}, {fcm.fmt_pct(fronteira.mean())} dos clientes ficam na fronteira no fuzzy — "
        "informação que o K-Means não oferece."
    )


# --------------------------------------------------------------------------
# Curvas dos grupos
# --------------------------------------------------------------------------
with tab_curvas:
    TAB10 = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]
    eixo_y = alt.Y("grau:Q", title=None, scale=alt.Scale(domain=[0, 1.05]),
                   axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))

    def eixo_x(t: dict) -> alt.X:
        fmt = ",.0f" if t["q3"] >= 10 else ".2f"
        marcas = [0.0, t["q1"], t["q2"], t["q3"], t["xmax"]]
        escala = alt.Scale(type="symlog" if t["log"] else "linear", nice=False)
        return alt.X("x:Q", title=None, scale=escala,
                     axis=alt.Axis(values=marcas, format=fmt, labelFontSize=10))

    with st.container(border=True):
        st.subheader("O que são estas curvas?")
        st.markdown(
            "Há **um gráfico por variável**. Em cada um, **cada linha colorida é um grupo**.\n\n"
            "- **Eixo horizontal:** o valor da variável (ex.: a frequência de saque).\n"
            "- **Eixo vertical:** a pertinência ao grupo, de 0 a 1. É o quanto um cliente com aquele valor se parece com o grupo.\n"
            "- **Linha no alto:** naquele valor, o cliente se parece muito com o grupo. **Onde duas linhas se cruzam:** "
            "é a fronteira entre dois grupos para aquela variável.\n\n"
            "Detalhe técnico: cada curva responde \"e se o cliente tivesse **este** valor nesta variável e a **média** em "
            "todas as outras?\". Isso isola o efeito de uma variável por vez."
        )
        st.markdown(
            "**Diferença para o Score fuzzy:** lá nós desenhamos as curvas e as regras. Aqui **ninguém desenha nada**: "
            "as curvas nascem da distância até o centro de cada grupo, que o algoritmo descobriu sozinho."
        )

    ids = df["CUST_ID"].tolist()
    cli_id = st.selectbox(
        "Cliente de exemplo", ids, index=int(np.argmin(pert_max)), key="cli_curva",
        help="Escolha um cliente: a linha tracejada de cada gráfico marca o valor dele. Começa no cliente mais "
             "indeciso (mais perto da fronteira entre os grupos).",
    )
    i_cli = ids.index(cli_id)
    pert_cli = " · ".join(f"{nomes_grupo[i]}: {fcm.fmt_pct(U[i_cli, i], 0)}" for i in range(c))
    st.markdown(
        f"**Cliente {cli_id}:** pertinência final aos grupos (considerando todas as variáveis juntas): {pert_cli}."
    )

    with st.container(border=True):
        rot_grupo = [f"{nomes_grupo[i]} · {perfis[i]}" for i in range(c)]
        st.markdown(
            "**Como ler o destaque:** a linha tracejada marca o valor do cliente. O grupo que mais combina com esse "
            "valor, **olhando só aquela variável**, fica grosso; os outros ficam finos e apagados. O título do gráfico "
            "diz qual é (pode ser diferente do grupo final do cliente, que depende de todas as variáveis juntas). "
            "Os números do eixo horizontal são os percentis 25, 50 e 75 da variável."
        )
        st.markdown(" &nbsp; ".join(
            f"<span style='color:{TAB10[i % 8]}'>●</span> {rot_grupo[i]}" for i in range(c)
        ), unsafe_allow_html=True)
        cor_grupo = alt.Scale(domain=rot_grupo, range=TAB10[:c])
        colunas_graf2 = st.columns(2)
        for k, col in enumerate(colunas):
            t = fcm.termos_fuzzy(df, col, usar_log)
            valor = float(fcm._coluna_limpa(df, col).iloc[i_cli])
            curva = fcm.curva_pertinencia(df, colunas, r["centros"], m, col, usar_log)
            curva["Grupo"] = [rot_grupo[g] for g in curva["grupo"]]
            xs = curva["x"].unique()
            x_prox = xs[np.argmin(np.abs(xs - valor))]
            no_ponto = curva[curva["x"] == x_prox].set_index("grupo")["grau"]
            g_dom = int(no_ponto.idxmax())
            curva["larg"] = np.where(curva["grupo"] == g_dom, 5, 1.5)
            curva["op"] = np.where(curva["grupo"] == g_dom, 1.0, 0.45)
            x_curva = eixo_x(t)
            linhas_c = alt.Chart(curva).mark_line().encode(
                x=x_curva, y=eixo_y, color=alt.Color("Grupo:N", scale=cor_grupo, legend=None),
                strokeWidth=alt.StrokeWidth("larg:Q", scale=None, legend=None),
                opacity=alt.Opacity("op:Q", scale=None, legend=None),
                tooltip=["Grupo", alt.Tooltip("x:Q", format=",.2f", title=fcm.rotulo(col)),
                         alt.Tooltip("grau:Q", format=".2f", title="Pertinência")])
            regua_c = alt.Chart(pd.DataFrame({"x": [valor]})).mark_rule(
                strokeDash=[5, 4], color="#c9ccd1", strokeWidth=1.5).encode(x=x_curva)
            titulo = alt.TitleParams(
                f"{fcm.rotulo_longo(col)}: {fcm.formatar_valor(col, valor)} → {nomes_grupo[g_dom]} ({fcm.fmt_pct(no_ponto.max(), 0)})",
                fontSize=13, anchor="start", color=TAB10[g_dom % 8])
            with colunas_graf2[k % 2]:
                st.altair_chart((linhas_c + regua_c).properties(height=190, title=titulo), width="stretch")
