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

def dec(x: float, casas: int = 1) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


ARQUIVO = Path(__file__).parent / "data" / "CC_GENERAL.csv"
FAIXA_C = range(2, 9)

st.set_page_config(
    page_title="Fuzzy C-Means · Cartões de crédito",
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
    }


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
df = carregar()
todas = [c for c in df.columns if c != "CUST_ID"]

with st.sidebar:
    st.header("Parâmetros")
    colunas = st.multiselect(
        "Variáveis usadas no agrupamento",
        options=todas,
        default=todas,
        format_func=fcm.rotulo,
    )
    m = st.slider(
        "Fuzzificador (m)", 1.1, 3.0, 1.5, 0.1,
        help="Quanto maior, mais 'borrada' a pertinência. Nesta base m = 2 faz os grupos "
             "colapsarem (veja a aba Visão geral).",
    )
    modo_c = st.segmented_control("Nº de grupos", ["Automático", "Manual"], default="Automático")
    c_manual = st.slider("c (manual)", 2, 8, 4, disabled=(modo_c != "Manual"))
    limiar = st.slider(
        "Limiar de fronteira", 0.40, 0.90, 0.60, 0.05,
        help="Clientes com pertinência máxima abaixo deste valor são considerados de fronteira.",
    )
    usar_log = st.toggle("Aplicar log(1+x) nas variáveis assimétricas", value=True)
    with st.expander("Avançado"):
        n_init = st.number_input("Inicializações", 1, 20, 5)
        seed = st.number_input("Semente", 0, 9999, 42)

if len(colunas) < 2:
    st.warning("Escolha pelo menos 2 variáveis na barra lateral.", icon=":material/warning:")
    st.stop()

# Cabeçalho aparece antes dos cálculos pesados.
st.title("Agrupamento fuzzy de clientes de cartão de crédito")
st.caption(
    f"Fuzzy C-Means em {len(df):,} clientes e {len(colunas)} variáveis · "
    "base: Credit Card Dataset for Clustering (Kaggle)".replace(",", ".")
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

tab_geral, tab_grupos, tab_pert, tab_curvas, tab_tabela, tab_interp = st.tabs([
    ":material/dashboard: Visão geral",
    ":material/bubble_chart: Grupos",
    ":material/blur_on: Pertinência",
    ":material/show_chart: Curvas fuzzy",
    ":material/table: Tabela",
    ":material/lightbulb: Interpretação",
])

# --------------------------------------------------------------------------
# Visão geral
# --------------------------------------------------------------------------
with tab_geral:
    with st.container(horizontal=True):
        st.metric("Clientes", f"{len(df):,}".replace(",", "."), border=True)
        st.metric("Grupos (c)", c, delta="sugerido" if c == c_sugerido else f"sugerido: {c_sugerido}",
                  delta_color="off", border=True)
        st.metric("Silhouette", f"{r['indices']['Silhouette']:.3f}", border=True)
        st.metric("Coef. de partição", f"{r['indices']['PC']:.3f}", border=True,
                  help="1 = grupos nítidos; 1/c = totalmente difuso.")
        st.metric("Clientes de fronteira", fcm.fmt_pct(fronteira.mean()),
                  delta=f"{int(fronteira.sum()):,} clientes".replace(",", "."), delta_color="off", border=True)

    with st.container(border=True):
        st.subheader("Índices de validação por número de grupos")
        st.caption(
            "A sugestão automática usa a média dos rankings de **Silhouette** (maior é melhor) e "
            "**Xie-Beni** (menor é melhor). PC e PE melhoram quase sempre quando c diminui, então "
            "são mostrados só como apoio."
        )
        definicoes = [
            ("PC", "Coeficiente de partição (↑)"),
            ("PE", "Entropia da partição (↓)"),
            ("XB", "Xie-Beni (↓)"),
            ("Silhouette", "Silhouette (↑)"),
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
    esq, dir_ = st.columns([3, 2])

    with esq, st.container(border=True):
        st.subheader("Mapa dos grupos (PCA)")
        st.caption(
            f"PC1 explica {fcm.fmt_pct(var_pca[0])} e PC2 {fcm.fmt_pct(var_pca[1])} da variância. "
            "Pontos mais transparentes têm pertinência máxima menor."
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
        st.caption("Centróides em desvios-padrão em relação à média geral (azul = acima, vermelho = abaixo).")
        centros = pd.DataFrame(r["centros"], columns=[fcm.rotulo(x) for x in colunas], index=nomes_grupo)
        longo = centros.reset_index(names="Grupo").melt(id_vars="Grupo", var_name="Variável", value_name="z")
        lim = float(max(1.0, np.abs(longo["z"]).max()))
        mapa = alt.Chart(longo).mark_rect().encode(
            x=alt.X("Grupo:N", sort=nomes_grupo, axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("Variável:N", sort=[fcm.rotulo(x) for x in colunas], title=None),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="redblue", domain=[-lim, lim]), title="desv.-padrão"),
            tooltip=["Grupo", "Variável", alt.Tooltip("z:Q", format="+.2f")],
        )
        texto = alt.Chart(longo).mark_text(fontSize=11).encode(
            x=alt.X("Grupo:N", sort=nomes_grupo), y=alt.Y("Variável:N", sort=[fcm.rotulo(x) for x in colunas]),
            text=alt.Text("z:Q", format="+.1f"),
            color=alt.condition(f"abs(datum.z) > {lim * 0.55}", alt.value("white"), alt.value("black")),
        )
        st.altair_chart((mapa + texto).properties(height=max(260, 24 * len(colunas))))

# --------------------------------------------------------------------------
# Pertinência
# --------------------------------------------------------------------------
with tab_pert:
    esq, dir_ = st.columns(2)
    with esq, st.container(border=True):
        st.subheader("Nitidez da pertinência")
        st.caption("Distribuição da maior pertinência de cada cliente. Valores baixos = cliente 'entre grupos'.")
        hist = alt.Chart(dados).mark_bar().encode(
            x=alt.X("Pertinência máxima:Q", bin=alt.Bin(maxbins=30), title="pertinência máxima"),
            y=alt.Y("count():Q", title="clientes"),
        )
        regra = alt.Chart(pd.DataFrame({"x": [limiar]})).mark_rule(color="red", strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((hist + regra).properties(height=320))

    with dir_, st.container(border=True):
        st.subheader("Funções de pertinência ao longo do PC1")
        st.caption("Pertinência média de cada grupo em faixas do primeiro componente principal.")
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
        st.subheader("Clientes mais ambíguos")
        n_amb = st.slider("Quantos clientes mostrar", 10, 60, 25, 5)
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
            valores = pd.DataFrame({
                "Variável": [fcm.rotulo(x) for x in colunas],
                "Cliente": df.loc[i, colunas].to_numpy(dtype=float),
                "Média geral": df[colunas].mean().to_numpy(),
            })
            st.dataframe(valores, hide_index=True, column_config={
                "Cliente": st.column_config.NumberColumn(format="%.2f"),
                "Média geral": st.column_config.NumberColumn(format="%.2f"),
            })

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

    # nome e frase automáticos de cada grupo
    perfis = [fcm.nomear_grupo(v, colunas) for v in centros_z]
    for nome_perfil in set(perfis):
        if perfis.count(nome_perfil) > 1:  # nomes repetidos ganham numeração
            seq = iter(range(1, perfis.count(nome_perfil) + 1))
            perfis = [f"{p} ({next(seq)})" if p == nome_perfil else p for p in perfis]
    frases = [fcm.frase_grupo(v, colunas) for v in centros_z]

    # ---------------- 1. O que o programa faz ----------------
    with st.container(border=True):
        st.subheader("O que este programa faz")
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
            st.markdown("**O que levou o programa a formar este grupo** (as variáveis que mais o separam da média):")
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

    with st.expander("O que significa cada variável"):
        st.dataframe(
            pd.DataFrame({
                "Variável": [fcm.rotulo(x) for x in colunas],
                "Significado": [fcm.DESCRICOES.get(x, "") for x in colunas],
            }),
            hide_index=True,
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
# Curvas fuzzy
# --------------------------------------------------------------------------
with tab_curvas:
    ids = df["CUST_ID"].tolist()
    cli_id = st.selectbox(
        "Cliente de exemplo", ids, index=int(np.argmin(pert_max)), key="cli_curva",
        help="Escolha um cliente: todos os gráficos abaixo se atualizam. Começa no cliente mais indeciso "
             "(mais perto da fronteira entre os grupos).",
    )
    i_cli = ids.index(cli_id)
    pert_cli = " · ".join(f"{nomes_grupo[i]}: {fcm.fmt_pct(U[i_cli, i], 0)}" for i in range(c))
    st.markdown(
        f"**Cliente {cli_id}** — pertinência final aos grupos (todas as variáveis juntas): {pert_cli}."
    )

    COR_TERMO = {"baixa": "#2ca02c", "média": "#d62728", "alta": "#1f77b4"}
    CINZA = "#8b8f98"
    TAB10 = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]
    eixo_y = alt.Y("grau:Q", title=None, scale=alt.Scale(domain=[0, 1.05]),
                   axis=alt.Axis(values=[0, 0.5, 1], format=".1f"))

    def eixo_x(t: dict) -> alt.X:
        fmt = ",.0f" if t["q3"] >= 10 else ".2f"
        marcas = [0.0, t["q1"], t["q2"], t["q3"], t["xmax"]]
        escala = alt.Scale(type="symlog" if t["log"] else "linear", nice=False)
        return alt.X("x:Q", title=None, scale=escala,
                     axis=alt.Axis(values=marcas, format=fmt, labelFontSize=10))

    # ---------- 1. Conjuntos nebulosos clássicos ----------
    with st.container(border=True):
        st.subheader("1. Fuzzificação: baixa, média ou alta?")
        st.markdown(
            "Na lógica fuzzy clássica, um valor não é apenas \"baixo\" ou \"alto\": ele pertence a cada faixa "
            "com um **grau entre 0 e 1**. Cada gráfico abaixo é uma variável. A linha tracejada marca onde o "
            "cliente está. **A faixa em que ele mais se encaixa fica colorida e grossa; as outras ficam cinza.** "
            "Onde duas curvas se cruzam, o valor é um pouco de cada faixa."
        )
        st.markdown(
            ":green[**● baixa**] &nbsp; :red[**● média**] &nbsp; :blue[**● alta**] &nbsp; "
            "(eixo horizontal em escala logarítmica nas variáveis em dinheiro e contagens)"
        )
        st.caption(
            "Pontos de corte nos percentis 25, 50 e 75 de cada variável (valores no eixo). Nas variáveis com "
            "muitos zeros, os percentis são calculados só entre os valores positivos; nas que têm poucos valores "
            "diferentes, a faixa é dividida em partes iguais."
        )
        colunas_graf = st.columns(2)
        for k, col in enumerate(colunas):
            t = fcm.termos_fuzzy(df, col, usar_log)
            valor = float(fcm._coluna_limpa(df, col).iloc[i_cli])
            graus = fcm.graus_termos(valor, t)
            dominante = max(graus, key=graus.get)
            vert = fcm.vertices_termos(t)
            x_enc = eixo_x(t)
            fundo = alt.Chart(vert[vert["termo"] != dominante]).mark_line(
                strokeWidth=1.5, color=CINZA, opacity=0.6).encode(x=x_enc, y=eixo_y, detail="termo:N")
            frente = alt.Chart(vert[vert["termo"] == dominante]).mark_line(
                strokeWidth=5, color=COR_TERMO[dominante]).encode(x=x_enc, y=eixo_y)
            regua = alt.Chart(pd.DataFrame({"x": [valor]})).mark_rule(
                strokeDash=[5, 4], color="#c9ccd1", strokeWidth=1.5).encode(x=x_enc)
            pts = pd.DataFrame({"x": [valor] * 3, "grau": list(graus.values()), "termo": list(graus.keys())})
            pts["cor"] = [COR_TERMO[tm] if tm == dominante else CINZA for tm in pts["termo"]]
            pontos = alt.Chart(pts).mark_point(filled=True, size=90, opacity=1).encode(
                x=x_enc, y=eixo_y, color=alt.Color("cor:N", scale=None),
                tooltip=["termo", alt.Tooltip("grau:Q", format=".2f")])
            titulo = alt.TitleParams(
                f"{fcm.rotulo(col)}: {fcm.formatar_valor(col, valor)} → {dominante} ({fcm.fmt_pct(graus[dominante], 0)})",
                fontSize=13, anchor="start", color=COR_TERMO[dominante])
            with colunas_graf[k % 2]:
                st.altair_chart((fundo + frente + regua + pontos).properties(height=190, title=titulo),
                                width="stretch")

    # ---------- 2. Curvas do FCM ----------
    with st.container(border=True):
        st.subheader("2. As curvas que o Fuzzy C-Means desenhou sozinho")
        st.markdown(
            "Acima, as faixas \"baixa/média/alta\" foram **definidas por nós**. No Fuzzy C-Means ninguém desenha "
            "a curva: ela **nasce da distância até o centro de cada grupo**. Cada gráfico mostra, para cada grupo, "
            "como a pertinência muda conforme aquela variável varia (as demais ficam na média da base). "
            "**O grupo que mais combina com o valor do cliente naquela variável fica grosso; os outros ficam "
            "finos e apagados.**"
        )
        rot_grupo = [f"{nomes_grupo[i]} · {perfis[i]}" for i in range(c)]
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
                f"{fcm.rotulo(col)}: {fcm.formatar_valor(col, valor)} → {nomes_grupo[g_dom]} ({fcm.fmt_pct(no_ponto.max(), 0)})",
                fontSize=13, anchor="start", color=TAB10[g_dom % 8])
            with colunas_graf2[k % 2]:
                st.altair_chart((linhas_c + regua_c).properties(height=190, title=titulo), width="stretch")
