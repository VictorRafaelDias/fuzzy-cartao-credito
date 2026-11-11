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

tab_geral, tab_grupos, tab_pert, tab_tabela, tab_interp = st.tabs([
    ":material/dashboard: Visão geral",
    ":material/bubble_chart: Grupos",
    ":material/blur_on: Pertinência",
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
            f"Com m = {m:.1f} e {len(colunas)} variáveis, a pertinência tende a ficar quase igual "
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
    medias_orig = fcm.perfil_original(df, colunas, rotulos).reindex(nomes_grupo)
    media_geral = df[colunas].mean()
    carac = fcm.descrever_grupos(r["centros"], colunas, top=4)

    st.subheader("Perfil de cada grupo")
    for i, nome in enumerate(nomes_grupo):
        with st.container(border=True):
            n_i = int(tamanhos[i])
            nit = int((~fronteira[rotulos == i]).sum())
            n_txt = f"{n_i:,}".replace(",", ".")
            st.markdown(f"**{nome}** · {n_txt} clientes ({fcm.fmt_pct(n_i / len(df))}) · "
                        f"{fcm.fmt_pct(nit / max(n_i, 1), 0)} com pertinência acima de {fcm.fmt_pct(limiar, 0)}")
            for nome_var, z in carac[i]:
                seta = ":green[↑ acima da média]" if z > 0 else ":red[↓ abaixo da média]"
                st.markdown(f"- {nome_var}: {seta} ({z:+.2f} desv.-padrão)")

    st.subheader("Médias por grupo (escala original)")
    tabela_medias = medias_orig.rename(columns=fcm.rotulo).T
    tabela_medias["Média geral"] = media_geral.rename(index=fcm.rotulo)
    st.dataframe(tabela_medias.round(2))

    st.subheader("Fuzzy C-Means × K-Means")
    k = r["kmeans"]
    with st.container(horizontal=True):
        st.metric("Silhouette (fuzzy)", f"{k['silhouette_fcm']:.3f}", border=True)
        st.metric("Silhouette (K-Means)", f"{k['silhouette_kmeans']:.3f}", border=True)
        st.metric("Concordância (ARI)", f"{k['ari']:.3f}", border=True,
                  help="1 = os dois métodos formam exatamente os mesmos grupos.")
    st.caption(
        f"Com m = {m:.1f}, {fcm.fmt_pct(fronteira.mean())} dos clientes ficam na fronteira — "
        "informação que o K-Means não oferece, pois ele atribui cada cliente a um único grupo."
    )
