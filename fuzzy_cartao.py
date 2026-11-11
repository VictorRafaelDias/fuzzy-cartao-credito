"""Protótipo de terminal: Fuzzy C-Means na base 'Credit Card Dataset for Clustering'.

Uso:
    python fuzzy_cartao.py                # c automático, m = 1.5
    python fuzzy_cartao.py --c 4 --m 1.5  # c fixo

Saídas em ./resultados/: índices, pertinências, perfil dos grupos, resumo e gráficos.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import fcm

AQUI = Path(__file__).parent
ARQUIVO = AQUI / "data" / "CC_GENERAL.csv"
SAIDA = AQUI / "resultados"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--m", type=float, default=1.5, help="fuzzificador (padrão 1.5)")
    ap.add_argument("--c", type=int, default=None, help="nº de grupos (padrão: automático)")
    ap.add_argument("--n-init", type=int, default=5, help="nº de inicializações")
    ap.add_argument("--limiar", type=float, default=0.6, help="limiar de fronteira")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    SAIDA.mkdir(exist_ok=True)
    df = pd.read_csv(ARQUIVO)
    colunas = [c for c in df.columns if c != "CUST_ID"]
    X = fcm.preparar_dados(df, colunas)
    print(f"{len(df)} clientes, {len(colunas)} variáveis")

    tabela = fcm.varredura(X, args.m, range(2, 9), args.n_init, args.seed)
    c_sug = fcm.sugerir_c(tabela)
    c = args.c or c_sug
    print(tabela.round(3).to_string(index=False))
    print(f"\nc sugerido: {c_sug} | c usado: {c}")
    tabela.to_csv(SAIDA / "indices_validacao.csv", index=False)

    res = fcm.fcm_multi(X, c, args.m, args.n_init, args.seed)
    rot = res.rotulos
    pert_max = res.U.max(axis=1)
    fronteira = pert_max < args.limiar
    km = fcm.comparar_kmeans(X, c, rot, args.seed)

    saida = pd.DataFrame(res.U, columns=[f"pert_grupo_{i + 1}" for i in range(c)])
    saida.insert(0, "CUST_ID", df["CUST_ID"])
    saida.insert(1, "grupo", rot + 1)
    saida["pert_max"] = pert_max
    saida["fronteira"] = fronteira
    saida.to_csv(SAIDA / "pertinencias.csv", index=False)

    perfil = fcm.perfil_original(df, colunas, rot)
    perfil.insert(0, "n_clientes", np.bincount(rot, minlength=c))
    perfil.round(2).to_csv(SAIDA / "perfil_grupos.csv")

    linhas = [
        f"Fuzzy C-Means | m={args.m} | c={c} | n_init={args.n_init} | seed={args.seed}",
        f"Iterações: {res.n_iter} | função objetivo J = {res.J:.1f}",
        f"Tamanho dos grupos (pertinência máxima): {np.bincount(rot, minlength=c).tolist()}",
        f"Clientes de fronteira (pertinência máx < {args.limiar:.0%}): {fronteira.sum()} ({fronteira.mean():.1%})",
        f"Silhouette FCM = {km['silhouette_fcm']:.3f} | K-Means = {km['silhouette_kmeans']:.3f} | ARI(FCM, K-Means) = {km['ari']:.3f}",
        "",
        "Características de cada grupo (desvios-padrão em relação à média geral):",
    ]
    for i, car in enumerate(fcm.descrever_grupos(res.centros, colunas)):
        linhas.append(f"  Grupo {i + 1}: " + ", ".join(f"{n} ({z:+.2f})" for n, z in car))
    (SAIDA / "resumo.txt").write_text("\n".join(linhas), encoding="utf-8")
    print("\n" + "\n".join(linhas))

    # ---- gráficos ----
    fig, eixos = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, col in zip(eixos, ["PC", "PE", "XB", "Silhouette"]):
        ax.plot(tabela["c"], tabela[col], marker="o")
        ax.axvline(c, color="red", ls="--", lw=1)
        ax.set_title(col)
        ax.set_xlabel("nº de grupos")
    fig.tight_layout()
    fig.savefig(SAIDA / "indices_validacao.png", dpi=130)
    plt.close(fig)

    coords = PCA(2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=rot, s=6, alpha=0.6, cmap="tab10")
    ax.scatter(*coords[fronteira].T, s=14, facecolors="none", edgecolors="black", lw=0.3, label="fronteira")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Grupos fuzzy (c={c}, m={args.m}) em PCA")
    ax.legend(*sc.legend_elements(), title="grupo", loc="best")
    fig.tight_layout()
    fig.savefig(SAIDA / "grupos_pca.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pert_max, bins=30)
    ax.axvline(args.limiar, color="red", ls="--")
    ax.set(xlabel="pertinência máxima", ylabel="clientes", title="Nitidez da pertinência")
    fig.tight_layout()
    fig.savefig(SAIDA / "nitidez_pertinencia.png", dpi=130)
    plt.close(fig)
    print(f"\nArquivos salvos em {SAIDA}")


if __name__ == "__main__":
    main()
