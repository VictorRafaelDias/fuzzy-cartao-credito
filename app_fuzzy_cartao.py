"""Entrada do app Streamlit: lógica fuzzy aplicada a clientes de cartão de crédito.

Rodar:  python -m streamlit run app_fuzzy_cartao.py

Páginas:
  - apresentacao.py: apresentação passo a passo do score fuzzy por regras (técnica principal)
  - apendice_fcm.py: apêndice com o agrupamento Fuzzy C-Means (técnica complementar)
"""
import streamlit as st

st.set_page_config(
    page_title="Lógica fuzzy · Cartões de crédito",
    page_icon=":material/credit_card:",
    layout="wide",
)

pagina = st.navigation([
    st.Page("apresentacao.py", title="Apresentação", icon=":material/slideshow:", default=True),
    st.Page("apendice_fcm.py", title="Apêndice: agrupamento (Fuzzy C-Means)", icon=":material/bubble_chart:"),
])
pagina.run()
