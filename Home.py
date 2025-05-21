import streamlit as st

st.set_page_config(
    page_title="Consultoria Preço do Petróleo",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ... (resto do home.py como fornecido na resposta "Ok, aqui estão os códigos completos...")
st.title("🛢️ Consultoria de Análise e Previsão do Preço do Petróleo Brent")

st.markdown("""
Bem-vindo à nossa consultoria especializada na análise de dados do preço do petróleo Brent.
Este dashboard interativo e os modelos preditivos foram desenvolvidos para fornecer insights
relevantes e auxiliar na tomada de decisão estratégica no volátil mercado de petróleo.

**Dados Utilizados:**
Os modelos são treinados e as análises são baseadas nos dados históricos do preço do petróleo Brent, obtidos do IPEADATA.


**Nossos Objetivos:**
- Criar um dashboard interativo com storytelling e insights relevantes.
- Desenvolver e analisar a performance de modelos de Machine Learning para previsão diária.
- Propor um plano de deploy para o modelo em produção.
- Apresentar um MVP (Minimum Viable Product) da solução.

Utilize o menu na barra lateral para navegar pelas diferentes seções da nossa análise.
""")

st.sidebar.success("Selecione uma página acima.")

st.markdown("---")
st.subheader("Fonte dos Dados:")
st.markdown("Os dados históricos do preço do petróleo Brent foram obtidos do [IPEADATA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view).")
st.markdown("Para esta aplicação, focamos nos **últimos 10 anos de dados** para treinamento e análise.")


st.subheader("Tecnologias Utilizadas:")
st.markdown("""
- **Python 3.10**
- **Pandas & NumPy:** Manipulação e análise de dados.
- **Plotly & Matplotlib/Seaborn:** Visualização de dados.
- **Statsmodels (SARIMA):** Modelagem de séries temporais.
- **TensorFlow/Keras (LSTM):** Modelagem de redes neurais recorrentes.
- **Scikit-learn:** Pré-processamento e métricas de avaliação.
- **Streamlit:** Criação do dashboard interativo e MVP.
- **StatsForecast:** Para o modelo AutoARIMA.
""")

st.markdown("""
---
### 🌐 Projeto disponível no GitHub:
[👉 Acesse aqui o repositório completo](https://github.com/JGIBIN/STREAMLIT_APP.git)
---
""")
