import streamlit as st

st.set_page_config(page_title="Sobre os Modelos Preditivos", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Entendendo os Modelos Preditivos Utilizados")
st.markdown("---")

st.markdown("""
Para prever o preço do petróleo Brent — uma série de dados que muda bastante e depende de vários fatores — usamos dois tipos principais de modelos: modelos estatísticos mais tradicionais e redes neurais modernas (aprendizado profundo). Essa combinação ajuda a entender tanto os padrões mais simples quanto as mudanças mais complexas nos preços.

Os modelos foram treinados com **dados dos últimos 10 anos**, buscando usar informações recentes sem perder um bom volume de dados.
""")

st.subheader("1. Modelos ARIMA, SARIMA e ARIMAX/SARIMAX")
st.markdown("""
Esses são modelos clássicos usados para prever séries temporais (dados organizados por tempo, como preços diários).

* **ARIMA:** Tenta prever o valor de hoje com base nos valores dos dias anteriores. Ele faz três coisas:
    * Entende os valores passados (AR),
    * Remove tendências para facilitar a previsão (I),
    * E ajusta os erros cometidos em previsões anteriores (MA).

* **SARIMA:** É parecido com o ARIMA, mas leva em conta padrões que se repetem, como os dias da semana ou meses do ano (sazonalidade).

* **ARIMAX / SARIMAX:** É o mesmo modelo, mas agora usando também outras informações (como o dia da semana ou o mês) para melhorar a previsão. Isso é importante porque o preço do petróleo costuma seguir certos padrões dependendo da época.

* **Transformações:** Antes de usar esses modelos, os dados passam por ajustes — como usar logaritmo (para suavizar variações muito grandes) e calcular médias móveis (para tirar tendências). Depois que o modelo prevê, esses ajustes são "desfeitos" para mostrar os valores reais.
""")

st.subheader("2. Modelo LSTM Híbrido (Rede Neural)")
st.markdown("""
LSTM é um tipo de rede neural muito bom para lidar com dados ao longo do tempo, como séries temporais. Ele é mais inteligente que os modelos ARIMA quando se trata de capturar relações mais complexas entre os dados.

* **O que tem de especial no LSTM:**
    * Consegue "lembrar" de coisas que aconteceram há muito tempo nos dados,
    * Lida bem com relações complicadas que os modelos antigos não enxergam,
    * Pode usar várias informações ao mesmo tempo (como dia da semana, mês, etc.).
    
"""
            """
* **Por que usamos o LSTM junto com o ARIMAX:**
    1. O modelo ARIMAX nos dá uma boa base, capturando padrões mais simples.
    2. O LSTM então aprende os detalhes mais complicados que o ARIMAX não consegue ver.
    3. Ele também recebe as mesmas informações sazonais (dia da semana, mês, etc.), para entender melhor os ciclos do tempo.
    4. Foi treinado com bastante cuidado, usando uma sequência grande de dias (por exemplo, 60 dias seguidos), técnicas para evitar que ele aprenda "errado", e parando o treinamento na hora certa.

* **Pré-processamento:** Os dados são normalizados (colocados numa escala de 0 a 1) para facilitar o aprendizado da rede neural. Depois da previsão, os resultados são convertidos de volta para os valores reais.

* **Desafios:** O LSTM precisa de mais dados e mais tempo para treinar, além de exigir muitos testes para ajustar seus parâmetros corretamente.
""")

st.markdown("---")
st.markdown("""
**Resumindo:** Usamos o ARIMAX para pegar os padrões mais óbvios e previsíveis, e o LSTM para entender as variações mais complicadas. Juntos, eles formam uma solução poderosa para prever o preço do petróleo Brent com mais precisão.
""")

st.markdown("---")
st.markdown("⬅️ Volte para a página de **Modelos Preditivos** para testar as previsões!")
