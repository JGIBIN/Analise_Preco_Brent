import streamlit as st


st.set_page_config(page_title="Sobre os Modelos Preditivos", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Entendendo os Modelos Preditivos Utilizados")
st.markdown("---")

st.markdown("""
Para prever o preço do petróleo Brent, uma série temporal notoriamente volátil e influenciada por diversos fatores, utilizamos uma combinação de modelos estatísticos clássicos e técnicas de aprendizado profundo. A escolha desses modelos visa capturar diferentes aspectos da dinâmica dos preços, desde padrões lineares e sazonais até relações mais complexas e dependências de longo prazo.

Nesta aplicação, os modelos foram treinados utilizando os **últimos 10 anos de dados históricos**, buscando um equilíbrio entre relevância dos padrões recentes e volume de dados suficiente para aprendizado.
""")

st.subheader("1. Modelos ARIMA, SARIMA e ARIMAX/SARIMAX")
st.markdown("""
Os modelos da família ARIMA são amplamente utilizados para análise e previsão de séries temporais.

* **ARIMA (AutoRegressive Integrated Moving Average):**
    * **AR (Autoregressivo):** Assume que o valor atual da série depende de seus valores passados.
    * **I (Integrado):** Envolve diferenciar a série temporal para torná-la estacionária (ou seja, remover tendências e variações na variância que mudam ao longo do tempo).
    * **MA (Média Móvel):** Modela o erro da previsão como uma combinação linear dos erros de previsão passados.
    * **Pontos Fortes:** São bons para capturar estruturas de autocorrelação lineares nos dados. São relativamente interpretáveis.

* **SARIMA (Seasonal ARIMA):**
    * É uma extensão do ARIMA que inclui componentes sazonais. Adiciona os parâmetros `(P, D, Q, S)` para modelar padrões que se repetem em um período fixo `S` (ex: S=7 para semanal, S=12 para mensal em dados mensais).
    * No nosso projeto, ao usar o `AutoARIMA` da biblioteca `statsforecast`, especificamos um `season_length` (como `ARIMAX_SEASON_LENGTH = 7` ou `14` no seu notebook). Isso permite que o `AutoARIMA` tente ajustar um modelo SARIMA, encontrando automaticamente as melhores ordens `P, D, Q` para essa sazonalidade, além das ordens não sazonais `p, d, q`.

* **ARIMAX / SARIMAX (com Variáveis eXógenas):**
    * O "X" no final significa que o modelo pode incorporar **variáveis exógenas**, ou seja, outras séries temporais ou features que podem influenciar a série que estamos tentando prever (o preço do petróleo).
    * **Por que usamos (ARIMAX):** No nosso caso, ao treinar o `AutoARIMA` no notebook, nós fornecemos features sazonais explícitas (como `dia_da_semana`, representações cíclicas de `mês` e `dia_do_ano`) como colunas adicionais no DataFrame de treinamento. O `AutoARIMA` da `statsforecast` (versão >= 1.0.0) detecta automaticamente essas colunas extras e as utiliza como variáveis exógenas. Assim, o modelo treinado e salvo como `sarima_model_sf.pkl` é, na verdade, um **ARIMAX** (ou **SARIMAX** se o `AutoARIMA` também encontrou componentes sazonais internos `P,D,Q` significativos).
    * **Benefício:** Incluir essas features sazonais como exógenas pode ajudar o modelo a capturar padrões sazonais de forma mais direta e potencialmente melhorar a precisão das previsões, complementando a capacidade do componente SARIMA interno.

* **Transformações:** Para os modelos ARIMA/ARIMAX, frequentemente aplicamos transformações como a **logarítmica** (para estabilizar a variância) e **diferenciação** (para tornar a série estacionária). No nosso caso, a pipeline de pré-processamento para o ARIMAX no notebook (`SomthDataIntervalValues`) realiza uma transformação logarítmica e uma diferenciação baseada em média móvel. As previsões precisam ser revertidas para a escala original, o que é feito na aplicação Streamlit.
""")

st.subheader("2. Modelo LSTM Híbrido (Long Short-Term Memory)")
st.markdown("""
LSTM é um tipo avançado de Rede Neural Recorrente (RNN) particularmente eficaz para aprender dependências de longo prazo em dados sequenciais, como séries temporais.

* **Pontos Fortes do LSTM:**
    * **Memória de Longo Prazo:** LSTMs possuem "portões" (gates) em sua arquitetura que lhes permitem lembrar informações por longos períodos e esquecer informações irrelevantes, superando o problema do "desvanecimento do gradiente" de RNNs mais simples.
    * **Relações Não Lineares:** São capazes de modelar relações complexas e não lineares nos dados, que modelos lineares como ARIMA podem não capturar.
    * **Flexibilidade com Múltiplas Features (Multivariado):** LSTMs podem facilmente incorporar múltiplas séries de entrada (features) para fazer previsões.

* **Por que usamos (LSTM Híbrido com Features Sazonais):**
    1.  **Capturar Dinâmicas Complexas:** O preço do petróleo é influenciado por muitos fatores interconectados de forma não linear. Um LSTM tem potencial para aprender esses padrões.
    2.  **Abordagem Híbrida:** No nosso notebook, o LSTM foi treinado de forma "híbrida". A série de entrada para o LSTM (`df_combinado_para_lstm_final['y']` no notebook) foi construída usando o histórico de preços reais complementado/substituído pelas previsões do modelo ARIMAX (as "features ARIMAX"). A ideia é que o ARIMAX capture bem os componentes lineares e sazonais mais estruturados, e o LSTM aprenda a modelar os resíduos ou as dinâmicas não lineares restantes, ou a refinar as previsões do ARIMAX.
    3.  **Features Sazonais Explícitas:** Assim como no ARIMAX, fornecemos ao LSTM as mesmas 5 features sazonais (`dia_da_semana`, `mes_sin`, `mes_cos`, `dia_do_ano_sin`, `dia_do_ano_cos`). Isso ajuda o LSTM a "entender" explicitamente os ciclos temporais. O `scaler.pkl` foi treinado nessas 6 features (a série 'y' combinada + 5 sazonais).
    4.  **Melhorias no Treinamento:** Para o LSTM, utilizamos um `seq_length` maior (ex: 60 dias) para dar mais contexto, uma arquitetura com mais unidades e camadas de `Dropout` (para reduzir overfitting), e `EarlyStopping` durante o treinamento para encontrar os melhores pesos e evitar treinar demais.

* **Pré-processamento:** Os dados de entrada para o LSTM são normalizados (geralmente para a escala 0-1 usando `MinMaxScaler`) porque redes neurais geralmente performam melhor com dados de entrada escalonados. As previsões do LSTM são então desnormalizadas para a escala original.

* **Desafios:** LSTMs geralmente requerem mais dados para treinar do que modelos ARIMA, podem ser mais lentos para treinar e exigem mais experimentação com hiperparâmetros (número de camadas, unidades, `seq_length`, taxa de aprendizado, etc.).
""")

st.markdown("---")
st.markdown("""
**Em resumo:** A combinação do ARIMAX (para capturar estrutura linear, sazonalidade e o impacto de features sazonais explícitas) e do LSTM Híbrido (para aprender padrões não lineares e dependências de longo prazo, também auxiliado por features sazonais e pela "visão" do ARIMAX) visa fornecer um conjunto robusto de ferramentas para a previsão do preço do petróleo Brent.
""")

st.markdown("---")
st.markdown("⬅️ Volte para a página de **Modelos Preditivos** para testar as previsões!")

