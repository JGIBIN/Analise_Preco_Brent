import streamlit as st

st.set_page_config(page_title="Sobre o Projeto", page_icon="📄", layout="wide")

st.title("📄 Sobre o Projeto de Consultoria")

st.markdown("""
### Contexto do Desafio
Fomos contratados para uma consultoria, e nosso trabalho envolve analisar os dados de preço do petróleo Brent, que podem ser encontrados no site do IPEA.
Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo para gerar insights relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.

### Objetivos Detalhados
1.  **Criar um Dashboard Interativo:** Utilizar ferramentas à escolha para apresentar os dados e insights de forma clara e interativa.
2.  **Storytelling com Insights Relevantes:** O dashboard deve narrar uma história sobre a variação do preço do petróleo, conectando-a com:
    * Situações geopolíticas.
    * Crises econômicas.
    * Demanda global por energia.
3.  **Modelo de Machine Learning para Forecasting:**
    * Desenvolver modelos (ARIMA e LSTM) para prever o preço do petróleo diariamente.
    * Incluir o código, análise de performance dos modelos e integrar as previsões ao storytelling.
4.  **Plano de Deploy em Produção:**
    * Criar um plano detalhado para colocar o modelo de Machine Learning em um ambiente de produção.
    * Listar as ferramentas e tecnologias necessárias para o deploy.
5.  **MVP do Modelo em Streamlit:**
    * Desenvolver um Produto Mínimo Viável (MVP) da funcionalidade de previsão utilizando la biblioteca Streamlit.

### Fonte e Janela de Dados Utilizada
Os dados históricos do preço do petróleo Brent foram obtidos do [IPEADATA](http://www.ipeadata.gov.br/Default.aspx). A série específica utilizada é o preço do petróleo bruto Brent (FOB) em dólares.

**Justificativa para a Janela de 10 Anos:**
Para esta aplicação, tanto a análise exploratória quanto o treinamento dos modelos preditivos focam nos **últimos 10 anos de dados** disponíveis (até novembro de 2024, conforme os modelos pré-treinados). A escolha por esta janela temporal se deve aos seguintes motivos:
- **Relevância dos Padrões Recentes:** O mercado de petróleo é dinâmico e influenciado por mudanças estruturais ao longo do tempo. Os últimos 10 anos refletem dinâmicas de mercado mais atuais e, teoricamente, mais pertinentes para previsões futuras.
- **Custo Computacional:** Trabalhar com uma janela menor de dados reduz significativamente o tempo de treinamento de modelos como ARIMA e LSTM, tornando a análise e o desenvolvimento mais ágeis.
- **Homogeneidade da Série:** Um período mais curto tem maior probabilidade de apresentar características estatísticas mais homogêneas, facilitando a modelagem da estacionariedade e a identificação de padrões sazonais relevantes para o período.
- **Equilíbrio:** Busca-se um equilíbrio entre ter um volume de dados suficiente para um treinamento robusto e garantir que os dados sejam representativos do comportamento mais atual do mercado, evitando que padrões muito antigos e possivelmente obsoletos influenciem excessivamente o aprendizado dos modelos.

### Abordagem da Consultoria
Nossa abordagem combinou uma análise exploratória detalhada dos últimos 10 anos de dados para extrair insights históricos com a aplicação de técnicas avançadas de modelagem de séries temporais para fornecer previsões acuradas, utilizando os mesmos dados para treinar os modelos.

### Plano para Deploy em Produção do Modelo (Sugestão)
1.  **Preparação e Treinamento:** Coleta automatizada, pré-processamento, treinamento (focado nos últimos 10 anos para relevância), versionamento (MLflow/DVC), salvar artefatos.
2.  **API de Inferência:** FastAPI/Flask, Docker.
3.  **Deploy da API em Cloud:** AWS, GCP, ou Azure.
4.  **Agendamento de Retreinamento:** Airflow, Step Functions, etc. (considerar a estratégia de janela móvel ou expansiva para retreino).
5.  **Deploy do Dashboard Streamlit:** Cloud Run, App Runner, etc.
6.  **Monitoramento e Manutenção:** Ferramentas da cloud, logging, CI/CD.

### Plano de Deploy realizado
1.  **Foi feito o deploy para o Streamlit utilizando a conexão via Github, isto foi feito com vistas da criação do MVP e posteriormente poderá ser implementada melhorias.
""")
