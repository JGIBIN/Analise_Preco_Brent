# Análise e Previsão do Preço do Petróleo Brent

![Badge do Projeto](https://img.shields.io/badge/status-conclu%C3%ADdo-green)
![Badge da Linguagem](https://img.shields.io/badge/python-3.10%2B-blue)

Este projeto realiza uma análise de série temporal do preço do petróleo Brent, aplicando e comparando dois modelos de previsão: SARIMA e LSTM (Long Short-Term Memory). O objetivo é explorar os dados históricos e construir modelos capazes de prever cotações futuras.

## 📊 Aplicação Web (Streamlit)

O projeto inclui uma aplicação web interativa construída com Streamlit. Para visualizar a análise e as previsões, execute a aplicação localmente.

*(Se você hospedar a aplicação, pode adicionar o link aqui)*
**Link da Aplicação:** `(insira o link aqui, se houver)`

## 🎯 Modelos Utilizados

Foram treinados e avaliados dois modelos distintos para a previsão dos preços:

1.  **SARIMA (Seasonal Autoregressive Integrated Moving Average):** Um modelo estatístico clássico para análise de séries temporais que captura padrões sazonais e tendências. O modelo treinado está salvo em `sarima_model_sf.pkl`.
2.  **LSTM (Long Short-Term Memory):** Uma rede neural recorrente (RNN) avançada, ideal para aprender padrões de longo prazo em dados sequenciais, como séries temporais. O modelo treinado está salvo em `lstm_model.h5`.

O notebook `notebook_treino.ipynb` contém todo o processo de análise exploratória, preparação de dados e treinamento dos modelos.

## 🛠️ Tecnologias e Bibliotecas

Este projeto foi desenvolvido utilizando as seguintes tecnologias:

* **Python 3.10+**
* **Streamlit:** Para a criação da aplicação web interativa.
* **Pandas:** Para manipulação e análise dos dados.
* **Statsmodels:** Para a implementação do modelo SARIMA.
* **TensorFlow / Keras:** Para a implementação do modelo LSTM.
* **Scikit-learn:** Para pré-processamento de dados (ex: `scaler.pkl`).
* **Jupyter Notebook:** Para experimentação e treinamento dos modelos.

## 🚀 Como Executar o Projeto

Para executar a aplicação localmente, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/JGIBIN/Analise_Preco_Brent.git](https://github.com/JGIBIN/Analise_Preco_Brent.git)
    cd Analise_Preco_Brent
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run Home.py
    ```

A aplicação será aberta automaticamente no seu navegador.

## 📂 Estrutura do Repositório
