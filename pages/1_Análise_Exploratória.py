import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_historical_data
import locale  # Para formatação de datas (tentativa)

st.set_page_config(page_title="Análise de Negócios Brent", page_icon="💰", layout="wide")

# Tentar configurar o locale para português do Brasil (para nomes de meses, etc.)
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, '')  # Tenta usar o locale padrão do sistema
        st.caption("Aviso: Locale pt_BR não disponível. Nomes de meses e formatação podem estar em inglês.")
    except locale.Error:
        st.caption("Aviso: Locales não configurados. Nomes de meses e formatação podem estar inconsistentes.")

# Carregar dados
df_historical_10a = load_historical_data()
if df_historical_10a.empty:
    st.error("Erro: Não foi possível carregar os dados históricos.")
    st.stop()

df_for_analysis = df_historical_10a.set_index('Data')['Value'].copy()  # Para statsmodels

# --- Título e Introdução ---
st.title("💰 Análise de Negócios para Compradores de Petróleo Brent (2014-2024)")
st.markdown("""
Esta análise explora os dados de preço do petróleo Brent dos últimos 10 anos, focando em fornecer insights práticos para compradores.
Entenderemos como as flutuações de preço impactam decisões de compra, gestão de risco e planejamento financeiro.
""")

# --- Seção 1: Preço do Brent e Tendências ---
st.header("📈 Preço Histórico e Tendências")
st.markdown("""
O gráfico abaixo mostra a evolução do preço do Brent, com médias móveis para identificar tendências de curto (50 dias) e longo prazo (200 dias).
""")

df_plot = df_historical_10a.copy()
df_plot['MA50'] = df_plot['Value'].rolling(window=50).mean()
df_plot['MA200'] = df_plot['Value'].rolling(window=200).mean()

fig_price = px.line(df_plot, x='Data', y='Value', title='Preço do Petróleo Brent (2014-2024)',
                  labels={'Value': 'Preço (USD)', 'Data': 'Data'})
fig_price.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['MA50'], mode='lines', name='Média Móvel (50d)'))
fig_price.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['MA200'], mode='lines', name='Média Móvel (200d)'))
fig_price.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)) # Ajuste da legenda
st.plotly_chart(fig_price, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Tendências de Alta:** Indicam aumento dos custos futuros, necessidade de garantir suprimentos ou usar instrumentos de hedge.
- **Tendências de Baixa:** Podem oferecer oportunidades de compra, mas exigem cautela devido à volatilidade.
- **Média Móvel:** Ajuda a suavizar flutuações de curto prazo e identificar a direção geral do mercado.
""")

# --- Seção 2: Volatilidade e Risco ---
st.header("⚠️ Volatilidade e Gestão de Risco")
st.markdown("""
A volatilidade do preço do Brent é um fator crucial para compradores. Ela representa o grau de incerteza e o potencial de grandes flutuações de preço em curtos períodos.
""")

df_volatility = df_historical_10a.copy()
df_volatility['Retorno Diário'] = df_volatility['Value'].pct_change()
df_volatility['Volatilidade 30d'] = df_volatility['Retorno Diário'].rolling(window=30).std() * np.sqrt(252) # Anualizada
df_volatility.dropna(inplace=True)

fig_volatility = px.line(df_volatility, x='Data', y='Volatilidade 30d',
                      title='Volatilidade Anualizada (30 Dias)',
                      labels={'Volatilidade 30d': 'Volatilidade', 'Data': 'Data'})
st.plotly_chart(fig_volatility, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Alta Volatilidade:** Aumenta o risco financeiro, dificulta o planejamento de custos e exige estratégias de hedge (futuros, opções).
- **Baixa Volatilidade:** Pode sugerir um período de maior estabilidade, mas não elimina a possibilidade de mudanças bruscas.
""")

# --- Seção 3: Sazonalidade ---
st.header("📅 Sazonalidade")
st.markdown("""
A sazonalidade refere-se a padrões de preço que se repetem em intervalos regulares (por exemplo, ao longo do ano).
""")

df_seasonal = df_historical_10a.copy()
df_seasonal['Mês'] = df_seasonal['Data'].dt.month_name()  # Obtém o nome do mês
df_seasonal['Ano'] = df_seasonal['Data'].dt.year

# Tentar obter nomes dos meses em português, se o locale estiver configurado
try:
    df_seasonal['Mês_num'] = df_seasonal['Data'].dt.month  # Numérico para ordenar
    df_seasonal['Mês_nome'] = df_seasonal['Mês'].apply(lambda x: pd.Timestamp(2023, 1, 1).month_name())
    months_order = list(df_seasonal.groupby('Mês_num')['Mês_nome'].max().sort_index())
except (AttributeError, ValueError): # Se locale não estiver configurado corretamente
    months_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]

fig_seasonal = px.box(df_seasonal, x='Mês', y='Value',
                     category_orders={'Mês': months_order},
                     title='Variação Sazonal do Preço do Brent',
                     labels={'Value': 'Preço (USD)', 'Mês': 'Mês'})
st.plotly_chart(fig_seasonal, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Picos Sazonais:** Identificar meses de alta demanda (inverno no hemisfério norte) para planejar compras ou negociar contratos.
- **Vales Sazonais:** Considerar a possibilidade de aumentar estoques ou obter melhores preços em períodos de baixa demanda.
""")

# --- Seção 4: Retornos Cumulativos ---
st.header("📈 Retornos Cumulativos")
st.markdown("""
Retornos cumulativos mostram o crescimento do preço do Brent ao longo do tempo, assumindo um investimento inicial.
""")

df_returns = df_historical_10a.copy()
df_returns['Retorno Diário'] = df_returns['Value'].pct_change()
df_returns['Retorno Cumulativo'] = (1 + df_returns['Retorno Diário']).cumprod()
df_returns.dropna(inplace=True)

fig_returns = px.line(df_returns, x='Data', y='Retorno Cumulativo',
                     title='Retornos Cumulativos do Preço do Brent',
                     labels={'Retorno Cumulativo': 'Retorno Cumulativo', 'Data': 'Data'})
st.plotly_chart(fig_returns, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Tendência Geral:** Avaliar o desempenho do Brent como um ativo de longo prazo.
- **Períodos de Crescimento/Declínio:** Identificar momentos para ajustar estratégias de compra ou alocação de recursos.
""")

# --- Seção 5: Distribuição dos Preços ---
st.header("📊 Distribuição dos Preços")
st.markdown("""
A distribuição dos preços mostra a frequência com que diferentes níveis de preço ocorrem.
""")

fig_hist = px.histogram(df_historical_10a, x='Value', nbins=30,
                    title='Distribuição dos Preços do Brent',
                    labels={'Value': 'Preço (USD)'})
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Preço Médio:** Usado como referência para avaliar se os preços atuais estão altos ou baixos.
- **Faixa de Preço:** Identificar os limites dentro dos quais o preço normalmente flutua.
- **Assimetria:** Se a distribuição for assimétrica, pode indicar uma tendência de o preço subir ou descer mais rapidamente.
""")

# --- Seção 6: Análise de Correlação (Opcional - Requer Dados Adicionais) ---
# st.header("🤝 Análise de Correlação (Opcional)")
# st.markdown("""
# A correlação mede a relação entre o preço do Brent e outros ativos ou indicadores econômicos.
# """)
#
# # Exemplo: Se você tiver dados de ações de energia (energy_stocks)
# # df_correlation = pd.merge(df_historical_10a[['Data', 'Value']], energy_stocks, on='Data', how='inner')
# # correlation_matrix = df_correlation.corr()
# # fig_correlation = px.imshow(correlation_matrix, text_auto=True, title="Matriz de Correlação")
# # st.plotly_chart(fig_correlation)
#
# st.markdown("""
# **Implicações para Compradores (Exemplo):**
# - **Correlação Positiva com Ações de Energia:** Sugere que os preços do Brent e as ações se movem na mesma direção.
# - **Correlação Negativa com o Dólar:** Pode indicar que um dólar forte torna o Brent mais caro para compradores com outras moedas.
# """)

# --- Seção 7: Conclusões ---
st.header("💡 Conclusões e Recomendações")
st.markdown("""
A análise dos dados de preço do Brent revela um mercado volátil e influenciado por diversos fatores.
Compradores podem usar esses insights para:

- **Planejar Orçamentos:** Considerar a volatilidade e as tendências para prever custos futuros.
- **Gerenciar Riscos:** Utilizar instrumentos de hedge para se proteger contra flutuações de preço.
- **Otimizar Compras:** Identificar padrões sazonais e momentos oportunos para adquirir petróleo.
- **Tomar Decisões Informadas:** Basear as estratégias de compra em dados e análises em vez de intuição.
""")
