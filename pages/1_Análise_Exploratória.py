import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_historical_data
import locale

st.set_page_config(page_title="Análise de Negócios Brent", page_icon="💰", layout="wide")

# Tentar configurar o locale
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, '')
        st.caption("Aviso: Locale pt_BR não disponível. Nomes de meses e formatação podem estar em inglês.")
    except locale.Error:
        st.caption("Aviso: Locales não configurados. Nomes de meses e formatação podem estar inconsistentes.")

# Carregar dados
df_historical_10a = load_historical_data()
if df_historical_10a.empty:
    st.error("Erro: Não foi possível carregar os dados históricos.")
    st.stop()

df_for_analysis = df_historical_10a.set_index('Data')['Value'].copy()

# --- Título e Introdução ---
st.title("💰 Análise de Negócios para Compradores de Petróleo Brent (2014-2024)")
st.markdown("""
Esta análise explora os dados de preço do petróleo Brent dos últimos 10 anos, focando em fornecer insights práticos para compradores.
Entenderemos como as flutuações de preço impactam decisões de compra, gestão de risco e planejamento financeiro.
""")

st.info(f"Nossa jornada temporal abrange o período de **{df_historical_10a['Data'].min().strftime('%d/%m/%Y')}** até **{df_historical_10a['Data'].max().strftime('%d/%m/%Y')}**.")

# --- Seção 1: Preço do Brent e Tendências ---
st.header("📈 Preço Histórico, Tendências e Eventos Chave")
st.markdown("""
O gráfico abaixo mostra a evolução do preço do Brent, com médias móveis para identificar tendências e anotações para eventos importantes.
""")

df_plot = df_historical_10a.copy()
df_plot['MA50'] = df_plot['Value'].rolling(window=50).mean()
df_plot['MA200'] = df_plot['Value'].rolling(window=200).mean()

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['Value'], mode='lines', name='Preço Brent', line=dict(color='deepskyblue', width=2)))
fig_price.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['MA50'], mode='lines', name='Média Móvel (50d)', line=dict(color='orange', width=1.5, dash='dot')))
fig_price.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['MA200'], mode='lines', name='Média Móvel (200d)', line=dict(color='crimson', width=1.5, dash='dash')))

eventos = [
    {'Data': '2014-11-27', 'descricao': 'OPEP mantém produção, preços caem', 'color': 'white', 'ay_offset': -40},
    {'Data': '2016-01-20', 'descricao': 'Preço atinge mínima da década', 'color': 'white', 'ay_offset': -70},
    {'Data': '2020-03-11', 'descricao': 'Pandemia COVID-19 declarada', 'color': 'white', 'ay_offset': -100},
    {'Data': '2020-04-20', 'descricao': 'WTI Negativo (Impacto Brent)', 'color': 'white', 'ay_offset': -130},
    {'Data': '2022-02-24', 'descricao': 'Início da Guerra na Ucrânia', 'color': 'white', 'ay_offset': -160}
]
eventos_filtrados_plot = [e for e in eventos if pd.to_datetime(e['Data']) >= df_historical_10a['Data'].min() and pd.to_datetime(e['Data']) <= df_historical_10a['Data'].max()]
annotations_list = []
shapes_list = []
max_y_plot = df_historical_10a['Value'].max() if not df_historical_10a.empty else 150
min_y_plot = df_historical_10a['Value'].min() if not df_historical_10a.empty else 0
for i, evento in enumerate(eventos_filtrados_plot):
    event_date = pd.to_datetime(evento['Data'])
    shapes_list.append({'type': 'line', 'x0': event_date, 'y0': 0, 'x1': event_date, 'y1': 1, 'xref': 'x', 'yref': 'paper',
                        'line': {'color': evento['color'], 'width': 1.5, 'dash': 'dashdot'}})
    annotations_list.append({'x': event_date, 'y': max_y_plot * (1.05 + i * 0.05), 'xref': 'x', 'yref': 'y',
                             'text': f"<b>{evento['descricao']}</b><br>({event_date.strftime('%b %Y')})", 'showarrow': True,
                             'arrowhead': 2, 'arrowwidth': 1.5, 'arrowcolor': evento['color'], 'ax': 0, 'ay': evento['ay_offset'],
                             'font': {'color': 'black', 'size': 10}, 'bgcolor': evento['color'], 'opacity': 0.7,
                             'bordercolor': 'black', 'borderwidth': 1, 'borderpad': 2})

fig_price.update_layout(
    title='Preço do Brent, Médias Móveis e Eventos Chave (2014-2024)',
    xaxis_title='Data', yaxis_title='Preço (US$)', template='plotly_white', height=700,
    shapes=shapes_list, annotations=annotations_list,
    yaxis_range=[min_y_plot * 0.85, max_y_plot * 1.45],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_price, use_container_width=True)

st.markdown("""
**Implicações para Compradores:**
- **Tendências de Alta:** Indicam aumento dos custos futuros, necessidade de garantir suprimentos ou usar instrumentos de hedge.
- **Tendências de Baixa:** Podem oferecer oportunidades de compra, mas exigem cautela devido à volatilidade.
- **Média Móvel:** Ajuda a suavizar flutuações de curto prazo e identificar a direção geral do mercado.
- **Eventos Chave:** Compreender o impacto de eventos geopolíticos e econômicos é crucial para antecipar movimentos de preço.
""")

# --- Seção 2: Decifrando os Eventos e Seus Impactos ---
st.markdown("---")
st.header("🗓️ Decifrando os Eventos e Seus Impactos")

insight_cols = st.columns(2)
with insight_cols[0]:
    st.subheader("Insight 1: OPEP e Guerra de Preços (2014-2016)")
    st.markdown("A decisão da OPEP em 2014 de manter a produção causou uma queda nos preços, mostrando o poder da OPEP.")

    st.subheader("Insight 2: Choque da COVID-19 (2020)")
    st.markdown("A pandemia em 2020 reduziu drasticamente a demanda por petróleo, levando a uma queda acentuada nos preços.")

with insight_cols[1]:
    st.subheader("Insight 3: Guerra na Ucrânia (2022)")
    st.markdown("A guerra na Ucrânia em 2022 aumentou os preços devido a preocupações com o fornecimento.")

    st.subheader("Insight 4: Recuperação Pós-Pandemia")
    st.markdown("A recuperação econômica pós-pandemia e as ações da OPEP+ influenciaram os preços, com preocupações sobre inflação adicionando volatilidade.")

st.subheader("Insight Adicional: Volatilidade")
st.markdown("A volatilidade é uma característica constante do mercado de petróleo, exigindo gerenciamento de risco.")

# --- Seção 3: Análise Detalhada ---
st.markdown("---")
st.header("🔍 Análise Detalhada: Decomposição, Distribuição e Volatilidade")

# --- 3.1 Decomposição da Série Temporal ---
st.subheader("Decompondo a Série: Tendência, Sazonalidade e Resíduos")
st.markdown("A decomposição separa a série em tendência, sazonalidade e resíduos.")

decomp_period = 252
if len(df_for_analysis.dropna()) >= decomp_period * 2:
    try:
        decomposition = seasonal_decompose(df_for_analysis.dropna(), model='additive', period=decomp_period)

        fig_decomp, axes_decomp = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        decomposition.trend.plot(ax=axes_decomp[0], legend=False, ylabel='Tendência')
        decomposition.seasonal.plot(ax=axes_decomp[1], legend=False, ylabel=f'Sazonalidade ({decomp_period}d)')
        decomposition.resid.plot(ax=axes_decomp[2], legend=False, linestyle=':', ylabel='Resíduo')
        df_for_analysis.plot(ax=axes_decomp[3], legend=False, ylabel='Original')  # Adicionado original
        fig_decomp.suptitle('Decomposição da Série Temporal do Preço do Brent', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_decomp)
        st.markdown("* **Tendência:** Direção geral. * **Sazonalidade:** Padrões recorrentes. * **Resíduo:** Flutuações restantes.")

    except Exception as e:
        st.warning(f"Não foi possível realizar a decomposição sazonal: {e}.")
else:
    st.warning(f"Dados insuficientes para decomposição sazonal com período {decomp_period}.")

# --- 3.2 Distribuição dos Preços e Retornos ---
st.subheader("Distribuição dos Preços e Retornos")
df_returns = df_historical_10a.copy()
df_returns['Retorno Diário'] = df_returns['Value'].pct_change()
df_returns.dropna(inplace=True)

col_dist_preco, col_dist_ret = st.columns(2)
with col_dist_preco:
    fig_hist_price = px.histogram(df_historical_10a, x="Value", nbins=50, title="Distribuição dos Preços")
    fig_hist_price.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist_price, use_container_width=True)
    st.markdown("O histograma mostra a frequência dos diferentes níveis de preço.")

with col_dist_ret:
    if not df_returns.empty:
        fig_hist_returns = px.histogram(df_returns, x="Retorno Diário", nbins=100, title="Distribuição dos Retornos Diários")
        fig_hist_returns.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist_returns, use_container_width=True)
        st.markdown("A distribuição dos retornos frequentemente apresenta 'caudas pesadas' (mais eventos extremos).")
        st.write(f"**Curtose dos Retornos:** {df_returns['Retorno Diário'].kurtosis():.2f}")
        st.write(f"**Assimetria dos Retornos:** {df_returns['Retorno Diário'].skew():.2f}")
    else:
        st.warning("Não foi possível gerar a distribuição dos retornos.")

# --- 3.3 Volatilidade ao Longo do Tempo ---
st.subheader("Volatilidade Móvel")
st.markdown("Calculamos a volatilidade móvel (30 dias, anualizada) para visualizar períodos de turbulência.")
if not df_returns.empty:
    df_historical_10a['Volatilidade 30d'] = df_returns['Retorno Diário'].rolling(window=30).std() * np.sqrt(252)
    fig_volatility = px.line(df_historical_10a.dropna(subset=['Volatilidade 30d']),
                             x='Data', y='Volatilidade 30d',
                             title='Volatilidade Móvel de 30 Dias (Anualizada)',
                             labels={'Volatilidade 30d': 'Volatilidade', 'Data': 'Data'})
    fig_volatility.update_layout(yaxis_title='Volatilidade Anualizada', template='plotly_white')
    st.plotly_chart(fig_volatility, use_container_width=True)
    st.markdown("Picos indicam alta incerteza.")
else:
    st.warning("Não foi possível calcular a volatilidade móvel.")

# --- Seção 4: Análise de Autocorrelação e Estacionariedade ---
st.markdown("---")
st.header("🔍 Análise de Autocorrelação e Estacionariedade")

# --- 4.1 Teste ADF ---
st.subheader("Teste de Estacionariedade (ADF)")
if not df_for_analysis.dropna().empty:
    adf_stat_orig, p_value_orig, _, _, critical_values_orig, _ = adfuller(df_for_analysis.dropna())
    st.write(f"**Série de Preços Original ('Value'):**")
    st.write(f"  - Estatística ADF: {adf_stat_orig:.4f}")
    st.write(f"  - P-valor: {p_value_orig:.4f}")
    st.write(f"  - É estacionária? {'Sim' if p_value_orig <= 0.05 else 'Não'}")
    st.caption(f"Valores Críticos: 1%: {critical_values_orig['1%']:.2f}, 5%: {critical_values_orig['5%']:.2f}, 10%: {critical_values_orig['10%']:.2f}")

    log_series = np.log(df_for_analysis.dropna().replace(0, 1e-5))
    diff_log_series = log_series.diff().dropna()
    if not diff_log_series.empty:
        adf_stat_trans, p_value_trans, _, _, critical_values_trans, _ = adfuller(diff_log_series)
        st.write(f"**Série Log-Diferenciada:**")
        st.write(f"  - Estatística ADF: {adf_stat_trans:.4f}")
        st.write(f"  - P-valor: {p_value_trans:.4f}")
        st.write(f"  - É estacionária? {'Sim' if p_value_trans <= 0.05 else 'Não'}")
    st.markdown("Um p-valor baixo sugere que a série é estacionária após transformações.")
else:
    st.warning("Não foi possível realizar o teste ADF.")

# --- 4.2 ACF e PACF ---
st.subheader("Autocorrelação (ACF e PACF)")
col1_acf, col2_pacf = st.columns(2)
if not df_for_analysis.dropna().empty:
    with col1_acf:
        fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
        plot_acf(df_for_analysis.dropna(), lags=40, ax=ax_acf)
        ax_acf.set_title("Autocorrelação (ACF) - Preços Originais")
        st.pyplot(fig_acf)
        st.markdown("Decaimento lento no ACF sugere não-estacionariedade.")

    with col2_pacf:
        fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
        plot_pacf(df_for_analysis.dropna(), lags=40, ax=ax_pacf, method='ywm')
        ax_pacf.set_title("Autocorrelação Parcial (PACF)")
