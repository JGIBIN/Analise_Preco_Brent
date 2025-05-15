import streamlit as st
import pandas as pd
import numpy as np # Adicionado para log e outras operações
import plotly.graph_objects as go
import plotly.express as px 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller # Para o teste ADF
import matplotlib.pyplot as plt
import seaborn as sns # Para o heatmap
from utils import load_historical_data 

st.set_page_config(page_title="Análise Exploratória", page_icon="📊", layout="wide")

st.title("📊 Desvendando a Dinâmica do Preço do Petróleo Brent (Últimos 10 Anos)")
st.markdown("""
O mercado de petróleo Brent é um palco global onde tensões geopolíticas, decisões econômicas e a incessante busca por energia se entrelaçam.
Nesta análise, mergulharemos nos últimos 10 anos de dados para entender os principais fatores que moldaram o valor deste crucial recurso energético.
""")

# Carregar dados (utils.py agora filtra para os últimos 10 anos por padrão)
df_historical_10a = load_historical_data() 

if df_historical_10a.empty:
    st.error("Não foi possível carregar os dados históricos dos últimos 10 anos.")
    st.stop()
    
st.info(f"Nossa jornada temporal abrange o período de **{df_historical_10a['Data'].min().strftime('%d/%m/%Y')}** até **{df_historical_10a['Data'].max().strftime('%d/%m/%Y')}**.")
    
df_for_analysis_stats = df_historical_10a.set_index('Data')['Value'].copy() # Para statsmodels

# --- Seção 1: Visualizando a Montanha-Russa dos Preços ---
st.header("🎢 A Montanha-Russa dos Preços: Uma Década em Perspectiva")
st.markdown("""
O gráfico abaixo é a nossa janela para o passado recente. Observe as subidas íngremes, as quedas abruptas e os períodos de relativa
calmaria. Cada movimento conta uma história. Adicionamos médias móveis para ajudar a identificar tendências de curto (50 dias) e longo prazo (200 dias).
""")

df_historical_10a['MA50'] = df_historical_10a['Value'].rolling(window=50).mean()
df_historical_10a['MA200'] = df_historical_10a['Value'].rolling(window=200).mean()

fig_hist_ma = go.Figure()
fig_hist_ma.add_trace(go.Scatter(x=df_historical_10a['Data'], y=df_historical_10a['Value'],
                               mode='lines', name='Preço Brent', line=dict(color='deepskyblue', width=2))) # Cor alterada
fig_hist_ma.add_trace(go.Scatter(x=df_historical_10a['Data'], y=df_historical_10a['MA50'],
                               mode='lines', name='Média Móvel 50 Dias', line=dict(color='orange', width=1.5, dash='dot')))
fig_hist_ma.add_trace(go.Scatter(x=df_historical_10a['Data'], y=df_historical_10a['MA200'],
                               mode='lines', name='Média Móvel 200 Dias', line=dict(color='crimson', width=1.5, dash='dash'))) # Cor alterada
eventos = [
    {'Data': '2014-11-27', 'descricao': 'OPEP mantém produção, preços caem', 'color': 'white', 'ay_offset': -40},
    {'Data': '2016-01-20', 'descricao': 'Preço atinge mínima da década (pós-2014)', 'color': 'white', 'ay_offset': -70}, # Cor alterada
    {'Data': '2020-03-11', 'descricao': 'Pandemia COVID-19 declarada', 'color': 'white', 'ay_offset': -100},
    {'Data': '2020-04-20', 'descricao': 'WTI Negativo (Impacto Brent)', 'color': 'white', 'ay_offset': -130},
    {'Data': '2022-02-24', 'descricao': 'Início da Guerra na Ucrânia', 'color': 'white', 'ay_offset': -160}
]
eventos_filtrados_plot_ma = [e for e in eventos if pd.to_datetime(e['Data']) >= df_historical_10a['Data'].min() and pd.to_datetime(e['Data']) <= df_historical_10a['Data'].max()]
annotations_list_ma = []
shapes_list_ma = []
max_y_plot_ma = df_historical_10a['Value'].max() if not df_historical_10a.empty else 150
min_y_plot_ma = df_historical_10a['Value'].min() if not df_historical_10a.empty else 0
for i, evento in enumerate(eventos_filtrados_plot_ma):
    event_date = pd.to_datetime(evento['Data'])
    shapes_list_ma.append({'type': 'line','x0': event_date, 'y0': 0, 'x1': event_date, 'y1': 1, 'xref': 'x', 'yref': 'paper','line': {'color': evento['color'], 'width': 1.5, 'dash': 'dashdot'}})
    annotations_list_ma.append({'x': event_date, 'y': max_y_plot_ma * (1.05 + i*0.05), 'xref': 'x', 'yref': 'y', 'text': f"<b>{evento['descricao']}</b><br>({event_date.strftime('%b %Y')})", 'showarrow': True, 'arrowhead': 2, 'arrowwidth':1.5, 'arrowcolor':evento['color'],'ax': 0, 'ay': evento['ay_offset'], 'font': {'color': 'black', 'size': 10},'bgcolor': evento['color'], 'opacity': 0.7, 'bordercolor': 'black', 'borderwidth':1, 'borderpad':2}) # Fundo com a cor do evento

fig_hist_ma.update_layout(
    title='Preço do Brent, Médias Móveis e Eventos Chave (Últimos 10 Anos)',
    xaxis_title='Data', yaxis_title='Preço (US$)', template='plotly_white', height=700,
    shapes=shapes_list_ma, annotations=annotations_list_ma,
    yaxis_range=[min_y_plot_ma * 0.85, max_y_plot_ma * 1.45], # Mais espaço para anotações
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_hist_ma, use_container_width=True)

# --- Seção 2: Decifrando os Eventos e Seus Impactos ---
st.markdown("---")
st.header("🗓️ Decifrando os Eventos e Seus Impactos")
# ... (Mantenha seu storytelling detalhado dos insights 1-4 e o adicional aqui) ...
insight_cols = st.columns(2)
with insight_cols[0]:
    st.subheader("Insight 1: O Poder da OPEP e a Guerra de Preços (2014-2016)")
    st.markdown("Em **novembro de 2014**, a OPEP chocou os mercados ao decidir **manter seus níveis de produção**, apesar da crescente oferta de óleo de xisto dos EUA. O resultado? Uma **queda vertiginosa nos preços**, como vemos no gráfico, que se estendeu até o início de 2016. Este período ilustra vividamente como as decisões estratégicas dos grandes produtores podem ditar tendências de preço.")
    st.subheader("Insight 2: COVID-19 - Um Choque de Demanda Sem Precedentes (2020)")
    st.markdown("O ano de 2020 trouxe a pandemia de COVID-19. Com o mundo em lockdown, a **demanda por petróleo despencou**. O gráfico mostra a queda abrupta em março/abril de 2020. A recuperação foi gradual, acompanhando a reabertura das economias e os cortes de produção da OPEP+.")
with insight_cols[1]:
    st.subheader("Insight 3: Tensões Geopolíticas e o Prêmio de Risco (Guerra na Ucrânia, 2022)")
    st.markdown("A **invasão da Ucrânia pela Rússia em fevereiro de 2022** reintroduziu um forte \"prêmio de risco\" geopolítico. O temor de interrupções no fornecimento russo levou a uma **alta expressiva** nos preços.")
    st.subheader("Insight 4: A Recuperação Pós-Pandemia e a Nova Dinâmica (2021-Presente)")
    st.markdown("Após o choque da COVID-19, vimos uma **recuperação sustentada dos preços** em 2021-2022, impulsionada pela retomada econômica e políticas da OPEP+. No entanto, preocupações com inflação e crescimento global em 2023-2024 introduziram nova volatilidade.")
st.subheader("Insight Adicional: Volatilidade como Constante")
st.markdown("Analisando a série, a **volatilidade** é uma característica marcante. Períodos de estabilidade são frequentemente interrompidos por movimentos bruscos, refletindo a complexa interação de oferta, demanda e geopolítica.")

# --- Seção 3: Mergulhando Mais Fundo nos Dados ---
st.markdown("---")
st.header("🔍 Mergulhando Mais Fundo: Decomposição, Distribuição e Volatilidade")

# --- 3.1 Decomposição da Série Temporal ---
st.subheader("Decompondo a Série: Tendência, Sazonalidade e Resíduos")
st.markdown("A decomposição nos ajuda a separar a tendência de longo prazo, padrões sazonais (se houver) e flutuações residuais.")
decomp_period = 252 # Dias úteis em um ano, para sazonalidade anual
if len(df_for_analysis_stats.dropna()) >= decomp_period * 2: 
    try:
        # Usar df_for_analysis_stats que é uma Série indexada por Data
        decomposition = seasonal_decompose(df_for_analysis_stats.dropna(), model='additive', period=decomp_period)
        
        fig_decomp = plt.figure(figsize=(12, 8))
        ax_trend = fig_decomp.add_subplot(411)
        decomposition.trend.plot(ax=ax_trend, legend=False)
        ax_trend.set_ylabel('Tendência')
        ax_trend.set_title('Decomposição da Série Temporal do Preço do Brent', fontsize=14)

        ax_seasonal = fig_decomp.add_subplot(412, sharex=ax_trend)
        decomposition.seasonal.plot(ax=ax_seasonal, legend=False)
        ax_seasonal.set_ylabel(f'Sazonalidade ({decomp_period}d)')
        
        ax_resid = fig_decomp.add_subplot(413, sharex=ax_trend)
        decomposition.resid.plot(ax=ax_resid, legend=False, linestyle=':')
        ax_resid.set_ylabel('Resíduo')

        ax_original = fig_decomp.add_subplot(414, sharex=ax_trend) # Adicionado para mostrar original
        df_for_analysis_stats.plot(ax=ax_original, legend=False)
        ax_original.set_ylabel('Original')
        
        plt.tight_layout()
        st.pyplot(fig_decomp)
        st.markdown("* **Tendência:** Direção geral. * **Sazonalidade:** Padrões recorrentes. * **Resíduo:** Flutuações restantes.")
    except Exception as e:
        st.warning(f"Não foi possível realizar a decomposição sazonal com período {decomp_period}: {e}.")
else:
    st.warning(f"Dados insuficientes para decomposição sazonal com período {decomp_period}.")

# --- 3.2 Distribuição dos Preços e Retornos ---
st.subheader("Como os Preços e Seus Retornos se Distribuem?")
df_historical_10a['RetornoDiario'] = df_historical_10a['Value'].pct_change()
df_returns_analysis_app = df_historical_10a.dropna(subset=['RetornoDiario'])

col_dist_preco, col_dist_ret = st.columns(2)
with col_dist_preco:
    fig_hist_dist = px.histogram(df_historical_10a, x="Value", nbins=50, title="Histograma dos Preços")
    fig_hist_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist_dist, use_container_width=True)
    st.markdown("O histograma mostra a frequência dos diferentes níveis de preço.")

with col_dist_ret:
    if not df_returns_analysis_app.empty:
        fig_ret_dist_app = px.histogram(df_returns_analysis_app, x="RetornoDiario", nbins=100, title="Distribuição dos Retornos Diários")
        fig_ret_dist_app.update_layout(bargap=0.1)
        st.plotly_chart(fig_ret_dist_app, use_container_width=True)
        st.markdown("A distribuição dos retornos frequentemente apresenta 'caudas pesadas' (mais eventos extremos).")
        st.write(f"**Curtose dos Retornos:** {df_returns_analysis_app['RetornoDiario'].kurtosis():.2f}")
        st.write(f"**Assimetria dos Retornos:** {df_returns_analysis_app['RetornoDiario'].skew():.2f}")
    else:
        st.warning("Não foi possível gerar a distribuição dos retornos.")

# --- 3.3 Volatilidade ao Longo do Tempo ---
st.subheader("Medindo a Instabilidade: Volatilidade Móvel")
st.markdown("Calculamos o desvio padrão dos retornos diários (janela de 30 dias, anualizada) para visualizar períodos de maior turbulência.")
if not df_returns_analysis_app.empty: # Reusa df_returns_analysis_app
    df_historical_10a['Volatilidade30d'] = df_returns_analysis_app['RetornoDiario'].rolling(window=30).std() * np.sqrt(252) 
    fig_vol = px.line(df_historical_10a.dropna(subset=['Volatilidade30d']), 
                      x='Data', y='Volatilidade30d', 
                      title='Volatilidade Móvel de 30 Dias (Anualizada)')
    fig_vol.update_layout(yaxis_title='Volatilidade Anualizada', template='plotly_white')
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("Picos neste gráfico indicam períodos de alta incerteza, frequentemente coincidindo com eventos chave.")
else:
    st.warning("Não foi possível calcular a volatilidade móvel.")

# --- Seção 4: Padrões de Autocorrelação e Estacionariedade ---
st.markdown("---")
st.header("🔍 Padrões de Autocorrelação e Teste de Estacionariedade")

# --- 4.1 Teste ADF ---
st.subheader("Teste Formal de Estacionariedade (ADF)")
if not df_for_analysis_stats.dropna().empty:
    adf_stat_orig, p_value_orig, _, _, critical_values_orig, _ = adfuller(df_for_analysis_stats.dropna())
    st.write(f"**Série de Preços Original ('Value'):**")
    st.write(f"  - Estatística ADF: {adf_stat_orig:.4f}")
    st.write(f"  - P-valor: {p_value_orig:.4f}")
    st.write(f"  - É estacionária (p <= 0.05)? {'Sim' if p_value_orig <= 0.05 else 'Não'}")
    st.caption(f"Valores Críticos: 1%: {critical_values_orig['1%']:.2f}, 5%: {critical_values_orig['5%']:.2f}, 10%: {critical_values_orig['10%']:.2f}")

    log_series = np.log(df_for_analysis_stats.dropna().replace(0, 1e-5)) # Evitar log(0)
    diff_log_series = log_series.diff().dropna()
    if not diff_log_series.empty:
        adf_stat_trans, p_value_trans, _, _, critical_values_trans, _ = adfuller(diff_log_series)
        st.write(f"**Série Log-Diferenciada:**")
        st.write(f"  - Estatística ADF: {adf_stat_trans:.4f}")
        st.write(f"  - P-valor: {p_value_trans:.4f}")
        st.write(f"  - É estacionária (p <= 0.05)? {'Sim' if p_value_trans <= 0.05 else 'Não'}")
    st.markdown("Um p-valor baixo sugere que a série é estacionária após as transformações, o que é importante para modelos ARIMA.")
else:
    st.warning("Não foi possível realizar o teste ADF.")

# --- 4.2 ACF e PACF ---
st.subheader("Gráficos de Autocorrelação (ACF e PACF) da Série Original")
col1_acf_end, col2_pacf_end = st.columns(2)
if not df_for_analysis_stats.dropna().empty: 
    with col1_acf_end:
        fig_acf_plot, ax_acf = plt.subplots(figsize=(8, 4))
        plot_acf(df_for_analysis_stats.dropna(), lags=40, ax=ax_acf) 
        ax_acf.set_title("Autocorrelação (ACF) - Preços Originais")
        st.pyplot(fig_acf_plot)
        st.markdown("O decaimento lento no ACF sugere não-estacionariedade e a necessidade de diferenciação.")
    with col2_pacf_end:
        fig_pacf_plot, ax_pacf = plt.subplots(figsize=(8, 4))
        plot_pacf(df_for_analysis_stats.dropna(), lags=40, ax=ax_pacf, method='ywm') 
        ax_pacf.set_title("Autocorrelação Parcial (PACF) - Preços Originais")
        st.pyplot(fig_pacf_plot)
        st.markdown("O PACF ajuda a identificar a ordem 'p' de um modelo AR.")
else:
    st.warning("Não foi possível plotar ACF/PACF.")

# --- 4.3 Heatmap de Sazonalidade Mensal ---
st.subheader("Heatmap de Sazonalidade Mensal")
st.markdown("Visualiza o preço médio mensal ao longo dos anos para identificar possíveis padrões sazonais.")
df_heatmap_src = df_historical_10a.copy()
df_heatmap_src['Ano'] = df_heatmap_src['Data'].dt.year
# Tentar obter nomes dos meses em português, se o locale estiver configurado
try:
    import locale
    # Tentar configurar para o locale do sistema ou um específico pt_BR
    try:
        locale.setlocale(locale.LC_TIME, '') # Locale do sistema
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8') # pt_BR específico
        except locale.Error:
            st.caption("Locale pt_BR não disponível, meses podem aparecer em inglês.")
    df_heatmap_src['Mês'] = df_heatmap_src['Data'].dt.strftime('%B').str.capitalize()
    months_order_heatmap = [pd.Timestamp(2000, i, 1).strftime('%B').capitalize() for i in range(1,13)]
except ImportError: # Se locale não estiver disponível
    df_heatmap_src['Mês'] = df_heatmap_src['Data'].dt.month_name()
    months_order_heatmap = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]


heatmap_data_plot = pd.pivot_table(df_heatmap_src, values='Value', 
                              index='Ano', columns='Mês', 
                              aggfunc='mean')
# Reordenar colunas para ordem cronológica dos meses
try:
    heatmap_data_plot = heatmap_data_plot.reindex(columns=months_order_heatmap)
except Exception: # Se a reordenação falhar (ex: nomes de meses diferentes)
    st.caption("Não foi possível reordenar os meses no heatmap automaticamente.")
    heatmap_data_plot.sort_index(axis=1, inplace=True) # Tenta ordenação alfabética como fallback

if not heatmap_data_plot.empty:
    fig_heatmap_plot, ax_heatmap = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data_plot, annot=True, fmt=".1f", cmap="viridis", ax=ax_heatmap, linewidths=.5, cbar_kws={'label': 'Preço Médio (USD)'})
    ax_heatmap.set_title('Preço Médio Mensal do Petróleo Brent por Ano (Últimos 10 Anos)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig_heatmap_plot)
    st.markdown("O heatmap mostra o preço médio para cada mês ao longo dos anos. Cores mais claras indicam preços mais altos. Isso pode ajudar a identificar se certos meses consistentemente apresentam preços mais elevados ou mais baixos.")
else:
    st.warning("Não foi possível gerar o heatmap de sazonalidade.")

st.markdown("---")
st.subheader("Conclusões da Análise Exploratória")
st.markdown("""
A análise dos últimos 10 anos do preço do petróleo Brent revela uma série temporal complexa, caracterizada por:
- **Não-estacionariedade:** A série de preços original exibe tendências e não possui média e variância constantes.
- **Impacto de Eventos Externos:** Eventos geopolíticos, decisões da OPEP e crises globais (como a pandemia) têm um impacto significativo e muitas vezes abrupto nos preços.
- **Volatilidade Variável:** Existem períodos de maior e menor instabilidade.
- **Sazonalidade Potencial:** Embora a decomposição e o heatmap possam sugerir alguns padrões sazonais (ex: anuais ou mensais), eles podem não ser estritamente periódicos e podem ser influenciados por outros fatores dominantes. A sazonalidade precisa ser cuidadosamente considerada na modelagem.

Esses insights são cruciais para a seleção e configuração dos modelos preditivos, destacando a necessidade de transformações de dados (para estacionariedade) e a possível inclusão de features que capturem sazonalidade e outros fatores externos.
""")