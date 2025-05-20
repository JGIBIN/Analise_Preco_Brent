import streamlit as st
import pandas as pd
import numpy as np # Usado para cálculos numéricos como logaritmo e raiz quadrada
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose # Para decompor a série temporal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Para gráficos de autocorrelação
from statsmodels.tsa.stattools import adfuller # Para o teste de estacionariedade (ADF)
import matplotlib.pyplot as plt # Para alguns gráficos estatísticos
from utils import load_historical_data # Função para carregar os dados

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
                               mode='lines', name='Preço Brent', line=dict(color='deepskyblue', width=2)))
fig_hist_ma.add_trace(go.Scatter(x=df_historical_10a['Data'], y=df_historical_10a['MA50'],
                               mode='lines', name='Média Móvel 50 Dias', line=dict(color='orange', width=1.5, dash='dot')))
fig_hist_ma.add_trace(go.Scatter(x=df_historical_10a['Data'], y=df_historical_10a['MA200'],
                               mode='lines', name='Média Móvel 200 Dias', line=dict(color='crimson', width=1.5, dash='dash')))
eventos = [
    {'Data': '2014-11-27', 'descricao': 'OPEP mantém produção, preços caem', 'color': 'darkgrey', 'ay_offset': -40},
    {'Data': '2016-01-20', 'descricao': 'Preço atinge mínima da década (pós-2014)', 'color': 'darkgrey', 'ay_offset': -70},
    {'Data': '2020-03-11', 'descricao': 'Impacto COVID-19', 'color': 'darkgrey', 'ay_offset': -50},
    {'Data': '2020-04-20', 'descricao': 'WTI Negativo (Impacto Brent)', 'color': 'darkgrey', 'ay_offset': -70},
    {'Data': '2022-02-24', 'descricao': 'Início da Guerra na Ucrânia', 'color': 'darkgrey', 'ay_offset': -50}
]
eventos_filtrados_plot_ma = [e for e in eventos if pd.to_datetime(e['Data']) >= df_historical_10a['Data'].min() and pd.to_datetime(e['Data']) <= df_historical_10a['Data'].max()]

annotations_list_ma = []
shapes_list_ma = []

max_y_plot_ma = df_historical_10a['Value'].max() if not df_historical_10a.empty else 150
min_y_plot_ma = df_historical_10a['Value'].min() if not df_historical_10a.empty else 0

# Calcule os limites reais do eixo Y que serão usados no layout do gráfico
y_axis_plot_lower_bound = min_y_plot_ma * 0.85
y_axis_plot_upper_bound = max_y_plot_ma * 1.45 # Garante espaço para as anotações acima dos dados

for i, evento in enumerate(eventos_filtrados_plot_ma):
    event_date = pd.to_datetime(evento['Data'])
    y_annotation_level = max_y_plot_ma * (1.05 + i * 0.05)

    # --- MODIFIED SHAPE DEFINITION ---
    shapes_list_ma.append({
        'type': 'line',
        'x0': event_date,
        'y0': y_axis_plot_lower_bound,    # Linha começa na base calculada do eixo Y
        'x1': event_date,
        'y1': y_annotation_level,       # Linha termina no nível Y do texto da anotação
        'xref': 'x',
        'yref': 'y',                    # Coordenadas Y agora são em unidades de dados
        'line': {
            'color': evento['color'],
            'width': 1.5,
            'dash': 'dashdot'
        }
    })
    # --- END OF MODIFIED SHAPE DEFINITION ---

    annotations_list_ma.append({
        'x': event_date,
        'y': y_annotation_level,
        'xref': 'x',
        'yref': 'y',
        'text': f"<b>{evento['descricao']}</b><br>({event_date.strftime('%b %Y')})",
        'showarrow': True,
        'arrowhead': 2,
        'arrowwidth': 1.5,
        'arrowcolor': evento['color'],
        'ax': 0,
        'ay': evento['ay_offset'],
        'font': {'color': 'black', 'size': 10},
        'bgcolor': evento['color'],
        'opacity': 0.75, # Ajustado para melhor visibilidade com 'darkgrey'
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2
    })

fig_hist_ma.update_layout(
    title='Preço do Brent, Médias Móveis e Eventos Chave (Últimos 10 Anos)',
    xaxis_title='Data', yaxis_title='Preço (US$)', template='plotly_white', height=700,
    shapes=shapes_list_ma,
    annotations=annotations_list_ma,
    yaxis_range=[y_axis_plot_lower_bound, y_axis_plot_upper_bound], # Usa os limites calculados
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_hist_ma, use_container_width=True)

# --- Seção 2: Decifrando os Eventos e Seus Impactos ---
st.markdown("---")
st.header("🗓️ Decifrando os Eventos e Seus Impactos: O Que Aprendemos?")
st.markdown("""
Analisar o gráfico acima nos ajuda a entender como certos acontecimentos globais podem sacudir o mercado de petróleo.
A seguir, alguns dos principais aprendizados que tiramos ao observar a relação entre os eventos e os preços:
""")

insight_cols = st.columns(2)
with insight_cols[0]:
    st.subheader("Insight 1: O Poder da OPEP e a Guerra de Preços (2014-2016)")
    st.markdown("No final de 2014, a OPEP (Organização dos Países Exportadores de Petróleo) surpreendeu o mundo ao decidir **não cortar sua produção**, mesmo com mais petróleo vindo dos EUA (o chamado óleo de xisto). O resultado? Os **preços despencaram**, como uma pedra rolando ladeira abaixo, até o início de 2016. Isso mostra como as decisões de grandes produtores podem ditar o rumo dos preços.")

    st.subheader("Insight 2: COVID-19 - Um Choque de Demanda Sem Precedentes (2020)")
    st.markdown("A chegada da pandemia em 2020 foi um baque enorme. Com o mundo parando (lockdowns, voos cancelados, menos carros nas ruas), a **procura por petróleo caiu drasticamente**. O gráfico mostra essa queda brusca em março/abril de 2020. A recuperação dos preços foi lenta, acompanhando a reabertura gradual das economias e os cortes de produção feitos pela OPEP e seus aliados (OPEP+).")
with insight_cols[1]:
    st.subheader("Insight 3: Tensões Geopolíticas e o \"Prêmio de Risco\" (Guerra na Ucrânia, 2022)")
    st.markdown("Quando a **Rússia invadiu a Ucrânia em fevereiro de 2022**, o mercado ficou muito preocupado com a oferta de petróleo, já que a Rússia é um grande produtor. Esse medo de faltar petróleo fez os **preços dispararem**. Chamamos isso de \"prêmio de risco geopolítico\" – basicamente, o preço sobe porque há uma incerteza grande sobre o fornecimento futuro devido a conflitos.")

    st.subheader("Insight 4: A Recuperação Pós-Pandemia e a Nova Dinâmica (2021-Presente)")
    st.markdown("Depois do susto da COVID-19, os preços se recuperaram bem em 2021 e parte de 2022. A economia mundial estava voltando ao normal e a OPEP+ controlava a produção. Mas, mais recentemente (2023-2024), novas preocupações com inflação alta e um crescimento econômico mais fraco no mundo trouxeram mais instabilidade e altos e baixos para os preços.")

st.subheader("Insight Adicional: A Volatilidade é a Regra, Não a Exceção")
st.markdown("Olhando para toda a década, uma coisa fica clara: o preço do petróleo é **volátil**, ou seja, muda bastante e às vezes de forma muito rápida. Momentos de calmaria são muitas vezes seguidos por grandes saltos ou quedas. Isso acontece porque o preço é influenciado por uma teia complexa de fatores: quanta oferta existe, quanta demanda existe, e o que está acontecendo na política global.")

# --- Seção 3: Mergulhando Mais Fundo nos Dados ---
st.markdown("---")
st.header("🔍 Mergulhando Mais Fundo: Desvendando Padrões Escondidos")
st.markdown("""
Agora, vamos usar algumas ferramentas estatísticas para olhar os dados de formas diferentes.
Isso nos ajuda a encontrar padrões que não são óbvios apenas olhando o gráfico de preços.
Não se preocupe com os nomes técnicos, vamos focar no que cada análise nos diz.
""")

# --- 3.1 Decomposição da Série Temporal ---
st.subheader("Separando o Sinal do Ruído: Tendência, Sazonalidade e Resíduos")
st.markdown("""
**O que é isso?** A "decomposição" é como desmontar um motor para entender cada peça. Separamos o movimento do preço em três partes:
* **Tendência:** A direção geral do preço ao longo do tempo (está subindo, descendo ou estável no longo prazo?).
* **Sazonalidade:** Padrões que se repetem em intervalos fixos? Por exemplo, será que o preço do petróleo tende a ser mais alto em certos meses do ano, todos os anos? (Usamos um ciclo anual de 252 dias úteis).
* **Resíduo:** O que sobra depois de tirarmos a tendência e a sazonalidade. São as flutuações mais aleatórias, o "ruído" do mercado.

**Por que isso importa?** Entender essas partes nos ajuda a ver se existem padrões previsíveis e o quão "barulhento" ou imprevisível é o mercado.
""")
decomp_period = 252 # Aproximadamente o número de dias úteis em um ano, para tentar capturar uma sazonalidade anual.
if len(df_for_analysis_stats.dropna()) >= decomp_period * 2: # Precisamos de pelo menos dois ciclos completos para decompor
    try:
        decomposition = seasonal_decompose(df_for_analysis_stats.dropna(), model='additive', period=decomp_period)

        fig_decomp = plt.figure(figsize=(12, 10)) # Aumentado o tamanho para melhor visualização

        # Gráfico da Série Original
        ax_original = fig_decomp.add_subplot(411)
        df_for_analysis_stats.plot(ax=ax_original, legend=False)
        ax_original.set_ylabel('Preço Original')
        ax_original.set_title('Decomposição da Série Temporal do Preço do Brent', fontsize=14)

        # Gráfico da Tendência
        ax_trend = fig_decomp.add_subplot(412, sharex=ax_original) # Compartilha o eixo X com o original
        decomposition.trend.plot(ax=ax_trend, legend=False)
        ax_trend.set_ylabel('Tendência')
        ax_trend.tick_params(axis='x', which='both', bottom=False, labelbottom=False) # Remove labels do eixo x para economizar espaço

        # Gráfico da Sazonalidade
        ax_seasonal = fig_decomp.add_subplot(413, sharex=ax_original)
        decomposition.seasonal.plot(ax=ax_seasonal, legend=False)
        ax_seasonal.set_ylabel(f'Sazonalidade ({decomp_period}d)')
        ax_seasonal.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Gráfico do Resíduo
        ax_resid = fig_decomp.add_subplot(414, sharex=ax_original)
        decomposition.resid.plot(ax=ax_resid, legend=False, linestyle=':')
        ax_resid.set_ylabel('Resíduo (Ruído)')

        plt.xlabel("Data") # Adiciona label do eixo X apenas no último gráfico
        plt.tight_layout() # Ajusta o layout para evitar sobreposições
        st.pyplot(fig_decomp)
        st.markdown("""
        * **Observando a Tendência:** Vemos a direção de longo prazo, filtrando as oscilações diárias.
        * **Observando a Sazonalidade:** Se houver um padrão claro que se repete anualmente, ele aparecerá aqui. Muitas vezes, no petróleo, a sazonalidade pode ser sutil ou ofuscada por grandes eventos.
        * **Observando o Resíduo:** Mostra as variações que não são explicadas pela tendência ou sazonalidade. Idealmente, queremos que os resíduos sejam aleatórios.
        """)
    except Exception as e:
        st.warning(f"Não foi possível realizar a decomposição da série. Pode ser que não haja dados suficientes ou outro problema técnico: {e}")
else:
    st.warning(f"Não temos dados suficientes (precisamos de pelo menos {decomp_period*2} dias) para realizar a decomposição com um período anual de {decomp_period} dias.")

# --- 3.2 Distribuição dos Preços e Retornos ---
st.subheader("Com Que Frequência Cada Preço (ou Variação de Preço) ocorreu?")
st.markdown("""
Aqui, olhamos para dois tipos de "fotografias" dos dados:
1.  **Histograma dos Preços:** Mostra quantas vezes o preço do petróleo atingiu cada faixa de valor (ex: quantas vezes esteve entre $50-$55, $55-$60, etc.). Ajuda a ver se existem níveis de preço mais comuns.
2.  **Distribuição dos Retornos Diários:** Em vez do preço em si, olhamos para a *variação percentual* do preço de um dia para o outro (o "retorno"). Isso nos diz se grandes variações diárias são comuns ou raras.
""")

df_historical_10a['RetornoDiario'] = df_historical_10a['Value'].pct_change() # Calcula a variação percentual diária
df_returns_analysis_app = df_historical_10a.dropna(subset=['RetornoDiario']) # Remove dias sem retorno (o primeiro dia)

col_dist_preco, col_dist_ret = st.columns(2)
with col_dist_preco:
    fig_hist_dist = px.histogram(df_historical_10a, x="Value", nbins=50, title="Como os Preços se Distribuíram?")
    fig_hist_dist.update_layout(bargap=0.1, yaxis_title="Frequência (nº de dias)", xaxis_title="Preço (US$)")
    st.plotly_chart(fig_hist_dist, use_container_width=True)
    st.markdown("Este gráfico (histograma) mostra a frequência de diferentes níveis de preço. Barras mais altas indicam faixas de preço que ocorreram mais vezes nos últimos 10 anos.")

with col_dist_ret:
    if not df_returns_analysis_app.empty:
        fig_ret_dist_app = px.histogram(df_returns_analysis_app, x="RetornoDiario", nbins=100, title="Como as Variações Diárias de Preço se Distribuíram?")
        fig_ret_dist_app.update_layout(bargap=0.1, yaxis_title="Frequência (nº de dias)", xaxis_title="Variação Percentual Diária")
        st.plotly_chart(fig_ret_dist_app, use_container_width=True)
        st.markdown("""
        Este gráfico mostra a frequência das variações percentuais diárias do preço.
        * **"Caudas Pesadas" (Curtose Alta):** Se você vir que variações extremas (muito positivas ou muito negativas, nas pontas do gráfico) são mais frequentes do que o esperado numa curva "normal" (forma de sino), dizemos que a distribuição tem "caudas pesadas". Isso significa que dias com grandes surpresas (altas ou baixas fortes) são mais comuns.
        * **Assimetria:** Se o gráfico pender mais para um lado, indica se as variações positivas foram mais/menos frequentes ou intensas que as negativas.
        """)
        st.write(f"**Curtose dos Retornos (quão 'pesadas' são as caudas):** {df_returns_analysis_app['RetornoDiario'].kurtosis():.2f} (Valores > 3 indicam caudas mais pesadas que o normal)")
        st.write(f"**Assimetria dos Retornos (o quanto pende para um lado):** {df_returns_analysis_app['RetornoDiario'].skew():.2f} (Próximo de 0 é simétrico; negativo pende para retornos negativos; positivo para retornos positivos)")
    else:
        st.warning("Não foi possível gerar a distribuição dos retornos diários.")

# --- 3.3 Volatilidade ao Longo do Tempo ---
st.subheader("Medindo a Febre do Mercado: A Volatilidade dos Preços")
st.markdown("""
**O que é Volatilidade?** Pense na volatilidade como a "intensidade das variações" do preço. Se o preço sobe e desce muito e rapidamente, a volatilidade é alta. Se ele muda pouco, é baixa.
Calculamos uma "volatilidade móvel": para cada dia, olhamos para o desvio padrão (uma medida de quão espalhados estão os dados) dos retornos diários nos últimos 30 dias. Multiplicamos por um fator para ter uma ideia anualizada.

**Por que isso importa?** Este gráfico nos mostra os períodos em que o mercado esteve mais "nervoso" ou incerto (picos de volatilidade) versus períodos de maior calmaria. Geralmente, os picos coincidem com aqueles eventos importantes que vimos antes.
""")
if not df_returns_analysis_app.empty:
    # Calcula a volatilidade móvel de 30 dias, anualizada
    df_historical_10a['Volatilidade30d'] = df_returns_analysis_app['RetornoDiario'].rolling(window=30).std() * np.sqrt(252) # 252 dias úteis no ano

    fig_vol = px.line(df_historical_10a.dropna(subset=['Volatilidade30d']),
                      x='Data', y='Volatilidade30d',
                      title='Intensidade das Variações do Preço (Volatilidade Móvel Anualizada)')
    fig_vol.update_layout(yaxis_title='Volatilidade Anualizada Estimada', xaxis_title='Data', template='plotly_white')
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("Picos neste gráfico indicam períodos de alta incerteza e instabilidade nos preços, frequentemente ligados a eventos econômicos ou geopolíticos significativos, exemplo a alta do preço entre 2020 e 2021 devido a pandemia de COVID-19"
else:
    st.warning("Não foi possível calcular a volatilidade móvel, pois não há dados de retorno diário.")

# --- Seção 4: Padrões de Autocorrelação e Estacionariedade ---
st.markdown("---")
st.header("🧠 Entendendo a Memória dos Preços e sua Estabilidade")
st.markdown("""
Nesta seção, vamos investigar duas coisas importantes para quem quer prever preços:
1.  **Estacionariedade:** Basicamente, queremos saber se as características do preço do petróleo (como a média e a volatilidade) mudam muito com o tempo. Pense num rio: um rio estacionário tem um fluxo e nível mais ou menos constantes. Um não-estacionário pode ter secas e enchentes. Modelos de previsão geralmente funcionam melhor com dados "estacionários".
2.  **Autocorrelação:** Queremos ver se o preço de hoje tem relação com o preço de ontem, de anteontem, etc. Isso é como perguntar: o preço tem "memória"?
""")

# --- 4.1 Teste de Estacionariedade (ADF) ---
st.subheader("O Preço é Estável ao Longo do Tempo? (Teste ADF)")
st.markdown("""
Usamos um teste estatístico chamado **Augmented Dickey-Fuller (ADF)** para verificar a estacionariedade.
* **Hipótese Nula (H0) do teste:** A série NÃO é estacionária (possui raiz unitária, ou seja, tem uma tendência que não se reverte).
* **P-valor:** Se o p-valor for baixo (geralmente < 0.05), rejeitamos H0 e concluímos que a série É provavelmente estacionária. Se for alto, não podemos dizer que é estacionária.

Muitas vezes, os preços brutos não são estacionários. Uma transformação comum é usar a **log-diferença**, que é basicamente olhar para a variação percentual diária de uma forma um pouco diferente. Isso ajuda a estabilizar a série.
""")
if not df_for_analysis_stats.dropna().empty:
    # Teste na série original de preços
    adf_result_orig = adfuller(df_for_analysis_stats.dropna())
    st.write(f"**1. Teste nos Preços Originais ('Value'):**")
    st.write(f"  - Estatística do Teste ADF: {adf_result_orig[0]:.4f}")
    st.write(f"  - P-valor: {adf_result_orig[1]:.4f}")
    if adf_result_orig[1] <= 0.05:
        st.markdown("  - <span style='color:green'>**Conclusão: A série de preços original PARECE ser estacionária (P-valor ≤ 0.05).**</span> (Isso é incomum para séries de preços brutos, vale investigar mais a fundo se outros indicadores como ACF/PACF e o gráfico visual também sugerem isso).", unsafe_allow_html=True)
    else:
        st.markdown("  - <span style='color:red'>**Conclusão: A série de preços original NÃO parece ser estacionária (P-valor > 0.05).**</span> (Isso é o esperado para a maioria das séries de preços.)", unsafe_allow_html=True)
    st.caption(f"  Valores Críticos do Teste: 1%: {adf_result_orig[4]['1%']:.2f}, 5%: {adf_result_orig[4]['5%']:.2f}, 10%: {adf_result_orig[4]['10%']:.2f}. Se a Estatística ADF for menor (mais negativa) que esses valores, isso reforça a estacionariedade.")

    # Transformação: logaritmo e depois diferenciação
    log_series = np.log(df_for_analysis_stats.dropna().replace(0, 1e-9)) # Evitar log(0), substituindo 0 por um valor muito pequeno
    diff_log_series = log_series.diff().dropna() # Pega a diferença entre um dia e o anterior (na escala log)

    if not diff_log_series.empty:
        adf_result_trans = adfuller(diff_log_series)
        st.write(f"**2. Teste nos Retornos Logarítmicos Diferenciados (uma forma de ver a variação diária):**")
        st.write(f"  - Estatística do Teste ADF: {adf_result_trans[0]:.4f}")
        st.write(f"  - P-valor: {adf_result_trans[1]:.4f}")
        if adf_result_trans[1] <= 0.05:
            st.markdown("  - <span style='color:green'>**Conclusão: A série transformada PARECE ser estacionária (P-valor ≤ 0.05).**</span> (Isso é bom! Significa que as variações diárias são mais estáveis e previsíveis do que os níveis de preço brutos).", unsafe_allow_html=True)
        else:
            st.markdown("  - <span style='color:red'>**Conclusão: Mesmo após a transformação, a série NÃO parece ser estacionária (P-valor > 0.05).**</span> (Isso pode indicar a necessidade de outras transformações ou que há padrões mais complexos).", unsafe_allow_html=True)
    st.markdown("Para construir bons modelos de previsão (como ARIMA), geralmente precisamos que os dados sejam estacionários. Se não são, a transformação (como a log-diferenciação) é um passo importante.")
else:
    st.warning("Não foi possível realizar o teste ADF, pois não há dados suficientes na série principal.")

# --- 4.2 ACF e PACF ---
st.subheader("O Preço de Hoje Depende do de Ontem? (Gráficos de Autocorrelação - ACF e PACF)")
st.markdown("""
Estes gráficos nos ajudam a ver se os preços passados têm "memória" e influenciam os preços atuais.
* **ACF (Função de Autocorrelação):** Mostra a correlação do preço de hoje com os preços de 1 dia atrás, 2 dias atrás, 3 dias atrás, etc. (chamamos esses "atrasos" ou "lags"). Uma barra que sai da área azul indica uma correlação significativa.
* **PACF (Função de Autocorrelação Parcial):** É um pouco mais esperta. Também mostra a correlação com preços passados, mas remove o efeito dos "atrasos" intermediários. Por exemplo, para o lag 3, ela mostra a correlação direta com 3 dias atrás, tirando a influência que os lags 1 e 2 já teriam.

**O que procurar (para a série de preços original, que geralmente NÃO é estacionária):**
* **No ACF:** Se as barras diminuem LENTAMENTE, isso é um forte sinal de que a série NÃO é estacionária e tem uma tendência. É como um eco que demora muito para sumir.
* **No PACF:** Frequentemente, o primeiro lag (1 dia atrás) é muito forte, e os outros caem rapidamente.

**Por que isso importa?** Esses gráficos dão pistas importantes para configurar modelos de previsão mais avançados (como os modelos ARIMA), ajudando a decidir quantos "atrasos" do passado são importantes para prever o futuro. Para stakeholders, o principal é saber que estamos investigando esses padrões para melhorar a precisão das previsões.
""")
col1_acf_end, col2_pacf_end = st.columns(2)
if not df_for_analysis_stats.dropna().empty:
    with col1_acf_end:
        fig_acf_plot, ax_acf = plt.subplots(figsize=(8, 4))
        plot_acf(df_for_analysis_stats.dropna(), lags=40, ax=ax_acf) # Mostra até 40 dias de atraso
        ax_acf.set_title("Autocorrelação (ACF) - Preços Originais")
        st.pyplot(fig_acf_plot)
        st.markdown("Se as barras do ACF demoram a cair para dentro da área azul, isso sugere que o preço de hoje é fortemente influenciado pelos preços de muitos dias atrás, indicando uma tendência e não-estacionariedade.")
    with col2_pacf_end:
        fig_pacf_plot, ax_pacf = plt.subplots(figsize=(8, 4))
        plot_pacf(df_for_analysis_stats.dropna(), lags=40, ax=ax_pacf, method='ywm') # Método 'ywm' é comum
        ax_pacf.set_title("Autocorrelação Parcial (PACF) - Preços Originais")
        st.pyplot(fig_pacf_plot)
        st.markdown("O PACF nos ajuda a ver a influência direta de um preço passado específico, sem o efeito acumulado dos dias anteriores. Um pico forte no primeiro lag é comum.")
else:
    st.warning("Não foi possível plotar os gráficos ACF/PACF, pois não há dados suficientes.")


st.markdown("---")
st.subheader("🏁 Conclusões da Nossa Exploração Inicial")
st.markdown("""
Após essa jornada pelos dados do preço do petróleo Brent nos últimos 10 anos, o que aprendemos?

1.  **O Preço Não é Estável Sozinho (Não-Estacionário):** O preço do petróleo, quando olhado em sua forma bruta, tende a seguir tendências de longo prazo e não fica variando em torno de uma média constante. Sua "personalidade" (média, volatilidade) muda com o tempo. Para prever, geralmente precisamos transformar esses dados (ex: olhando as variações diárias) para torná-los mais estáveis.

2.  **Eventos Globais Mandam no Jogo:** Decisões da OPEP, crises econômicas globais (como a pandemia de COVID-19) e tensões geopolíticas (como a guerra na Ucrânia) têm um impacto GIGANTE e muitas vezes imediato nos preços. Isso torna o mercado de petróleo muito sensível ao que acontece no mundo.

3.  **A "Febre" do Mercado (Volatilidade) Muda Constantemente:** Vimos que há períodos em que os preços sobem e descem freneticamente (alta volatilidade) e outros de maior calmaria (baixa volatilidade). Entender esses "humores" do mercado é crucial.

4.  **Padrões Anuais (Sazonalidade) Podem Existir, Mas São Discretos:** Nossa análise de decomposição tenta buscar padrões que se repetem todo ano. Embora possa haver alguma influência sazonal (por exemplo, maior demanda por combustível de aquecimento no inverno do hemisfério norte), esses padrões são frequentemente "abafados" pelos grandes eventos globais. Precisamos investigar mais a fundo se essa sazonalidade é forte o suficiente para ser usada em modelos de previsão.

**E agora?**
Esses aprendizados são como o mapa do tesouro para os próximos passos: construir modelos de previsão. Eles nos dizem que:
* Precisaremos tratar os dados para que fiquem "estacionários".
* Modelos que consideram a "memória" dos preços (como os que usam ACF/PACF) podem ser úteis.
* Seria ideal se pudéssemos incorporar o impacto de grandes eventos externos ou indicadores de volatilidade nos nossos modelos, embora isso seja um desafio complexo.

Esta análise exploratória é o alicerce. Com ela, estamos mais preparados para escolher as ferramentas certas e construir previsões mais informadas sobre o futuro do preço do petróleo.
""")
