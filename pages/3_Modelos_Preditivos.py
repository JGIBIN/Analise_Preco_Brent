import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime as dt 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import load_model
import joblib 
from statsforecast import StatsForecast 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler # Importado para o scaler das exógenas do ARIMAX
# Importar tudo de utils, incluindo as classes de Pipeline e create_seasonal_features_for_streamlit
from utils import load_historical_data, PrepareData, FillNANValues, SomthDataIntervalValues, create_seasonal_features_for_streamlit, test_stationarity 

st.set_page_config(page_title="Modelos Preditivos", page_icon="🤖", layout="wide")
st.title("🤖 Modelos Preditivos para o Preço do Petróleo Brent")

df_historical_10a_app = load_historical_data() 
if df_historical_10a_app.empty:
    st.error("Não foi possível carregar os dados históricos (últimos 10 anos).")
    st.stop()

st.sidebar.info(f"Modelos usam dados de: {df_historical_10a_app['Data'].min().strftime('%Y-%m-%d')} a {df_historical_10a_app['Data'].max().strftime('%Y-%m-%d')}")
series_data_for_models_app = df_historical_10a_app.set_index('Data')['Value'].asfreq('D').ffill().bfill()

@st.cache_resource
def load_prediction_artifacts_app():
    lstm_m, scaler_l, sarima_m_sf, scaler_exog_ari = None, None, None, None # Adicionado scaler_exog_ari
    try:
        lstm_m = load_model('lstm_model.h5') 
        scaler_l = joblib.load('scaler.pkl')   
        st.sidebar.success("Modelo LSTM e scaler (multivariado) carregados!")
    except Exception as e: st.sidebar.warning(f"LSTM/scaler não carregado: {e}.")
    try:
        sarima_m_sf = StatsForecast.load('sarima_model_sf.pkl') 
        st.sidebar.info("Modelo ARIMAX (StatsForecast) pré-treinado carregado.")
        # Carregar o scaler para as features exógenas do ARIMAX
        scaler_exog_ari = joblib.load('scaler_exog_arima.pkl')
        st.sidebar.info("Scaler para features exógenas do ARIMAX carregado.")
    except Exception as e: st.sidebar.info(f"ARIMAX (StatsForecast) ou seu scaler não encontrado: {e}.")
    return lstm_m, scaler_l, sarima_m_sf, scaler_exog_ari

lstm_model_app, scaler_lstm_app, sarimax_model_sf_loaded_app, scaler_exog_arimax_app = load_prediction_artifacts_app() # Nome atualizado

@st.cache_data
def load_error_metrics_app(file_path='df_erros.csv'): 
    try: return pd.read_csv(file_path)
    except FileNotFoundError: return pd.DataFrame()
df_performance_metrics_app = load_error_metrics_app()

st.sidebar.header("Configurações da Previsão")
modelo_escolhido_app = st.sidebar.selectbox("🧠 Escolha o modelo",
    ["LSTM Híbrido", "ARIMAX", "SARIMAX"]) # Nomes atualizados

last_hist_date_app = series_data_for_models_app.index.max()
max_pred_date_app = pd.to_datetime('2025-12-31')
max_days_pred_app = (max_pred_date_app - last_hist_date_app).days if max_pred_date_app > last_hist_date_app else 365
periodo_previsao_app = st.sidebar.slider("🔮 Dias para prever", 1, max(1,max_days_pred_app), min(30,max(1,max_days_pred_app)))

if st.sidebar.button("Realizar Previsão 🚀"):
    col_name_plot_app = None 
    if periodo_previsao_app <= 0: st.warning("Período de previsão deve ser > 0.")
    else:
        future_dates_app = pd.date_range(start=last_hist_date_app + timedelta(days=1), periods=periodo_previsao_app, freq='D')
        df_forecast_display_app = pd.DataFrame({'Data': future_dates_app})
        
        # --- ARIMAX (StatsForecast Pré-treinado com Reversão e Features Exógenas) ---
        if modelo_escolhido_app == "ARIMAX":
            st.subheader(f"🔮 Previsão com ARIMAX para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão ARIMAX '
            if sarimax_model_sf_loaded_app and scaler_exog_arimax_app:
                with st.spinner("Processando ARIMAX (StatsForecast)..."):
                    try:
                        # 1. Aplicar a pipeline de pré-processamento aos dados históricos para obter y_ma_log
                        pipeline_arimax_prep_sf_app = Pipeline([
                            ('data_prepator', PrepareData(date_col='Data', value_col='Value')),
                            ('filler_nan_values', FillNANValues(value_col='Value', new_value_col='y', new_date_col='ds')),
                            ('smoother_data_interval', SomthDataIntervalValues(value_col='y'))])
                        df_processed_hist_sf_app = pipeline_arimax_prep_sf_app.fit_transform(df_historical_10a_app.copy())
                        
                        if df_processed_hist_sf_app.empty or 'y_ma_log' not in df_processed_hist_sf_app.columns:
                            st.error("Falha ao gerar 'y_ma_log' para reversão do StatsForecast.")
                            raise ValueError("Falha ao obter y_ma_log")
                        last_known_ma_log_app = df_processed_hist_sf_app['y_ma_log'].iloc[-1]
                        
                        # 2. Gerar e escalar features exógenas para o período futuro
                        df_future_dates_for_exog_arimax = pd.DataFrame({'Data': future_dates_app})
                        exog_future_arimax_unscaled = create_seasonal_features_for_streamlit(df_future_dates_for_exog_arimax, date_col_name='Data')
                        exog_future_arimax_scaled = scaler_exog_arimax_app.transform(exog_future_arimax_unscaled)
                        # StatsForecast X_df espera um DataFrame com colunas 'unique_id' e 'ds'
                        X_df_future_arimax = pd.DataFrame(exog_future_arimax_scaled, columns=exog_future_arimax_unscaled.columns)
                        X_df_future_arimax['unique_id'] = 'Brent' # Adicionar unique_id e ds para X_df
                        X_df_future_arimax['ds'] = future_dates_app

                        # 3. Fazer previsões (estarão na escala y_diff_log)
                        forecast_output_sf_app = sarimax_model_sf_loaded_app.predict(h=periodo_previsao_app, X_df=X_df_future_arimax)
                        
                        pred_col_sf_internal_app = 'AutoARIMA' 
                        if pred_col_sf_internal_app not in forecast_output_sf_app.columns:
                            pred_col_sf_internal_app = forecast_output_sf_app.columns[0] # Fallback
                        forecast_diff_log_app = forecast_output_sf_app[pred_col_sf_internal_app].values
                        
                        # 4. Reverter transformações
                        forecast_log_scale_app = forecast_diff_log_app + last_known_ma_log_app
                        df_forecast_display_app[col_name_plot_app] = np.exp(forecast_log_scale_app)
                        st.info("Previsão ARIMAX (StatsForecast) realizada e revertida.")
                    except Exception as e:
                        st.error(f"Erro no ARIMAX (StatsForecast) ou reversão: {e}")
                        df_forecast_display_app[col_name_plot_app] = [np.nan]*periodo_previsao_app
            else:
                st.error("Modelo ARIMAX (StatsForecast) ou seu scaler de exógenas não carregado.")
                df_forecast_display_app[col_name_plot_app] = [np.nan]*periodo_previsao_app

        # --- SARIMAX (Statsmodels com Features, treinado sob demanda) ---
        elif modelo_escolhido_app == "SARIMAX":
            # ... (código como na resposta anterior, que já incluía create_seasonal_features e scaler_exog_sarimax) ...
            # Certifique-se que create_seasonal_features_for_streamlit de utils.py é usado.
            st.subheader(f"🔮 Previsão com SARIMAX (statsmodels com features) para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão SARIMAX (statsmodels)'
            with st.spinner("Treinando SARIMAX (statsmodels)..."):
                df_hist_for_sarimax_app = df_historical_10a_app.copy() # Usa os dados de 10 anos
                
                target_series_log_app = np.log(df_hist_for_sarimax_app.set_index('Data')['Value'].replace(0, 1e-5)).dropna()

                exog_hist_df_app = create_seasonal_features_for_streamlit(df_hist_for_sarimax_app.reset_index(), date_col_name='Data')
                exog_hist_df_app.index = df_hist_for_sarimax_app.set_index('Data').index # Alinhar índice se df_hist_for_sarimax_app foi indexado por Data
                
                common_index_app = target_series_log_app.index.intersection(exog_hist_df_app.index)
                target_series_log_app = target_series_log_app.loc[common_index_app]
                exog_hist_df_app = exog_hist_df_app.loc[common_index_app]

                scaler_exog_sarimax_app_demand = MinMaxScaler() # Scaler para exógenas do SARIMAX on-demand
                exog_hist_scaled_app = scaler_exog_sarimax_app_demand.fit_transform(exog_hist_df_app)
                
                d_order_app_demand = 0 if test_stationarity(target_series_log_app) else 1 # Usar test_stationarity de utils
                
                try:
                    model_sm_app = SARIMAX(target_series_log_app, exog=exog_hist_scaled_app, 
                                           order=(2,d_order_app_demand,2), seasonal_order=(1,1,1,7),
                                           enforce_stationarity=False, enforce_invertibility=False)
                    model_fit_sm_app = model_sm_app.fit(disp=False)
                    
                    df_future_dates_for_exog_sm = pd.DataFrame({'Data': future_dates_app})
                    exog_future_df_sm = create_seasonal_features_for_streamlit(df_future_dates_for_exog_sm, date_col_name='Data')
                    exog_future_scaled_sm = scaler_exog_sarimax_app_demand.transform(exog_future_df_sm)
                    
                    forecast_log_sm_app = model_fit_sm_app.forecast(steps=periodo_previsao_app, exog=exog_future_scaled_sm)
                    df_forecast_display_app[col_name_plot_app] = np.exp(forecast_log_sm_app).values
                except Exception as e:
                    st.error(f"Erro no SARIMAX (statsmodels): {e}")
                    df_forecast_display_app[col_name_plot_app] = [np.nan]*periodo_previsao_app
        
        # --- LSTM Híbrido com Features Sazonais ---
        elif modelo_escolhido_app == "LSTM Híbrido":
            # ... (código LSTM como na resposta anterior, que já era multivariado) ...
            # Apenas garanta que seq_length_lstm_app e num_features_lstm_app estão corretos
            st.subheader(f"🔮 Previsão com LSTM Híbrido para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão LSTM Híbrido'
            if lstm_model_app is not None and scaler_lstm_app is not None:
                with st.spinner("Realizando previsão com LSTM Híbrido..."):
                    seq_length_lstm_app = 60 
                    num_features_lstm_app = 6 # y + 5 features sazonais
                    
                    df_hist_tail_for_lstm_app_calc = series_data_for_models_app.tail(seq_length_lstm_app).reset_index() 
                    df_hist_tail_for_lstm_app_calc.rename(columns={'Value':'y', 'Data':'ds'}, inplace=True)
                    
                    df_seasonal_features_hist = create_seasonal_features_for_streamlit(df_hist_tail_for_lstm_app_calc, date_col_name='ds')
                    
                    df_hist_tail_ordered_features = pd.concat(
                        [df_hist_tail_for_lstm_app_calc[['y']], df_seasonal_features_hist], axis=1
                    )
                    last_known_sequence_unscaled = df_hist_tail_ordered_features.values
                    
                    if last_known_sequence_unscaled.shape[0] < seq_length_lstm_app :
                         st.error(f"Não foi possível obter {seq_length_lstm_app} pontos com features para LSTM.")
                         df_forecast_display_app[col_name_plot_app] = [np.nan]*periodo_previsao_app; st.stop()

                    last_known_sequence_scaled = scaler_lstm_app.transform(last_known_sequence_unscaled)
                    current_sequence_app = last_known_sequence_scaled.reshape((1, seq_length_lstm_app, num_features_lstm_app))
                    future_preds_y_scaled_list_app = []
                    
                    for i in range(periodo_previsao_app):
                        next_pred_y_scaled = lstm_model_app.predict(current_sequence_app, verbose=0)[0,0]
                        future_preds_y_scaled_list_app.append(next_pred_y_scaled)
                        
                        temp_pred_for_inverse = np.zeros((1, num_features_lstm_app)); temp_pred_for_inverse[0,0] = next_pred_y_scaled
                        y_pred_denorm_current_step = scaler_lstm_app.inverse_transform(temp_pred_for_inverse)[0,0]
                        next_step_features_unscaled_app = np.zeros(num_features_lstm_app)
                        next_step_features_unscaled_app[0] = y_pred_denorm_current_step 
                        next_prediction_true_date = last_hist_date_app + timedelta(days= i + 1)                         
                        df_next_date_temp = pd.DataFrame({'Data_temp': [next_prediction_true_date]})
                        seasonal_features_next_step = create_seasonal_features_for_streamlit(df_next_date_temp, date_col_name='Data_temp')
                        next_step_features_unscaled_app[1:] = seasonal_features_next_step.values.flatten()
                        next_step_scaled = scaler_lstm_app.transform(next_step_features_unscaled_app.reshape(1, -1))
                        new_timestep_for_lstm = next_step_scaled.reshape((1, 1, num_features_lstm_app))
                        current_sequence_app = np.append(current_sequence_app[:,1:,:], new_timestep_for_lstm, axis=1)
                    
                    final_preds_to_inverse = np.zeros((len(future_preds_y_scaled_list_app), num_features_lstm_app))
                    final_preds_to_inverse[:, 0] = np.array(future_preds_y_scaled_list_app).flatten()
                    df_forecast_display_app[col_name_plot_app] = scaler_lstm_app.inverse_transform(final_preds_to_inverse)[:, 0]
            else:
                st.error("Modelo LSTM ou scaler não carregado."); df_forecast_display_app[col_name_plot_app] = [np.nan]*periodo_previsao_app

        # Plotagem comum
        if col_name_plot_app and col_name_plot_app in df_forecast_display_app:
            fig_plot = go.Figure()
            # ... (código do gráfico permanece o mesmo) ...
            fig_plot.add_trace(go.Scatter(x=df_historical_10a_app['Data'].tail(180), y=df_historical_10a_app['Value'].tail(180), name="Histórico Recente", line=dict(color='blue')))
            fig_plot.add_trace(go.Scatter(x=df_forecast_display_app['Data'], y=df_forecast_display_app[col_name_plot_app], name=col_name_plot_app, line=dict(color='green' if "LSTM" in col_name_plot_app else 'orange', dash='dash')))
            fig_plot.update_layout(title=f"Previsão {modelo_escolhido_app}", xaxis_title="Data", yaxis_title="Preço (US$)")
            st.plotly_chart(fig_plot, use_container_width=True)
            
            st.write(f"Valores Previstos ({col_name_plot_app}):")
            
            # Prepara o DataFrame para exibição
            df_to_display = df_forecast_display_app[['Data', col_name_plot_app]].copy()
            
            # 1. Garante que a coluna 'Data' seja do tipo datetime (geralmente já é, mas bom verificar)
            df_to_display['Data'] = pd.to_datetime(df_to_display['Data'])
            
            # 2. Formata a coluna 'Data' para string no formato dd/mm/yyyy ANTES de definir como índice
            df_to_display['Data'] = df_to_display['Data'].dt.strftime('%d/%m/%Y')
            
            # 3. Define a coluna 'Data' (agora como string formatada) como índice
            df_to_display = df_to_display.set_index('Data')
            
            # 4. Arredonda a coluna de previsão para 2 casas decimais
            df_to_display[col_name_plot_app] = df_to_display[col_name_plot_app].round(2)
            
            st.dataframe(df_to_display)

        elif col_name_plot_app: st.error(f"Coluna de previsão '{col_name_plot_app}' não gerada.")
        else: st.error("Coluna de previsão não determinada.")

st.sidebar.markdown("---")
st.sidebar.subheader("Performance (Backtesting do Notebook - Dados de 10 anos)")
if not df_performance_metrics_app.empty:
    col_modelo_erro_app = 'modelo' if 'modelo' in df_performance_metrics_app.columns else 'Modelo' if 'Modelo' in df_performance_metrics_app.columns else None
    if col_modelo_erro_app: st.sidebar.dataframe(df_performance_metrics_app.set_index(col_modelo_erro_app))
    else: st.sidebar.dataframe(df_performance_metrics_app); st.sidebar.warning("Coluna 'modelo'/'Modelo' não encontrada em df_erros.csv.")
else:
    st.sidebar.info("Métricas de performance ('df_erros.csv') não encontradas ou vazias.")