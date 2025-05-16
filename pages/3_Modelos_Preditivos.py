import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime as dt 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import load_model
import joblib 
from statsforecast import StatsForecast 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler
from utils import (
    load_historical_data, PrepareData, FillNANValues,
    SomthDataIntervalValues, create_seasonal_features_for_streamlit, test_stationarity
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ------------------------- Função para sMAPE -------------------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# ------------------------- Configuração da Página -------------------------
st.set_page_config(page_title="Modelos Preditivos", page_icon="🤖", layout="wide")
st.title("🤖 Modelos Preditivos para o Preço do Petróleo Brent")

# ------------------------- Carregamento de Dados Históricos -------------------------
df_historical_10a_app = load_historical_data()
if df_historical_10a_app.empty:
    st.error("Não foi possível carregar os dados históricos (últimos 10 anos).")
    st.stop()

st.sidebar.info(f"Modelos usam dados de: {df_historical_10a_app['Data'].min().strftime('%d-%m-%Y')} a {df_historical_10a_app['Data'].max().strftime('%d-%m-%Y')}")

series_data_for_models_app = df_historical_10a_app.set_index('Data')['Value'].asfreq('D').ffill().bfill()

# ------------------------- Carregamento de Modelos -------------------------
@st.cache_resource
def load_prediction_artifacts_app():
    try:
        lstm_m = load_model('lstm_model.h5')
        scaler_l = joblib.load('scaler.pkl')
        st.sidebar.success("Modelo LSTM e scaler (multivariado) carregados!")
    except Exception as e:
        lstm_m, scaler_l = None, None
        st.sidebar.warning(f"LSTM/scaler não carregado: {e}.")

    try:
        sarima_m_sf = StatsForecast.load('sarima_model_sf.pkl')
        scaler_exog_ari = joblib.load('scaler_exog_arima.pkl')
        st.sidebar.info("Modelo ARIMAX (StatsForecast) e scaler carregados.")
    except Exception as e:
        sarima_m_sf, scaler_exog_ari = None, None
        st.sidebar.info(f"ARIMAX ou scaler não carregado: {e}.")

    return lstm_m, scaler_l, sarima_m_sf, scaler_exog_ari

lstm_model_app, scaler_lstm_app, sarimax_model_sf_loaded_app, scaler_exog_arimax_app = load_prediction_artifacts_app()

# ------------------------- Carregamento de Métricas Anteriores -------------------------
@st.cache_data
def load_error_metrics_app(file_path='df_erros.csv'):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_performance_metrics_app = load_error_metrics_app()

# ------------------------- Interface de Seleção -------------------------
st.sidebar.header("Configurações da Previsão")
modelo_escolhido_app = st.sidebar.selectbox("🧠 Escolha o modelo", ["LSTM Híbrido", "ARIMAX", "SARIMAX"])
last_hist_date_app = series_data_for_models_app.index.max()
max_pred_date_app = pd.to_datetime('2025-12-31')
max_days_pred_app = (max_pred_date_app - last_hist_date_app).days if max_pred_date_app > last_hist_date_app else 365
periodo_previsao_app = st.sidebar.slider("🔮 Dias para prever", 1, max(1, max_days_pred_app), min(30, max(1, max_days_pred_app)))
if st.sidebar.button("Realizar Previsão 🚀"):
    col_name_plot_app = None
    if periodo_previsao_app <= 0:
        st.warning("Período de previsão deve ser > 0.")
    else:
        future_dates_app = pd.date_range(start=last_hist_date_app + timedelta(days=1), periods=periodo_previsao_app, freq='D')
        df_forecast_display_app = pd.DataFrame({'Data': future_dates_app})

        # ------------------ ARIMAX ------------------
        if modelo_escolhido_app == "ARIMAX":
            st.subheader(f"🔮 Previsão com ARIMAX para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão ARIMAX '
            if sarimax_model_sf_loaded_app and scaler_exog_arimax_app:
                with st.spinner("Processando ARIMAX (StatsForecast)..."):
                    try:
                        # 1. Prepara dados históricos transformados
                        pipeline_arimax = Pipeline([
                            ('data_prepator', PrepareData(date_col='Data', value_col='Value')),
                            ('filler_nan_values', FillNANValues(value_col='Value', new_value_col='y', new_date_col='ds')),
                            ('smoother_data_interval', SomthDataIntervalValues(value_col='y'))
                        ])
                        df_hist_transf = pipeline_arimax.fit_transform(df_historical_10a_app.copy())

                        if df_hist_transf.empty or 'y_ma_log' not in df_hist_transf.columns:
                            st.error("Erro ao gerar y_ma_log para reversão do ARIMAX.")
                            raise ValueError("y_ma_log ausente.")

                        last_known_ma_log = df_hist_transf['y_ma_log'].iloc[-1]

                        # 2. Cria exógenas futuras e aplica scaler
                        df_future_exog = pd.DataFrame({'Data': future_dates_app})
                        exog_unscaled = create_seasonal_features_for_streamlit(df_future_exog, date_col_name='Data')
                        exog_scaled = scaler_exog_arimax_app.transform(exog_unscaled)

                        X_df = pd.DataFrame(exog_scaled, columns=exog_unscaled.columns)
                        X_df['unique_id'] = 'Brent'
                        X_df['ds'] = future_dates_app

                        # 3. Previsão com StatsForecast
                        forecast_output = sarimax_model_sf_loaded_app.predict(h=periodo_previsao_app, X_df=X_df)
                        pred_col = 'AutoARIMA' if 'AutoARIMA' in forecast_output.columns else forecast_output.columns[0]
                        forecast_diff_log = forecast_output[pred_col].values

                        # 4. Reversão das transformações
                        forecast_log = forecast_diff_log + last_known_ma_log
                        df_forecast_display_app[col_name_plot_app] = np.exp(forecast_log)
                        st.success("Previsão ARIMAX realizada e revertida com sucesso.")

                        # 5. Cálculo de métricas de erro
                        y_true = df_historical_10a_app.set_index('Data')['Value'].reindex(future_dates_app).dropna()
                        y_pred = df_forecast_display_app.set_index('Data').reindex(y_true.index)[col_name_plot_app]

                        if not y_true.empty and not y_pred.isnull().all():
                            mae = mean_absolute_error(y_true, y_pred)
                            rmse = mean_squared_error(y_true, y_pred, squared=False)
                            smape_val = smape(y_true.values, y_pred.values)

                            new_row = pd.DataFrame([{
                                'modelo': 'ARIMAX',
                                'MAE': mae,
                                'RMSE': rmse,
                                'sMAPE': smape_val,
                                'data_geracao': pd.Timestamp.now()
                            }])

                            file_path = 'df_erros.csv'
                            if os.path.exists(file_path):
                                df_existing = pd.read_csv(file_path)
                                df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                            else:
                                df_updated = new_row

                            df_updated.to_csv(file_path, index=False)
                            st.info("Métricas de erro ARIMAX salvas com sucesso.")

                    except Exception as e:
                        st.error(f"Erro ao processar ARIMAX: {e}")
                        df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
            else:
                st.error("Modelo ARIMAX ou scaler de exógenas não carregado.")
                df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
        # ------------------ SARIMAX ------------------
        elif modelo_escolhido_app == "SARIMAX":
            st.subheader(f"🔮 Previsão com SARIMAX (statsmodels) para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão SARIMAX (statsmodels)'
            with st.spinner("Treinando SARIMAX..."):
                try:
                    # 1. Série alvo (log) e exógenas
                    df_hist = df_historical_10a_app.copy()
                    target_series_log = np.log(df_hist.set_index('Data')['Value'].replace(0, 1e-5)).dropna()

                    exog_df = create_seasonal_features_for_streamlit(df_hist.reset_index(), date_col_name='Data')
                    exog_df.index = df_hist.set_index('Data').index

                    common_idx = target_series_log.index.intersection(exog_df.index)
                    target_series_log = target_series_log.loc[common_idx]
                    exog_df = exog_df.loc[common_idx]

                    scaler_exog = MinMaxScaler()
                    exog_scaled = scaler_exog.fit_transform(exog_df)

                    # 2. Teste de estacionariedade
                    d_order = 0 if test_stationarity(target_series_log) else 1

                    # 3. Treinamento do modelo
                    model = SARIMAX(
                        target_series_log,
                        exog=exog_scaled,
                        order=(2, d_order, 2),
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_fit = model.fit(disp=False)

                    # 4. Geração de exógenas futuras
                    df_future_exog = pd.DataFrame({'Data': future_dates_app})
                    exog_future_df = create_seasonal_features_for_streamlit(df_future_exog, date_col_name='Data')
                    exog_future_scaled = scaler_exog.transform(exog_future_df)

                    # 5. Previsão
                    forecast_log = model_fit.forecast(steps=periodo_previsao_app, exog=exog_future_scaled)
                    df_forecast_display_app[col_name_plot_app] = np.exp(forecast_log.values)

                    # 6. Cálculo de métricas
                    y_true = df_historical_10a_app.set_index('Data')['Value'].reindex(future_dates_app).dropna()
                    y_pred = df_forecast_display_app.set_index('Data').reindex(y_true.index)[col_name_plot_app]

                    if not y_true.empty and not y_pred.isnull().all():
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = mean_squared_error(y_true, y_pred, squared=False)
                        smape_val = smape(y_true.values, y_pred.values)

                        new_row = pd.DataFrame([{
                            'modelo': 'SARIMAX',
                            'MAE': mae,
                            'RMSE': rmse,
                            'sMAPE': smape_val,
                            'data_geracao': pd.Timestamp.now()
                        }])

                        file_path = 'df_erros.csv'
                        if os.path.exists(file_path):
                            df_existing = pd.read_csv(file_path)
                            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                        else:
                            df_updated = new_row

                        df_updated.to_csv(file_path, index=False)
                        st.info("Métricas de erro SARIMAX salvas com sucesso.")

                except Exception as e:
                    st.error(f"Erro ao processar SARIMAX: {e}")
                    df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
        # ------------------ LSTM HÍBRIDO ------------------
        elif modelo_escolhido_app == "LSTM Híbrido":
            st.subheader(f"🔮 Previsão com LSTM Híbrido para {periodo_previsao_app} dias")
            col_name_plot_app = 'Previsão LSTM Híbrido'
            if lstm_model_app is not None and scaler_lstm_app is not None:
                with st.spinner("Realizando previsão com LSTM Híbrido..."):
                    try:
                        seq_length = 60
                        num_features = 6  # y + 5 sazonais

                        df_tail = series_data_for_models_app.tail(seq_length).reset_index()
                        df_tail.rename(columns={'Value': 'y', 'Data': 'ds'}, inplace=True)

                        seasonal_features = create_seasonal_features_for_streamlit(df_tail, date_col_name='ds')
                        df_features = pd.concat([df_tail[['y']], seasonal_features], axis=1)

                        if df_features.shape[0] < seq_length:
                            st.error(f"Menos de {seq_length} pontos disponíveis para o LSTM.")
                            df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
                            st.stop()

                        scaled_sequence = scaler_lstm_app.transform(df_features.values)
                        current_seq = scaled_sequence.reshape((1, seq_length, num_features))
                        lstm_preds_scaled = []

                        for i in range(periodo_previsao_app):
                            pred_scaled = lstm_model_app.predict(current_seq, verbose=0)[0, 0]
                            lstm_preds_scaled.append(pred_scaled)

                            temp_input = np.zeros((1, num_features))
                            temp_input[0, 0] = pred_scaled
                            pred_denorm = scaler_lstm_app.inverse_transform(temp_input)[0, 0]

                            next_date = last_hist_date_app + timedelta(days=i + 1)
                            df_next = pd.DataFrame({'Data_temp': [next_date]})
                            seasonal_next = create_seasonal_features_for_streamlit(df_next, date_col_name='Data_temp')

                            next_features = np.zeros(num_features)
                            next_features[0] = pred_denorm
                            next_features[1:] = seasonal_next.values.flatten()

                            scaled_next_step = scaler_lstm_app.transform(next_features.reshape(1, -1))
                            next_input = scaled_next_step.reshape((1, 1, num_features))
                            current_seq = np.append(current_seq[:, 1:, :], next_input, axis=1)

                        all_preds = np.zeros((len(lstm_preds_scaled), num_features))
                        all_preds[:, 0] = lstm_preds_scaled
                        df_forecast_display_app[col_name_plot_app] = scaler_lstm_app.inverse_transform(all_preds)[:, 0]

                        # ------------------ Métricas LSTM ------------------
                        y_true = df_historical_10a_app.set_index('Data')['Value'].reindex(future_dates_app).dropna()
                        y_pred = df_forecast_display_app.set_index('Data').reindex(y_true.index)[col_name_plot_app]

                        if not y_true.empty and not y_pred.isnull().all():
                            mae = mean_absolute_error(y_true, y_pred)
                            rmse = mean_squared_error(y_true, y_pred, squared=False)
                            smape_val = smape(y_true.values, y_pred.values)

                            new_row = pd.DataFrame([{
                                'modelo': 'LSTM Híbrido',
                                'MAE': mae,
                                'RMSE': rmse,
                                'sMAPE': smape_val,
                                'data_geracao': pd.Timestamp.now()
                            }])

                            file_path = 'df_erros.csv'
                            if os.path.exists(file_path):
                                df_existing = pd.read_csv(file_path)
                                df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                            else:
                                df_updated = new_row

                            df_updated.to_csv(file_path, index=False)
                            st.info("Métricas de erro LSTM salvas com sucesso.")

                    except Exception as e:
                        st.error(f"Erro na previsão com LSTM Híbrido: {e}")
                        df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
            else:
                st.error("Modelo LSTM ou scaler não carregado.")
                df_forecast_display_app[col_name_plot_app] = [np.nan] * periodo_previsao_app
        # ------------------ Plotagem comum e exibição ------------------
        if col_name_plot_app and col_name_plot_app in df_forecast_display_app:
            fig_plot = go.Figure()
            fig_plot.add_trace(go.Scatter(
                x=df_historical_10a_app['Data'].tail(180),
                y=df_historical_10a_app['Value'].tail(180),
                name="Histórico Recente",
                line=dict(color='blue')
            ))
            fig_plot.add_trace(go.Scatter(
                x=df_forecast_display_app['Data'],
                y=df_forecast_display_app[col_name_plot_app],
                name=col_name_plot_app,
                line=dict(color='green' if "LSTM" in col_name_plot_app else 'orange', dash='dash')
            ))
            fig_plot.update_layout(
                title=f"Previsão {modelo_escolhido_app}",
                xaxis_title="Data",
                yaxis_title="Preço (US$)"
            )
            st.plotly_chart(fig_plot, use_container_width=True)

            st.write(f"📊 Valores Previstos ({col_name_plot_app}):")
            df_to_display = df_forecast_display_app[['Data', col_name_plot_app]].copy()
            df_to_display['Data'] = pd.to_datetime(df_to_display['Data'])
            df_to_display['Data'] = df_to_display['Data'].dt.strftime('%d/%m/%Y')
            df_to_display = df_to_display.set_index('Data')
            df_to_display[col_name_plot_app] = df_to_display[col_name_plot_app].round(2)
            st.dataframe(df_to_display)
        elif col_name_plot_app:
            st.error(f"Coluna de previsão '{col_name_plot_app}' não gerada.")
        else:
            st.error("Coluna de previsão não determinada.")
# ------------------ Exibição de métricas no sidebar ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Performance dos Modelos")
if not df_performance_metrics_app.empty:
    col_modelo = 'modelo' if 'modelo' in df_performance_metrics_app.columns else (
                 'Modelo' if 'Modelo' in df_performance_metrics_app.columns else None)
    if col_modelo:
        st.sidebar.dataframe(df_performance_metrics_app.set_index(col_modelo))
    else:
        st.sidebar.dataframe(df_performance_metrics_app)
        st.sidebar.warning("Coluna 'modelo'/'Modelo' não encontrada em df_erros.csv.")
else:
    st.sidebar.info("Métricas de performance ainda não registradas.")
