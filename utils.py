import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from datetime import timedelta

@st.cache_data
def load_historical_data(file_path='C:/Users/ReDragon/Desktop/TECH_CHALLENTE_F4/dados/preco_petroleo.csv', use_last_n_years=10):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Arquivo CSV '{file_path}' não encontrado.")
        return pd.DataFrame(columns=['Data', 'Value'])
    except Exception as e:
        st.error(f"Erro ao ler o CSV '{file_path}': {e}")
        return pd.DataFrame(columns=['Data', 'Value'])

    renamed_data = False; renamed_value = False
    if 'Data' in df.columns: renamed_data = True
    elif 'Date' in df.columns: df.rename(columns={'Date': 'Data'}, inplace=True); renamed_data = True
    elif len(df.columns) > 0 and df.columns[0].lower() in ['data', 'date', 'ds']: df.rename(columns={df.columns[0]: 'Data'}, inplace=True); renamed_data = True
    if 'Value' in df.columns: renamed_value = True
    elif 'Preço - petróleo bruto - Brent (FOB)' in df.columns: df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Value'}, inplace=True); renamed_value = True
    elif len(df.columns) > 1 and df.columns[1].lower() in ['value', 'preço', 'preco']: df.rename(columns={df.columns[1]: 'Value'}, inplace=True); renamed_value = True
    if not renamed_data or not renamed_value:
        st.error(f"Colunas 'Data' e/ou 'Value' não encontradas no CSV. Colunas: {df.columns.tolist()}")
        return pd.DataFrame(columns=['Data', 'Value'])
    try:
        df['Data'] = pd.to_datetime(df['Data'])
    except Exception as e:
        st.error(f"Erro ao converter 'Data' para datetime: {e}."); return pd.DataFrame(columns=['Data', 'Value'])
            
    df.sort_values('Data', inplace=True, ignore_index=True)

    if use_last_n_years is not None and use_last_n_years > 0:
        data_final_historico_util = df['Data'].max()
        if pd.notna(data_final_historico_util):
            data_inicio_n_anos_util = data_final_historico_util - pd.DateOffset(years=use_last_n_years)
            df_filtrado = df[df['Data'] >= data_inicio_n_anos_util].copy()
            if df_filtrado.empty:
                st.warning(f"DataFrame vazio após filtrar pelos últimos {use_last_n_years} anos.")
                return pd.DataFrame(columns=['Data', 'Value']) 
            df = df_filtrado
        else: st.warning("Não foi possível determinar data final para filtrar por anos.")
    
    df_indexed = df.set_index('Data'); df_resampled = df_indexed.asfreq('D') 
    df_resampled['Value'] = df_resampled['Value'].ffill().bfill(); df_resampled.reset_index(inplace=True) 
    if df_resampled.empty: st.warning("DataFrame resultante vazio.")
    return df_resampled[['Data', 'Value']]

def calcula_erro(predicao, real, modelo_nome): # Função como antes
    def symetric_mean_absolute_percentage_error(actual, predicted) -> float:
        actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
        mask = ~ (np.isnan(actual) | np.isinf(actual) | np.isnan(predicted) | np.isinf(predicted))
        actual, predicted = actual[mask], predicted[mask]
        if len(actual) == 0: return np.nan
        denominator = (np.abs(predicted) + np.abs(actual)) / 2.0; diff = np.abs(predicted - actual)
        smape_values = np.where(denominator == 0, 0.0, diff / denominator)
        return round(np.mean(smape_values) * 100, 2)
    real_flat = np.array(real).flatten(); predicao_flat = np.array(predicao).flatten()
    min_len = min(len(real_flat), len(predicao_flat))
    if min_len == 0: retorno = {'modelo': modelo_nome, 'mae': np.nan, 'rmse': np.nan, 'smape %': np.nan}
    else:
        real_flat, predicao_flat = real_flat[:min_len], predicao_flat[:min_len]
        mae = mean_absolute_error(real_flat, predicao_flat); rmse = np.sqrt(mean_squared_error(real_flat, predicao_flat))
        smape = symetric_mean_absolute_percentage_error(real_flat, predicao_flat)
        retorno = {'modelo': modelo_nome, 'mae': round(mae,3), 'rmse': round(rmse,3), 'smape %': smape}
    return retorno

def test_stationarity(series_to_test): # Função como antes
    series_clean = series_to_test.dropna(); return False if series_clean.empty else adfuller(series_clean)[1] <= 0.05

# --- Classes do Pipeline (Como antes, necessárias para reversão SARIMA/StatsForecast) ---
class PrepareData(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='Data', value_col='Value'): self.date_col, self.value_col = date_col, value_col
    def fit(self, df, y=None): return self
    def transform(self, df):
        df_copy = df.copy()
        if 'Date' in df_copy.columns and self.date_col == 'Data': df_copy.rename(columns={'Date': 'Data'}, inplace=True)
        if 'Preço - petróleo bruto - Brent (FOB)' in df_copy.columns and self.value_col == 'Value': df_copy.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Value'}, inplace=True)
        if self.date_col not in df_copy.columns: raise ValueError(f"'{self.date_col}' não encontrada.")
        if self.value_col not in df_copy.columns: raise ValueError(f"'{self.value_col}' não encontrada.")
        df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])
        df_copy.set_index(self.date_col, inplace=True); df_copy[self.value_col] = df_copy[self.value_col].astype(float)
        df_copy.dropna(subset=[self.value_col], inplace=True); df_copy.sort_index(inplace=True, ascending=True)
        return df_copy

class FillNANValues(BaseEstimator, TransformerMixin):
    def __init__(self, value_col='Value', new_value_col='y', new_date_col='ds'):
        self.value_col, self.new_value_col, self.new_date_col = value_col, new_value_col, new_date_col
    def fit(self, df, y=None): return self
    def transform(self, df):
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex): raise ValueError("DatetimeIndex esperado.")
        start_date, end_date = df_copy.index.min(), df_copy.index.max()
        if pd.isna(start_date) or pd.isna(end_date) or df_copy.empty: return pd.DataFrame(columns=[self.new_date_col, self.new_value_col])
        df_copy = df_copy.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
        df_copy[self.value_col].ffill(inplace=True); df_copy[self.value_col].bfill(inplace=True)
        df_copy.dropna(subset=[self.value_col], inplace=True)
        if df_copy.empty: return pd.DataFrame(columns=[self.new_date_col, self.new_value_col])
        df_copy.reset_index(inplace=True)
        df_copy.rename(columns={'index': self.new_date_col, self.value_col: self.new_value_col}, inplace=True)
        return df_copy[[self.new_date_col, self.new_value_col]]

class SomthDataIntervalValues(BaseEstimator, TransformerMixin): 
    def __init__(self, value_col='y'): self.value_col = value_col
    def fit(self, df, y=None): return self
    def transform(self, df): 
        df_indexed = df.set_index('ds').copy()
        series_to_log = df_indexed[self.value_col].apply(lambda x: x if x > 0 else 1e-5)
        df_log = np.log(series_to_log); ma_log = df_log.rolling(window=7, min_periods=1).mean() 
        df_diff = (df_log - ma_log); df_transformed = pd.DataFrame(index=df_diff.index)
        df_transformed['y_diff_log'] = df_diff; df_transformed['y_ma_log'] = ma_log
        df_transformed.dropna(subset=['y_diff_log'], inplace=True); df_transformed.reset_index(inplace=True)
        return df_transformed

# --- NOVA FUNÇÃO: Criar Features Sazonais ---
def create_seasonal_features_for_streamlit(df_with_dates_col, date_col_name='Data'):
    """
    Cria features sazonais (dia_da_semana, mes_sin, mes_cos, dia_do_ano_sin, dia_do_ano_cos)
    a partir de uma coluna de data em um DataFrame.
    Retorna um DataFrame apenas com as features sazonais criadas.
    """
    df_feat = df_with_dates_col.copy()
    # Assegurar que a coluna de data é datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col_name]):
        df_feat[date_col_name] = pd.to_datetime(df_feat[date_col_name])
    
    df_feat['dia_da_semana'] = df_feat[date_col_name].dt.dayofweek
    df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat[date_col_name].dt.month / 12)
    df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat[date_col_name].dt.month / 12)
    df_feat['dia_do_ano_sin'] = np.sin(2 * np.pi * df_feat[date_col_name].dt.dayofyear / 365.25)
    df_feat['dia_do_ano_cos'] = np.cos(2 * np.pi * df_feat[date_col_name].dt.dayofyear / 365.25)
    
    # Ordem das colunas DEVE ser a mesma usada para treinar o scaler_exog_arima
    feature_cols_order = ['dia_da_semana', 'mes_sin', 'mes_cos', 'dia_do_ano_sin', 'dia_do_ano_cos']
    return df_feat[feature_cols_order]