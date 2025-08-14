from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Utiliser un backend non interactif pour Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import io
import base64
import os
import json
from datetime import datetime, timedelta
import warnings
from io import StringIO # Ajouté pour le re-parsing

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variables globales pour stocker les données et les modèles
global_data = None
scaler = MinMaxScaler(feature_range=(0, 1))
time_step = 60 
models = {
    'ETS': None,
    'ARIMA': None,
    'SVR': None,
    'XGBoost': None,
    'MLP': None
}
predictions = {
    'train': {},
    'test': {}
}
evaluation_metrics = {
    'train': {},
    'test': {}
}

# Fonctions auxiliaires
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_data(data_df): 
    cleaned_columns = []
    for col in data_df.columns:
        col_name = str(col)
        col_name = col_name.split('\\n')[0] 
        col_name = col_name.replace('"', '').strip()
        cleaned_columns.append(col_name)
    data_df.columns = cleaned_columns

    if 'Price' in data_df.columns and 'Close' not in data_df.columns:
        data_df.rename(columns={'Price': 'Close'}, inplace=True)

    required_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    if not all(col in data_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_df.columns]
        raise ValueError(f"Colonnes manquantes : {missing}.")

    data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
    data_df.dropna(subset=['Date'], inplace=True)
    data_df = data_df.sort_values(by='Date').reset_index(drop=True)

    for col in ['Close', 'Open', 'High', 'Low']:
        if col in data_df.columns:
            data_df[col] = data_df[col].astype(str).str.replace(',', '', regex=False)
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    data_df.dropna(subset=['Close', 'Open', 'High', 'Low'], inplace=True)

    if 'Vol.' in data_df.columns:
        def convert_volume(volume_val):
            if pd.isna(volume_val): return np.nan
            s_val = str(volume_val).strip() 
            if not s_val or s_val == '-': return np.nan
            s_upper = s_val.upper()
            if s_upper.endswith('K'): return float(s_upper.replace('K', '')) * 1000
            if s_upper.endswith('M'): return float(s_upper.replace('M', '')) * 1000000
            try: return float(s_val)
            except ValueError: return np.nan
        data_df['Vol.'] = data_df['Vol.'].apply(convert_volume)

    if 'Change %' in data_df.columns:
        data_df['Change %'] = data_df['Change %'].astype(str).str.replace('%', '', regex=False)
        data_df['Change %'] = pd.to_numeric(data_df['Change %'], errors='coerce')

    return data_df

def plot_to_base64(plt_obj):
    img = io.BytesIO()
    # Appliquer un style aux graphiques Matplotlib
    with plt.style.context('seaborn-v0_8-whitegrid'): # Utiliser un style Seaborn pour une meilleure esthétique
        plt_obj.savefig(img, format='png', bbox_inches='tight', dpi=100) # dpi pour la résolution
    
    if hasattr(plt_obj, 'close'): 
        plt_obj.close()
    else: 
        plt.close('all') 
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


def create_dataset(dataset_values, current_time_step=1):
    dataX, dataY = [], []
    if dataset_values.ndim == 1:
        dataset_values = dataset_values.reshape(-1, 1)

    for i in range(len(dataset_values) - current_time_step - 1):
        a = dataset_values[i:(i + current_time_step), 0]
        dataX.append(a)
        dataY.append(dataset_values[i + current_time_step, 0])
    if not dataX: 
        return np.array([]).reshape(0, current_time_step), np.array([])
    return np.array(dataX), np.array(dataY)


def calculate_metrics(y_true, y_pred):
    y_true_flat, y_pred_flat = np.array(y_true).flatten(), np.array(y_pred).flatten()
    min_len = min(len(y_true_flat), len(y_pred_flat))

    if min_len == 0: 
        return {'ME': np.nan, 'MAE': np.nan, 'MSE': np.nan, 
                'RMSE': np.nan, 'MPE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    y_true_aligned, y_pred_aligned = y_true_flat[:min_len], y_pred_flat[:min_len]
    
    metrics = {
        'ME': np.mean(y_true_aligned - y_pred_aligned),
        'MAE': mean_absolute_error(y_true_aligned, y_pred_aligned),
        'MSE': mean_squared_error(y_true_aligned, y_pred_aligned),
        'RMSE': np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned)),
        'R2': r2_score(y_true_aligned, y_pred_aligned)
    }
    
    mask_zero = y_true_aligned != 0
    if np.any(mask_zero):
        valid_true_for_percentage = y_true_aligned[mask_zero]
        valid_pred_for_percentage = y_pred_aligned[mask_zero]
        if valid_true_for_percentage.size > 0:
            metrics['MPE'] = np.mean((valid_true_for_percentage - valid_pred_for_percentage) / valid_true_for_percentage) * 100
            metrics['MAPE'] = mean_absolute_percentage_error(valid_true_for_percentage, valid_pred_for_percentage) * 100
        else:
            metrics['MPE'] = np.nan
            metrics['MAPE'] = np.nan
    else:
        metrics['MPE'] = np.nan
        metrics['MAPE'] = np.nan
        
    return metrics

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file_route():
    global global_data 
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            df = None
            delimiters_to_try = [',', ';', '\t']
            
            for delim in delimiters_to_try:
                try:
                    file.seek(0)
                    df_attempt = pd.read_csv(file, encoding='utf-8-sig', skipinitialspace=True, delimiter=delim)
                    if not df_attempt.empty and len(df_attempt.columns) > 1:
                        df = df_attempt
                        break 
                except pd.errors.ParserError:
                    continue 
                except Exception: 
                    continue
            
            if df is None or df.empty or len(df.columns) <= 1:
                file.seek(0)
                temp_df_one_col = pd.read_csv(file, header=None, encoding='utf-8-sig', skipinitialspace=True, keep_default_na=False, na_values=[''])
                if temp_df_one_col.shape[1] == 1:
                    series_data = temp_df_one_col.iloc[:, 0].astype(str)
                    csv_string_content = series_data.str.cat(sep='\n')
                    if not csv_string_content.strip():
                        raise ValueError("CSV content for re-parsing is empty.")
                    
                    df_reparsed = pd.read_csv(StringIO(csv_string_content), skipinitialspace=True)
                    
                    if not df_reparsed.empty and len(df_reparsed.columns) > 1:
                        df = df_reparsed
                    else:
                        raise ValueError("Failed to parse CSV after notebook-like re-parsing attempt. Check file structure and quoting.")
                else:
                    raise ValueError("Failed to parse CSV. File structure is not recognized or is empty, even when read as a single column.")

            if df is None or df.empty:
                 raise ValueError("Failed to parse CSV with any attempted method.")

            global_data = clean_data(df.copy())
            
            data_info = {
                'columns': global_data.columns.tolist(),
                'start_date': str(global_data['Date'].min().date()),
                'end_date': str(global_data['Date'].max().date()),
                'num_rows': len(global_data),
                'descriptive_stats': json.loads(global_data['Close'].describe().to_json())
            }
            
            return jsonify({
                'message': 'Fichier téléversé et traité avec succès', 
                'data_info': data_info
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/data/overview')
def data_overview():
    if global_data is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    stats = {
        'close': json.loads(global_data['Close'].describe().to_json()) if 'Close' in global_data else None,
        'open': json.loads(global_data['Open'].describe().to_json()) if 'Open' in global_data else None,
        'high': json.loads(global_data['High'].describe().to_json()) if 'High' in global_data else None,
        'low': json.loads(global_data['Low'].describe().to_json()) if 'Low' in global_data else None,
        'volume': json.loads(global_data['Vol.'].describe().to_json()) if 'Vol.' in global_data.columns and not global_data['Vol.'].empty else None,
        'change': json.loads(global_data['Change %'].describe().to_json()) if 'Change %' in global_data.columns and not global_data['Change %'].empty else None
    }
    
    # Améliorations graphiques pour l'évolution du prix de clôture
    plt.style.use('seaborn-v0_8-darkgrid') # Style plus moderne
    fig_close_price = plt.figure(figsize=(12, 6)) # Taille ajustée
    ax_close_price = fig_close_price.add_subplot(1, 1, 1) 
    
    ax_close_price.plot(global_data['Date'], global_data['Close'], label='Prix de Clôture', color='#00796B', linewidth=2) # Couleur principale
    ax_close_price.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax_close_price.set_ylabel('Prix de Clôture', fontsize=14, fontweight='bold')
    ax_close_price.set_title('Évolution du Prix de Clôture', fontsize=18, fontweight='bold', color='#004D40')
    
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.yticks(fontsize=10)
    ax_close_price.legend(fontsize=12, loc='upper left')
    ax_close_price.grid(True, linestyle='--', alpha=0.7, color='#cccccc') # Grille plus subtile
    ax_close_price.tick_params(colors='#333333') # Couleur des ticks
    fig_close_price.tight_layout(pad=1.5)
    plot_url = plot_to_base64(fig_close_price) 
    
    return jsonify({
        'stats': stats,
        'close_price_plot': plot_url
    })

@app.route('/analysis/stationarity')
def stationarity_analysis():
    if global_data is None or 'Close' not in global_data.columns:
        return jsonify({'error': 'Aucune donnée ou colonne "Close" non trouvée'}), 400
    
    adf_series = global_data['Close'].dropna()
    if adf_series.empty:
        return jsonify({'error': 'La colonne "Close" ne contient aucune donnée valide pour les tests de stationnarité'}), 400

    adf_result = adfuller(adf_series)
    adf_output = {
        'statistic': adf_result[0],
        'pvalue': adf_result[1],
        'critical_values': adf_result[4],
        'result': 'non stationnaire' if adf_result[1] > 0.05 else 'stationnaire'
    }
    
    try:
        kpss_result = kpss(adf_series, regression='c', nlags='auto')
        kpss_output = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'critical_values': kpss_result[3],
            'result': 'non stationnaire' if kpss_result[1] < 0.05 else 'stationnaire'
        }
    except Exception as e:
        kpss_output = {'error': str(e)}
    
    global_data['Close_Diff'] = global_data['Close'].diff()
    adf_series_diff = global_data['Close_Diff'].dropna()
    
    acf_pacf_plot_url = None 
    adf_diff_output = {'error': 'La série différenciée est vide ou trop courte pour le test ADF'}

    if not adf_series_diff.empty and len(adf_series_diff) > 20 : 
        adf_result_diff = adfuller(adf_series_diff)
        adf_diff_output = {
            'statistic': adf_result_diff[0],
            'pvalue': adf_result_diff[1],
            'critical_values': adf_result_diff[4],
            'result': 'non stationnaire' if adf_result_diff[1] > 0.05 else 'stationnaire'
        }
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig_acf_pacf, axes_acf_pacf = plt.subplots(1, 2, figsize=(16, 5)) # Taille ajustée
        
        # ACF Plot
        plot_acf(adf_series_diff, ax=axes_acf_pacf[0], lags=min(40, len(adf_series_diff)//2 -1), 
                   title='Fonction d\'Autocorrélation (ACF)', color='#FF8F00', vlines_kwargs={"colors": '#FF8F00'})
        axes_acf_pacf[0].set_xlabel('Lag', fontsize=12, fontweight='bold')
        axes_acf_pacf[0].set_ylabel('Autocorrélation', fontsize=12, fontweight='bold')
        axes_acf_pacf[0].title.set_fontsize(16)
        axes_acf_pacf[0].title.set_fontweight('bold')
        axes_acf_pacf[0].grid(True, linestyle=':', alpha=0.6)

        # PACF Plot
        plot_pacf(adf_series_diff, ax=axes_acf_pacf[1], lags=min(40, len(adf_series_diff)//2 -1), 
                    title='Fonction d\'Autocorrélation Partielle (PACF)', color='#00796B', method='ywm')
        axes_acf_pacf[1].set_xlabel('Lag', fontsize=12, fontweight='bold')
        axes_acf_pacf[1].set_ylabel('Autocorrélation Partielle', fontsize=12, fontweight='bold')
        axes_acf_pacf[1].title.set_fontsize(16)
        axes_acf_pacf[1].title.set_fontweight('bold')
        axes_acf_pacf[1].grid(True, linestyle=':', alpha=0.6)
        
        fig_acf_pacf.tight_layout(pad=2.0)
        acf_pacf_plot_url = plot_to_base64(fig_acf_pacf) 
    elif not adf_series_diff.empty:
        adf_diff_output = {'error': f'La série différenciée ne contient que {len(adf_series_diff)} valeurs, trop peu pour le test ADF ou les graphiques ACF/PACF (besoin > 20).'}

    return jsonify({
        'adf_test': adf_output,
        'kpss_test': kpss_output,
        'adf_diff_test': adf_diff_output,
        'acf_pacf_plot': acf_pacf_plot_url
    })

@app.route('/analysis/correlation')
def correlation_analysis():
    if global_data is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    cols_for_corr = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'Close']
    numeric_cols_present = []
    
    for col in cols_for_corr:
        if col in global_data.columns and pd.api.types.is_numeric_dtype(global_data[col]):
            numeric_cols_present.append(col)
    
    if len(numeric_cols_present) < 2:
        return jsonify({'error': 'Pas assez de colonnes numériques pour la corrélation'}), 400
    
    correlation_matrix = global_data[numeric_cols_present].dropna().corr()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_heatmap = plt.figure(figsize=(10, 7)) # Taille ajustée
    ax_heatmap = fig_heatmap.add_subplot(1,1,1)
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, 
                ax=ax_heatmap, annot_kws={"size": 10}, cbar_kws={'shrink': .8})
    ax_heatmap.set_title("Matrice de Corrélation", fontsize=18, fontweight='bold', color='#004D40')
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    fig_heatmap.tight_layout(pad=1.5)
    heatmap_plot_url = plot_to_base64(fig_heatmap)
    
    return jsonify({
        'correlation_matrix': correlation_matrix.to_dict(),
        'heatmap_plot': heatmap_plot_url
    })

@app.route('/models/train', methods=['POST'])
def train_models_route(): 
    global models, predictions, evaluation_metrics, scaler 
    if global_data is None or 'Close' not in global_data.columns:
        return jsonify({'error': 'Aucune donnée ou colonne "Close" non trouvée'}), 400
    
    try:
        models = { 'ETS': None, 'ARIMA': None, 'SVR': None, 'XGBoost': None, 'MLP': None }
        predictions = { 'train': {}, 'test': {} }
        evaluation_metrics = { 'train': {}, 'test': {} }

        close_prices = global_data['Close'].values.reshape(-1,1)
        if np.isnan(close_prices).all():
            return jsonify({'error': 'Les données de prix de clôture sont toutes NaN après remodelage.'}), 400

        global_data['Close_Scaled'] = scaler.fit_transform(close_prices)
        
        train_size = int(len(global_data) * 0.80)
        
        train_data_scaled_ts = global_data['Close_Scaled'].iloc[0:train_size]
        test_data_scaled_ts = global_data['Close_Scaled'].iloc[train_size:len(global_data)]

        X_all, y_all = create_dataset(global_data['Close_Scaled'].values, time_step)

        if X_all.size == 0 or y_all.size == 0:
             return jsonify({'error': f'Pas assez de données pour créer un jeu de données supervisé avec time_step {time_step}. Besoin d\'au moins {time_step + 2} points de données.'}), 400
        
        num_supervised_train_samples = train_size - time_step -1
        
        if num_supervised_train_samples <=0:
             return jsonify({'error': f'La taille des données d\'entraînement ({train_size}) est trop petite pour time_step ({time_step}).'}), 400

        X_train_ml_raw, y_train_ml_raw = X_all[:num_supervised_train_samples], y_all[:num_supervised_train_samples]
        X_test_ml_raw, y_test_ml_raw = X_all[num_supervised_train_samples:], y_all[num_supervised_train_samples:]

        if X_train_ml_raw.size == 0 : 
            return jsonify({'error': 'Pas assez de données pour la transformation supervisée ML après division (ensemble d\'entraînement vide).'}), 400
        
        X_train_ml = X_train_ml_raw
        X_test_ml = X_test_ml_raw
        y_train_ml = y_train_ml_raw
        y_test_ml = y_test_ml_raw
                
        results = {}
        
        try:
            if len(train_data_scaled_ts) < 2 * (12 if 'add' in ['add', 'mul'] or 'mul' in ['add', 'mul'] else 1): 
                raise ValueError(f"Pas assez de points de données pour le modèle ETS (besoin d'au moins {2*12}, obtenu {len(train_data_scaled_ts)}).")
            ets_model = ExponentialSmoothing(train_data_scaled_ts, trend='add', seasonal=None, initialization_method='estimated') 
            fit_ets_model = ets_model.fit()
            
            models['ETS'] = fit_ets_model
            predictions['train']['ETS'] = scaler.inverse_transform(fit_ets_model.fittedvalues.values.reshape(-1,1)).flatten().tolist()
            if not test_data_scaled_ts.empty:
                forecast_ets_scaled = fit_ets_model.forecast(steps=len(test_data_scaled_ts))
                predictions['test']['ETS'] = scaler.inverse_transform(forecast_ets_scaled.values.reshape(-1,1)).flatten().tolist()
            else:
                predictions['test']['ETS'] = []
            results['ETS'] = {'status': 'success', 'summary': "Modèle ETS entraîné."} 
        except Exception as e:
            results['ETS'] = {'status': 'error', 'message': str(e)}
        
        try:
            if len(train_data_scaled_ts.dropna()) < 20: 
                raise ValueError(f"Pas assez de points de données pour le modèle ARIMA (besoin d'au moins 20, obtenu {len(train_data_scaled_ts.dropna())}).")
            auto_model_fit = auto_arima(train_data_scaled_ts.dropna(),
                                    start_p=1, start_q=1, test='adf', max_p=3, max_q=3,
                                    m=1, d=None, seasonal=False, start_P=0, D=0, trace=False,
                                    error_action='ignore', suppress_warnings=True, stepwise=True)
            final_arima_model = SARIMAX(train_data_scaled_ts.dropna(), order=auto_model_fit.order,
                                        seasonal_order=auto_model_fit.seasonal_order if hasattr(auto_model_fit, 'seasonal_order') else (0,0,0,0),
                                        enforce_stationarity=False, enforce_invertibility=False)
            fit_arima_model = final_arima_model.fit(disp=False)
            models['ARIMA'] = fit_arima_model
            predictions['train']['ARIMA'] = scaler.inverse_transform(fit_arima_model.fittedvalues.values.reshape(-1,1)).flatten().tolist()
            if not test_data_scaled_ts.empty:
                forecast_arima_scaled = fit_arima_model.get_forecast(steps=len(test_data_scaled_ts)).predicted_mean
                predictions['test']['ARIMA'] = scaler.inverse_transform(forecast_arima_scaled.values.reshape(-1,1)).flatten().tolist()
            else:
                predictions['test']['ARIMA'] = []
            results['ARIMA'] = {'status': 'success', 'order': auto_model_fit.order, 'summary': "Modèle ARIMA entraîné." }
        except Exception as e:
            results['ARIMA'] = {'status': 'error', 'message': str(e)}
        
        n_splits_cv = min(3, len(X_train_ml) // (time_step if time_step > 0 else 1) ) 
        if n_splits_cv < 2 and len(X_train_ml) > time_step : 
             n_splits_cv = 2 
        elif len(X_train_ml) <= time_step or n_splits_cv <2 : 
            msg_ml_error = f'Pas assez d\'échantillons dans X_train_ml ({len(X_train_ml)}) pour TimeSeriesSplit avec n_splits={n_splits_cv} et time_step={time_step}.'
            results['SVR'] = {'status': 'error', 'message': msg_ml_error}
            results['XGBoost'] = {'status': 'error', 'message': msg_ml_error}
            results['MLP'] = {'status': 'error', 'message': msg_ml_error}
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits_cv)
            try:
                svr_model = SVR()
                svr_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
                svr_grid = GridSearchCV(svr_model, svr_params, cv=tscv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
                svr_grid.fit(X_train_ml, y_train_ml)
                best_svr = svr_grid.best_estimator_
                models['SVR'] = best_svr
                predictions['train']['SVR'] = scaler.inverse_transform(best_svr.predict(X_train_ml).reshape(-1,1)).flatten().tolist()
                if X_test_ml.size > 0:
                     predictions['test']['SVR'] = scaler.inverse_transform(best_svr.predict(X_test_ml).reshape(-1,1)).flatten().tolist()
                else:
                    predictions['test']['SVR'] = []
                results['SVR'] = {'status': 'success', 'best_params': svr_grid.best_params_, 'score': svr_grid.best_score_ }
            except Exception as e:
                results['SVR'] = {'status': 'error', 'message': str(e)}
            
            try:
                xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
                xgb_params = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
                xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=tscv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
                xgb_grid.fit(X_train_ml, y_train_ml)
                best_xgb = xgb_grid.best_estimator_
                models['XGBoost'] = best_xgb
                predictions['train']['XGBoost'] = scaler.inverse_transform(best_xgb.predict(X_train_ml).reshape(-1,1)).flatten().tolist()
                if X_test_ml.size > 0:
                    predictions['test']['XGBoost'] = scaler.inverse_transform(best_xgb.predict(X_test_ml).reshape(-1,1)).flatten().tolist()
                else:
                    predictions['test']['XGBoost'] = []
                results['XGBoost'] = {'status': 'success', 'best_params': xgb_grid.best_params_, 'score': xgb_grid.best_score_}
            except Exception as e:
                results['XGBoost'] = {'status': 'error', 'message': str(e)}

            try:
                mlp_model = MLPRegressor(random_state=42, early_stopping=True, max_iter=300)
                mlp_params = { 'hidden_layer_sizes': [(50,), (50,25)], 'activation': ['relu', 'tanh'],
                               'alpha': [0.0001, 0.001], 'learning_rate_init': [0.001, 0.01] }
                mlp_grid = GridSearchCV(mlp_model, mlp_params, cv=tscv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
                mlp_grid.fit(X_train_ml, y_train_ml)
                best_mlp = mlp_grid.best_estimator_
                models['MLP'] = best_mlp
                predictions['train']['MLP'] = scaler.inverse_transform(best_mlp.predict(X_train_ml).reshape(-1,1)).flatten().tolist()
                if X_test_ml.size > 0:
                    predictions['test']['MLP'] = scaler.inverse_transform(best_mlp.predict(X_test_ml).reshape(-1,1)).flatten().tolist()
                else:
                    predictions['test']['MLP'] = []
                results['MLP'] = {'status': 'success', 'best_params': mlp_grid.best_params_, 'score': mlp_grid.best_score_}
            except Exception as e:
                results['MLP'] = {'status': 'error', 'message': str(e)}

        y_train_actual_ts = scaler.inverse_transform(train_data_scaled_ts.values.reshape(-1,1)).flatten()
        y_test_actual_ts = scaler.inverse_transform(test_data_scaled_ts.values.reshape(-1,1)).flatten()
        y_train_actual_ml = scaler.inverse_transform(y_train_ml.reshape(-1,1)).flatten() if y_train_ml.size > 0 else np.array([])
        y_test_actual_ml = scaler.inverse_transform(y_test_ml.reshape(-1,1)).flatten() if y_test_ml.size > 0 else np.array([])

        for model_name_eval in models.keys(): 
            if model_name_eval in predictions['train'] and predictions['train'][model_name_eval] and len(predictions['train'][model_name_eval]) > 0:
                actual_train_target = y_train_actual_ts
                if model_name_eval == 'ARIMA' and models['ARIMA']:
                    fitted_values_len = len(predictions['train'][model_name_eval])
                    if len(train_data_scaled_ts.dropna()) >= fitted_values_len:
                         actual_train_target_arima = scaler.inverse_transform(train_data_scaled_ts.dropna().iloc[-fitted_values_len:].values.reshape(-1,1)).flatten()
                         if len(actual_train_target_arima) == len(predictions['train'][model_name_eval]): actual_train_target = actual_train_target_arima
                         else: actual_train_target = y_train_actual_ts[-len(predictions['train'][model_name_eval]):] if len(y_train_actual_ts) >= len(predictions['train'][model_name_eval]) else y_train_actual_ts
                elif model_name_eval not in ['ETS', 'ARIMA']: actual_train_target = y_train_actual_ml
                
                if len(actual_train_target) > 0 and len(actual_train_target) == len(predictions['train'][model_name_eval]):
                    evaluation_metrics['train'][model_name_eval] = calculate_metrics(actual_train_target, predictions['train'][model_name_eval])
            
            if model_name_eval in predictions['test'] and predictions['test'][model_name_eval] and len(predictions['test'][model_name_eval]) > 0:
                actual_test_target = y_test_actual_ts if model_name_eval in ['ETS', 'ARIMA'] else y_test_actual_ml
                if actual_test_target.size > 0 and len(actual_test_target) == len(predictions['test'][model_name_eval]):
                    evaluation_metrics['test'][model_name_eval] = calculate_metrics(actual_test_target, predictions['test'][model_name_eval])
        
        return jsonify({'status': 'success', 'results': results, 'metrics': evaluation_metrics }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route('/models/evaluate')
def evaluate_models_route():
    if not evaluation_metrics or (not evaluation_metrics.get('train') and not evaluation_metrics.get('test')):
        return jsonify({'error': 'Aucun modèle entraîné ou évalué pour le moment'}), 400
    plots = {}
    metrics_to_plot_bar = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    plt.style.use('seaborn-v0_8-darkgrid') # Appliquer un style aux graphiques d'évaluation

    for metric in metrics_to_plot_bar:
        metric_present_in_test = any(metric in evaluation_metrics.get('test', {}).get(model, {}) for model in evaluation_metrics.get('test', {}))
        if not metric_present_in_test: continue
        
        fig_metric_bar = plt.figure(figsize=(12, 7))
        ax_metric_bar = fig_metric_bar.add_subplot(1,1,1)
        train_values_plot, test_values_plot, model_names_for_plot_bar = [], [], []
        sorted_model_names = sorted(evaluation_metrics.get('test', {}).keys())
        
        for model_name_bar in sorted_model_names:
            if metric in evaluation_metrics['test'].get(model_name_bar, {}):
                model_names_for_plot_bar.append(model_name_bar)
                test_values_plot.append(evaluation_metrics['test'][model_name_bar][metric])
                train_values_plot.append(evaluation_metrics.get('train', {}).get(model_name_bar, {}).get(metric, np.nan))
        
        if not model_names_for_plot_bar: plt.close(fig_metric_bar); continue
        
        x_pos = np.arange(len(model_names_for_plot_bar)); width_bar = 0.35
        
        bar1 = ax_metric_bar.bar(x_pos - width_bar/2, train_values_plot, width_bar, label='Entraînement', color='#4DB6AC', alpha=0.8) # Couleur Teal clair
        bar2 = ax_metric_bar.bar(x_pos + width_bar/2, test_values_plot, width_bar, label='Test', color='#FF8F00', alpha=0.8) # Couleur Orange accent
        
        ax_metric_bar.set_xlabel('Modèles', fontsize=14, fontweight='bold')
        ax_metric_bar.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax_metric_bar.set_title(f'Comparaison des Modèles: {metric}', fontsize=18, fontweight='bold', color='#004D40')
        ax_metric_bar.set_xticks(x_pos)
        ax_metric_bar.set_xticklabels(model_names_for_plot_bar, rotation=45, ha='right', fontsize=10)
        ax_metric_bar.legend(fontsize=12); 
        ax_metric_bar.grid(axis='y', linestyle='--', alpha=0.7, color='#cccccc')
        ax_metric_bar.tick_params(colors='#333333', which='both') # Couleur des ticks pour une meilleure visibilité
        fig_metric_bar.tight_layout(pad=1.5)
        plots[metric] = plot_to_base64(fig_metric_bar)

    if 'test' in predictions and global_data is not None and 'Date' in global_data.columns:
        fig_preds_comp, ax_preds_comp = plt.subplots(figsize=(16, 7)) # Taille ajustée
        train_size_plot = int(len(global_data) * 0.80)
        actual_test_dates_plot = global_data['Date'].iloc[train_size_plot:]
        actual_test_values_plot = global_data['Close'].iloc[train_size_plot:]
        min_plot_len = float('inf')
        if not actual_test_values_plot.empty: min_plot_len = len(actual_test_values_plot)
        
        for model_name_comp, preds_list_comp in predictions['test'].items():
            if preds_list_comp and len(preds_list_comp) > 0: min_plot_len = min(min_plot_len, len(preds_list_comp))
        
        if min_plot_len == float('inf') or min_plot_len == 0 : plt.close(fig_preds_comp)
        else:
            ax_preds_comp.plot(actual_test_dates_plot.iloc[:min_plot_len], actual_test_values_plot.iloc[:min_plot_len], 
                               label='Valeurs Réelles (Test)', color='black', linewidth=2.5, linestyle='--')
            
            # Couleurs distinctes pour les prédictions
            prediction_colors = ['#00796B', '#FF8F00', '#1E88E5', '#D81B60', '#8E24AA'] # Teal, Orange, Blue, Pink, Purple
            color_idx = 0
            for model_name_comp, preds_list_comp in predictions['test'].items():
                if preds_list_comp and len(preds_list_comp) > 0:
                    ax_preds_comp.plot(actual_test_dates_plot.iloc[:min(len(preds_list_comp), min_plot_len)], 
                                       preds_list_comp[:min(len(preds_list_comp), min_plot_len)], 
                                       label=f'Prédictions {model_name_comp}', 
                                       color=prediction_colors[color_idx % len(prediction_colors)], 
                                       alpha=0.85, linewidth=1.8)
                    color_idx += 1
            
            ax_preds_comp.set_title('Comparaison des Prédictions sur l\'Ensemble de Test', fontsize=18, fontweight='bold', color='#004D40'); 
            ax_preds_comp.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax_preds_comp.set_ylabel('Prix de Clôture', fontsize=14, fontweight='bold'); 
            ax_preds_comp.legend(loc='best', fontsize=12) # 'best' pour un meilleur placement automatique
            fig_preds_comp.autofmt_xdate(rotation=45, ha='right'); 
            ax_preds_comp.grid(True, linestyle=':', alpha=0.7, color='#cccccc')
            ax_preds_comp.tick_params(colors='#333333', which='both')
            fig_preds_comp.tight_layout(pad=1.5); 
            plots['predictions_comparison'] = plot_to_base64(fig_preds_comp)
    return jsonify({'metrics': evaluation_metrics, 'plots': plots })

@app.route('/models/predict', methods=['POST'])
def make_prediction_route(): 
    if not any(m is not None for m in models.values()): return jsonify({'error': 'Aucun modèle entraîné pour le moment'}), 400
    data_req_pred = request.json; model_name_req_pred = data_req_pred.get('model'); steps_req_pred = data_req_pred.get('steps', 1)
    if model_name_req_pred not in models or models[model_name_req_pred] is None: return jsonify({'error': f'Modèle {model_name_req_pred} non trouvé ou non entraîné'}), 404
    try:
        future_predictions_list_res = []
        if model_name_req_pred == 'ETS':
            forecast_scaled_pred = models['ETS'].forecast(steps=steps_req_pred)
            future_predictions_list_res = scaler.inverse_transform(forecast_scaled_pred.values.reshape(-1,1)).flatten().tolist()
        elif model_name_req_pred == 'ARIMA':
            forecast_scaled_pred = models['ARIMA'].get_forecast(steps=steps_req_pred).predicted_mean
            future_predictions_list_res = scaler.inverse_transform(forecast_scaled_pred.values.reshape(-1,1)).flatten().tolist()
        else: 
            if global_data is None or 'Close_Scaled' not in global_data.columns: return jsonify({'error': 'Aucune donnée disponible pour les prédictions ML'}), 400
            last_observations_scaled_pred = global_data['Close_Scaled'].values[-time_step:].reshape(1, -1) 
            current_model_pred = models[model_name_req_pred]
            for _ in range(steps_req_pred):
                pred_scaled_single_pred = current_model_pred.predict(last_observations_scaled_pred)[0]
                future_predictions_list_res.append(float(scaler.inverse_transform([[pred_scaled_single_pred]])[0][0]))
                new_obs_reshaped_pred = np.array([[pred_scaled_single_pred]]) 
                last_observations_scaled_pred = np.append(last_observations_scaled_pred[:, 1:], new_obs_reshaped_pred, axis=1)
        last_date_in_data_pred = global_data['Date'].iloc[-1]
        prediction_dates_list_res = [str((last_date_in_data_pred + timedelta(days=i+1)).date()) for i in range(steps_req_pred)]
        return jsonify({'model': model_name_req_pred, 'predictions': future_predictions_list_res, 'dates': prediction_dates_list_res }), 200
    except Exception as e: return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route('/data/export', methods=['POST'])
def export_data_route(): 
    if global_data is None: return jsonify({'error': 'Aucune donnée à exporter'}), 400
    data_req_export = request.json; format_type_req_export = data_req_export.get('format', 'csv')
    include_predictions_req_export = data_req_export.get('include_predictions', False)
    current_time_step_export = time_step 
    try:
        export_df_res = global_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(export_df_res['Date']): export_df_res['Date'] = pd.to_datetime(export_df_res['Date'])
        export_df_res['Date'] = export_df_res['Date'].dt.strftime('%Y-%m-%d')
        if include_predictions_req_export and predictions:
            train_size_export = int(len(export_df_res) * 0.80)
            for model_name_iter_export in predictions.get('train', {}):
                train_preds_export = predictions['train'].get(model_name_iter_export, [])
                if train_preds_export:
                    pred_col_name_train_export = f'Pred_{model_name_iter_export}_Train'
                    start_idx_train_export = 0 
                    if model_name_iter_export not in ['ETS', 'ARIMA']: start_idx_train_export = current_time_step_export 
                    temp_series = pd.Series([np.nan] * len(export_df_res), index=export_df_res.index)
                    end_idx_actual_preds = start_idx_train_export + len(train_preds_export)
                    if end_idx_actual_preds <= len(export_df_res) and start_idx_train_export < len(export_df_res) :
                        temp_series.iloc[start_idx_train_export:end_idx_actual_preds] = train_preds_export
                    elif start_idx_train_export < len(export_df_res): 
                        can_fit = len(export_df_res) - start_idx_train_export
                        temp_series.iloc[start_idx_train_export:] = train_preds_export[:can_fit]
                    export_df_res[pred_col_name_train_export] = temp_series
            for model_name_iter_export in predictions.get('test', {}):
                test_preds_export = predictions['test'].get(model_name_iter_export, [])
                if test_preds_export:
                    pred_col_name_test_export = f'Pred_{model_name_iter_export}_Test'
                    start_idx_test_export = train_size_export
                    temp_series_test = pd.Series([np.nan] * len(export_df_res), index=export_df_res.index)
                    end_idx_actual_test_preds = start_idx_test_export + len(test_preds_export)
                    if end_idx_actual_test_preds <= len(export_df_res) and start_idx_test_export < len(export_df_res):
                        temp_series_test.iloc[start_idx_test_export:end_idx_actual_test_preds] = test_preds_export
                    elif start_idx_test_export < len(export_df_res): 
                        can_fit_test = len(export_df_res) - start_idx_test_export
                        temp_series_test.iloc[start_idx_test_export:] = test_preds_export[:can_fit_test]
                    export_df_res[pred_col_name_test_export] = temp_series_test
        export_df_res = export_df_res.drop(columns=['Close_Scaled', 'Close_Diff'], errors='ignore')
        
        if format_type_req_export == 'csv':
            output_csv = io.StringIO(); export_df_res.to_csv(output_csv, index=False, date_format='%Y-%m-%d')
            output_csv.seek(0)
            return send_file(io.BytesIO(output_csv.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='tunindex_analyse.csv')
        elif format_type_req_export == 'excel':
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer_excel: export_df_res.to_excel(writer_excel, index=False, sheet_name='AnalyseData')
            output_excel.seek(0)
            return send_file(output_excel, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='tunindex_analyse.xlsx')
        elif format_type_req_export == 'pdf':
            # La génération PDF réelle n'est pas implémentée.
            # Retourner un message d'erreur ou un CSV comme solution de repli.
            return jsonify({'error': 'Exportation PDF non encore implémentée. Veuillez choisir CSV ou Excel.'}), 501 # Not Implemented

        else: return jsonify({'error': 'Format d\'exportation non supporté'}), 400
    except Exception as e: return jsonify({'error': f'Échec de l\'exportation: {str(e)}', 'type': type(e).__name__}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)