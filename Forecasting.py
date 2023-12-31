import subprocess
import numpy as np 

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(['pip', 'install', 'matplotlib'])

import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 11, 11

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Spotify Music Forecasting Web App")
    st.sidebar.title("Forecasting Web App")
    st.markdown("What's about the future of the music?")
    st.sidebar.markdown("What's the future of the music")

    @st.cache_resource()
    def load_data():
        data = pd.read_csv('data.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        data = pd.DataFrame(data)
        return data

    @st.cache_resource() 
    def recortar_serie(data, percentage=(70, 30)):
        # Ordenar la serie cronológicamente
        serie_sorted = data.sort_index()
        # Obtener el índice donde debe realizarse el recorte
        idx_70 = int(len(serie_sorted) * percentage[0] / 100)
        # Recortar la serie
        train = serie_sorted.iloc[:idx_70]
        test = serie_sorted.iloc[idx_70:]
        return train, test
    
    def plot_metrics(metrics_list, model,data, train, test, forecast):
        if 'Time Series Decomposition' in metrics_list:
            # Descomposicion de la serie temporal Energy-Danceability
            rcParams['figure.figsize'] = 11, 11
            decomposeEnergy = sm.tsa.seasonal_decompose(data['en-dan'])
            decomposeEnergy.plot()
            plt.title("Time Series Decomposition")
            st.pyplot()

        if 'Forecasting' in metrics_list:
            # Plot past
            plt.figure(figsize=(10, 7))
            plt.plot(train ,label='train data')
            # Calcular los intervalos de confianza
            #e_conf_int = forecast.conf_int()
            # Graficar la predicción como una línea
            plt.plot(forecast,color='red', label='predicted')
            plt.plot(test, label='test data')
            dates_test = test.index
            #plt.fill_between(dates_test, e_conf_int['lower en-dan'], e_conf_int['upper en-dan'],color='lightgray', alpha=0.3);
            plt.legend()
            plt.title('Serie temporal')
            st.pyplot()


    df = load_data()
    st.sidebar.subheader("Choose the Model")
    classifier = st.sidebar.selectbox(
        "Classifier", 
        ("ARIMA", "SARIMAX"))

    if classifier == 'ARIMA':
        metrics = st.sidebar.multiselect("What Graph to plot?", ('Time Series Decomposition', 'Forecasting'))

        if st.sidebar.button("Predict", key='Predict'):
            st.subheader("ARIMA Time Series Prediction")
            df = df.dropna()
            # Aplicar la transformación de Box-Cox a la columna a_transformar
            indx = df.index
            col = df['en-dan']
            lmbda_value = 0.5
            transformed_data = boxcox(col, lmbda=lmbda_value)
            df_normal = pd.Series(transformed_data, index=indx, name='col_normal')
            train, test = recortar_serie(df_normal) # CV
    
            model = sm.tsa.arima.ARIMA(train, order=((0, 1, 1)))
            result = model.fit()
            forecast = result.forecast(len(test))
    
    
            RMSE = float(format(np.sqrt(mean_squared_error(test, forecast)),'.3f'))
            MSE = mean_squared_error(test, forecast)
            MAE = mean_absolute_error(test, forecast)
            r2 = r2_score(test, forecast)
    
            st.write("RMSE: ", RMSE)
            st.write("MSE: ", MSE)
            st.write("MAE: ", MAE)
            st.write("r2: ", r2)
    
            plot_metrics(metrics, model, df_normal, train, test, forecast)
            
    if classifier == 'SARIMAX':
        metrics = st.sidebar.multiselect("What Graph to plot?", ('Time Series Decomposition', 'Forecasting'))

        if st.sidebar.button("Predict", key='Predict'):
            st.subheader("SARIMAX Time Series Prediction")
            df = df.dropna()
            # Aplicar la transformación de Box-Cox a la columna a_transformar
            indx = df.index
            col = df['en-dan']
            lmbda_value = 0.5
            transformed_data = boxcox(col, lmbda=lmbda_value)
            df_normal = pd.Series(transformed_data, index=indx, name='col_normal')
            train, test = recortar_serie(df_normal) # CV

            # Determine the seasonal order
            seasonal_order = (0, 1, 1, 4)  # SARIMA(p, d, q, seasonal_period)
            # Fit the SARIMA model
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order)
            result = model.fit()
            forecast = result.forecast(len(test))    
    
            RMSE = float(format(np.sqrt(mean_squared_error(test, forecast)),'.3f'))
            MSE = mean_squared_error(test, forecast)
            MAE = mean_absolute_error(test, forecast)
            r2 = r2_score(test, forecast)
    
            st.write("RMSE: ", RMSE)
            st.write("MSE: ", MSE)
            st.write("MAE: ", MAE)
            st.write("r2: ", r2)
    
            plot_metrics(metrics, model, df_normal, train, test, forecast)

if __name__ == '__main__':
    main() 


                     
            
