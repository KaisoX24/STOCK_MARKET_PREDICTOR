import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import load_model
from datetime import datetime, timedelta
from pathlib import Path
from streamlit_option_menu import option_menu

st.set_page_config(page_title='STOCK_MARKET_PREDICTOR', page_icon='💸', layout='wide')
# Load custom CSS
def load_css():
    css_file = Path('styles.css')
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# Cache the loading of the model and scaler
@st.cache_data
def load_resources():
    model = load_model('stock_prediction_model.keras')
    scaler = MinMaxScaler(feature_range=(0,1))
    return model, scaler

model, scaler = load_resources()

# Sidebar menu
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Stock Prediction"], icons=['house', 'graph-up-arrow'], menu_icon="cast", default_index=0)

if selected == "Home":
    st.title("Welcome to the Stock Market Predictor")
    st.write("""
    Use the sidebar to navigate to the Stock Prediction page. 
    On the Stock Prediction page, you can select a stock ticker, view raw data, and see various plots and predictions for the selected stock.
    """)
elif selected == "Stock Prediction":
    st.title('Stock Market Predictor')

    ticker = st.selectbox('Select a stock ticker', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')  # Ensure correct date format

    @st.cache_data
    def fetch_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    if ticker:
        try:
            # Fetch stock data
            data = fetch_stock_data(ticker, start_date, end_date)
            
            if data.empty:
                st.write(f"No data found for {ticker} in the specified date range.")
            else:
                data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)], columns=['Close'])
                data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)], columns=['Close'])

                # Dropdown to select what data to show
                data_view = st.selectbox('Select the data to view', ['None', 'Raw Data'])

                if data_view == 'Raw Data':
                    st.subheader('Raw Data')
                    st.dataframe(data, width=800, height=500)

                # Plot Adjusted Close Price
                fig1 = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Stock Price from {start_date} to {end_date}')
                fig1.update_layout(xaxis_title='Date', yaxis_title='Adjusted Close Price')
                st.plotly_chart(fig1)

                # Calculate moving averages and plot
                data['MA50'] = data.Close.rolling(50).mean()
                data['MA100'] = data.Close.rolling(100).mean()
                data['MA200'] = data.Close.rolling(200).mean()

                # Plot Price vs MA50
                fig2 = px.line(data, x=data.index, y=['Close', 'MA50'], title=f'{ticker} Price vs MA50')
                fig2.update_layout(xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig2)

                # Plot Price vs MA50 vs MA100
                fig3 = px.line(data, x=data.index, y=['Close', 'MA50', 'MA100'], title=f'{ticker} Price vs MA50 vs MA100')
                fig3.update_layout(xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig3)

                # Plot Price vs MA100 vs MA200
                fig4 = px.line(data, x=data.index, y=['Close', 'MA100', 'MA200'], title=f'{ticker} Price vs MA100 vs MA200')
                fig4.update_layout(xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig4)

                # Prepare data for prediction
                data_train_scaled = scaler.fit_transform(data_train)
                data_test_scaled = scaler.transform(data_test)

                x = []
                y = []

                for i in range(100, len(data_train_scaled)):
                    x.append(data_train_scaled[i-100:i])
                    y.append(data_train_scaled[i, 0])

                x, y = np.array(x), np.array(y)

                # Cache the prediction to avoid recomputation
                @st.cache_data
                def make_predictions(x):
                    predict = model.predict(x)
                    return predict

                predict = make_predictions(x)
                predict = scaler.inverse_transform(np.concatenate([np.zeros((predict.shape[0], data_train_scaled.shape[1] - 1)), predict], axis=1))[:, 0]
                y = scaler.inverse_transform(np.concatenate([np.zeros((y.shape[0], data_train_scaled.shape[1] - 1)), y.reshape(-1, 1)], axis=1))[:, 0]

                # Prepare data for plotting predictions
                plot_data = pd.DataFrame({
                    'Time': np.arange(len(y)),
                    'Original Price': predict,
                    'Predicted Price': y
                })

                # Plot Original Price vs Predicted Price
                fig5 = px.line(plot_data, x='Time', y=['Original Price', 'Predicted Price'], title='Original Price vs Predicted Price')
                fig5.update_layout(xaxis_title='Time', yaxis_title='Price', legend_title='Legend')
                st.plotly_chart(fig5)

                # Predict future values
                last_100_days = data_train_scaled[-100:]
                future_days = st.slider('Select number of future days to predict', min_value=1, max_value=365, value=30)
                future_predictions = []

                for _ in range(future_days):
                    x_future = np.array([last_100_days])
                    prediction = model.predict(x_future)
                    future_predictions.append(prediction[0, 0])
                    
                    # Update input for next prediction
                    last_100_days = np.roll(last_100_days, -1, axis=0)
                    last_100_days[-1] = prediction[0, 0]

                # Convert predictions back to original scale
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_predictions = scaler.inverse_transform(np.concatenate([np.zeros((future_predictions.shape[0], data_train_scaled.shape[1] - 1)), future_predictions], axis=1))[:, 0]

                # Generate future dates
                future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

                # Plot Future Predictions
                plot_data_future = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })

                fig6 = px.line(plot_data_future, x='Date', y='Predicted Price', title=f'{ticker} Future Price Prediction')
                fig6.update_layout(xaxis_title='Date', yaxis_title='Predicted Price', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig6)

        except Exception as e:
            st.write(f"Error fetching data for {ticker}: {str(e)}")
    else:
        st.write("Please select a valid stock ticker.")
