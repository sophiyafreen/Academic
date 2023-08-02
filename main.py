from datetime import date
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Crypto Coin Price Prediction Web App')

crypto = ('BTC-GBP','XMR-GBP','ETC-GBP','XRP-GBP','LTC-GBP','VET-GBP','MANA-GBP','HBAR-GBP','FIL-GBP','THETA-GBP','XTZ-GBP','TUSD-GBP','BSV-GBP','EOS-GBP','ZEC-GBP','MKR-GBP','MIOTA-GBP','WAVES-GBP','FTM-GBP','QNT-GBP','RUNE-GBP','NEO-GBP','BAT-GBP','CHZ-GBP','ZIL-GBP','LRC-GBP','BTT-GBP','ENJ-GBP','KSM-GBP','HOT1-GBP','KAVA-GBP','XEM-GBP','XDC-GBP','SNX-GBP','COTI-GBP','CRO-GBP','CVC-GBP','DGB-GBP','DIME-GBP','EDG-GBP','ENJ-GBP','EOS-GBP','ERG-GBP','ETH-GBP','FLASH-GBP','FLUX-GBP','GAME-GBP','GAS-GBP','HTML-GBP','MED-GBP','META-GBP','MINT-GBP','NEO-GBP','NMR-GBP','OBSR-GBP','ONT-GBP','OXEN-GBP','PART-GBP','PHR-GBP','PIN-GBP','QASH-GBP','QNT-GBP','QRL-GBP','REP-GBP','SLS-GBP','SNX-GBP','XCP-GBP','ZEN-GBP','ZNN-GBP','ZRX-GBP','ZYN-GBP')
selected_crypto = st.selectbox('Select dataset for prediction', crypto)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365



def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_crypto)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.head())
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open_price"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close_price"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)