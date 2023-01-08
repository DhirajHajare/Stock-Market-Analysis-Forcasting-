import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('𝑺𝒕𝒐𝒄𝒌 𝑴𝒂𝒓𝒌𝒆𝒕 𝑨𝒏𝒂𝒍𝒚𝒔𝒊𝒔 ')

stocks = ('RELIANCE.NS', 'TCS.NS', 'HDFC.NS', 'MRF.NS','WIPRO.NS')
selected_stock = st.selectbox('𝐒𝐞𝐥𝐞𝐜𝐭 𝐝𝐚𝐭𝐚𝐬𝐞𝐭 𝐟𝐨𝐫 𝐩𝐫𝐞𝐝𝐢𝐜𝐭𝐢𝐨𝐧', stocks)

n_years = st.slider('𝒀𝒆𝒂𝒓𝒔 𝒐𝒇 𝒑𝒓𝒆𝒅𝒊𝒄𝒕𝒊𝒐𝒏:', 1, 7)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('𝐋𝐨𝐚𝐝𝐢𝐧𝐠 𝐝𝐚𝐭𝐚...')
data = load_data(selected_stock)
data_load_state.text('𝐋𝐨𝐚𝐝𝐢𝐧𝐠 𝐝𝐚𝐭𝐚... 𝐝𝐨𝐧𝐞!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='𝑻𝒊𝒎𝒆 𝑺𝒆𝒓𝒊𝒆𝒔 𝒅𝒂𝒕𝒂 𝒘𝒊𝒕𝒉 𝑹𝒂𝒏𝒈𝒆 𝑺𝒍𝒊𝒅𝒆𝒓', xaxis_rangeslider_visible=True)
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
st.subheader('𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭 𝐝𝐚𝐭𝐚')
st.write(forecast.tail())

st.write(f'𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭 𝐩𝐥𝐨𝐭 𝐟𝐨𝐫 {𝐧_𝐲𝐞𝐚𝐫𝐬} 𝐲𝐞𝐚𝐫𝐬')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭 𝐜𝐨𝐦𝐩𝐨𝐧𝐞𝐧𝐭𝐬")
fig2 = m.plot_components(forecast)
st.write(fig2)

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r"C:\Users\Lenovo\Downloads\windows-8-1-wallpaper-remodeled-wallpaper-preview.jpg")


st.snow()
