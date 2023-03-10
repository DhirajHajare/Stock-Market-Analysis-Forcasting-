import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('πΊππππ π΄πππππ π¨πππππππ ')

stocks = ('RELIANCE.NS', 'TCS.NS', 'HDFC.NS', 'MRF.NS','WIPRO.NS')
selected_stock = st.selectbox('πππ₯πππ­ πππ­ππ¬ππ­ ππ¨π« π©π«πππ’ππ­π’π¨π§', stocks)

n_years = st.slider('πππππ ππ ππππππππππ:', 1, 7)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('ππ¨πππ’π§π  πππ­π...')
data = load_data(selected_stock)
data_load_state.text('ππ¨πππ’π§π  πππ­π... ππ¨π§π!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='π»πππ πΊπππππ ππππ ππππ πΉππππ πΊπππππ', xaxis_rangeslider_visible=True)
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
st.subheader('ππ¨π«ππππ¬π­ πππ­π')
st.write(forecast.tail())

st.write(f'ππ¨π«ππππ¬π­ π©π₯π¨π­ ππ¨π« {π§_π²πππ«π¬} π²πππ«π¬')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("ππ¨π«ππππ¬π­ ππ¨π¦π©π¨π§ππ§π­π¬")
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
