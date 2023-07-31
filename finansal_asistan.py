import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import itertools
import plotly.express as px
from matplotlib.ticker import PercentFormatter
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def upload_data(file_path):

    if 'csv' in file_path:
        df = pd.read_csv(file_path)
    elif 'xlsx' in file_path:
        df = pd.read_excel(file_path)

    return df

def daily_revenue_graph(df, option=1):

    df = df.reset_index().groupby([pd.Grouper(key='Date', freq='D')]).sum()

    fig = px.line(df, x=df.index, y="Sales_Amount", title='Günlük Ciro')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    perc = round(((df.last(f'{str(int(option) * 2)}D')['Sales_Amount'].sum() - df.last(f'{option}D')['Sales_Amount'].sum()) / df.last(f'{option}D')['Sales_Amount'].sum()) * 100 - 100, 2)

    return (fig, perc)

def weekly_revenue_graph(df):

    df = df.reset_index().groupby([pd.Grouper(key='Date', freq='W')]).sum()

    fig = px.line(df, x=df.index, y='Sales_Amount', title='Haftalık Ciro')
    fig.update_xaxes(title_text='Date',
                    rangeslider_visible=True,
                    rangeselector=dict(
                            buttons=list([  
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
    return fig

def forecast(train, sl, st, ss, step=5):
    
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=sl, smoothing_trend=st, smoothing_seasonal=ss)
        
        y_pred = tes_model.forecast(step)

        return y_pred
 
def daily_forecast_graph(df):
    df = df.reset_index().groupby([pd.Grouper(key='Date', freq='D')]).sum()

    train = df[~df.index.isin(df.last('31D').index.to_list())]['Sales_Amount']
    test = df.last('31D')['Sales_Amount']

    y_test_pred = forecast(train, sl=0.8, st=0.7, ss=0.3, step=38)
    y_forecast = pd.DataFrame(np.round(y_test_pred[-7:], 0), columns=['Günlük Tahmin'], index=y_test_pred[-7:].index.date)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(train.last('30D').index, train.last('30D').values, label='Train', color='blue', linestyle='-')
    plt.plot(test.index, test.values, label='Test', color='orange', linestyle='-')
    plt.plot(y_test_pred.index, y_test_pred.values, label='Forecast', color='green', linestyle='-')
    plt.legend(loc='best')
    plt.title('7 Günlük Ciro Tahmini')

    return (fig, y_forecast)

def weekly_forecast_graph(df):
    df = df.reset_index().groupby([pd.Grouper(key='Date', freq='W')]).sum()

    train = df[~df.index.isin(df.last('4W').index.to_list())]['Sales_Amount']
    test = df.last('4W')['Sales_Amount']

    y_test_pred = forecast(train, sl=0.1, st=0.7, ss=0.3, step=8)
    y_forecast = pd.DataFrame(np.round(y_test_pred[-4:], 0), columns=['Haftalık Tahmin'], index=y_test_pred[-4:].index.date)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(train.last('4W').index, train.last('4W').values, label='Train', color='blue', linestyle='-')
    plt.plot(test.index, test.values, label='Test', color='orange', linestyle='-')
    plt.plot(y_test_pred.index, y_test_pred.values, label='Forecast', color='green', linestyle='-')

    plt.legend(loc='best')
    plt.title('4 Haftalık Ciro Tahmini')

    return (fig, y_forecast)

def financial_status_page(file_path):

    df = upload_data(file_path)

    tab1, tab2, tab3, tab4 = st.tabs(["Günlük Ciro", "Haftalık Ciro", "Günlük Tahmin", "Haftalık Tahmin"])
    with tab1:
        st.plotly_chart(daily_revenue_graph(df)[0], theme=None, use_container_width=True)
        option = st.selectbox(
    'Ciro Değişimini Görmek İstediğiniz Dönem:',
        ('Son 7 Gün', 'Son 14 Gün', 'Son 30 Gün'))
        option = re.findall(r'\d+', option)[0]
        st.write(f"Son {option} günün cirosunun önceki {option} güne göre değişimi: % {daily_revenue_graph(df, option)[1]}")
    with tab2:
        st.plotly_chart(weekly_revenue_graph(df), theme=None, use_container_width=True)
    with tab3:
        st.pyplot(daily_forecast_graph(df)[0])
        st.write("7 günlük tahmin: ")
        st.dataframe(daily_forecast_graph(df)[1])
    with tab4:
        st.pyplot(weekly_forecast_graph(df)[0])
        st.write("4 haftalık tahmin: ")
        st.dataframe(weekly_forecast_graph(df)[1])

def pareto_analysis_revenue(df, option):
    df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
    df_pareto = df.sort_values(by='Date', ascending=True)
    df_pareto.set_index('Date', inplace=True)
    df_pareto = df_pareto.last(f'{option}M')
    df_pareto_cat = df_pareto.groupby('SKU_Category').agg({'Sales_Amount': 'sum'}).reset_index().sort_values(by='Sales_Amount', ascending=False)
    df_pareto_cat['cumperc'] = df_pareto_cat.Sales_Amount.cumsum() / df_pareto_cat.Sales_Amount.sum() * 100
    df_pareto_cat2 = df_pareto_cat[df_pareto_cat.cumperc <= 50]

    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig, ax = plt.subplots()
    ax.bar(df_pareto_cat2['SKU_Category'], df_pareto_cat2['Sales_Amount'], color=color1)
    ax2 = ax.twinx()
    ax2.plot(df_pareto_cat2['SKU_Category'], df_pareto_cat2['cumperc'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    df_pareto_prod = df_pareto[df_pareto.SKU_Category.isin(df_pareto_cat2.SKU_Category.unique())].groupby('SKU').agg({'Sales_Amount': 'sum'}).reset_index().sort_values(by='Sales_Amount', ascending=False)
    df_pareto_prod['cumperc'] = df_pareto_prod.Sales_Amount.cumsum() / df_pareto.Sales_Amount.sum() * 100
    df_pareto_prod2 = df_pareto_prod[df_pareto_prod.cumperc <= 50]
    prod_num = df_pareto['SKU'].nunique()

    return (fig, df_pareto_cat, df_pareto_cat2, df_pareto_prod2, prod_num)

def pareto_analysis_quantity(df, option):
    df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
    df_pareto = df.sort_values(by='Date', ascending=True)
    df_pareto.set_index('Date', inplace=True)
    df_pareto = df_pareto.last(f'{option}M')
    df_pareto_cat = df_pareto.groupby('SKU_Category').agg({'Quantity': 'sum'}).reset_index().sort_values(by='Quantity', ascending=False)
    df_pareto_cat['cumperc'] = df_pareto_cat.Quantity.cumsum() / df_pareto_cat.Quantity.sum() * 100
    df_pareto_cat2 = df_pareto_cat[df_pareto_cat.cumperc <= 50]

    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig, ax = plt.subplots()
    ax.bar(df_pareto_cat2['SKU_Category'], df_pareto_cat2['Quantity'], color=color1)
    ax2 = ax.twinx()
    ax2.plot(df_pareto_cat2['SKU_Category'], df_pareto_cat2['cumperc'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    df_pareto_prod = df_pareto[df_pareto.SKU_Category.isin(df_pareto_cat2.SKU_Category.unique())].groupby('SKU').agg({'Quantity': 'sum'}).reset_index().sort_values(by='Quantity', ascending=False)
    df_pareto_prod['cumperc'] = df_pareto_prod.Quantity.cumsum() / df_pareto.Quantity.sum() * 100
    df_pareto_prod2 = df_pareto_prod[df_pareto_prod.cumperc <= 50]
    prod_num = df_pareto['SKU'].nunique()

    return (fig, df_pareto_cat, df_pareto_cat2, df_pareto_prod2, prod_num)

     
    

def product_analysis_page(file_path):
    df = upload_data(file_path)
    tab1, tab2 = st.tabs(["Ürünlerin Ciro Analizi", "Ürün Satış Adeti Analizi"])

    with tab1:
        option = st.selectbox("Ürünün Ciro Analizini Görmek İstediğiniz Dönem:", ('Son 1 Ay', 'Son 3 Ay', 'Son 12 Ay'))
        option = re.findall(r'\d+', option)[0]
        st.pyplot(pareto_analysis_revenue(df, option)[0])
        df_pareto_cat = pareto_analysis_revenue(df, option)[1]
        df_pareto_cat2 = pareto_analysis_revenue(df, option)[2]
        total_cat = df_pareto_cat['SKU_Category'].nunique()
        pareto_cat = df_pareto_cat2['SKU_Category'].nunique()
        st.write(f"Toplam Kategori Sayısı: {total_cat}")
        st.write(f"Satışların %50'sini oluşturan kategori sayısı: {pareto_cat}")
        st.write(f"""Pareto analizine göre, satışların %50'sini oluşturan kategorilerin toplam kategori sayısına oranı: %{int((pareto_cat/total_cat)*100)}""")
        st.dataframe(pareto_analysis_revenue(df, option)[3].iloc[:20])
        st.write(f"Satışların % {round(pareto_analysis_revenue(df, option)[3].iloc[19, 2], 2)} Ürünlerin % {round((20 / pareto_analysis_revenue(df, option)[4]) * 100, 2)} 'üne denk gelmektedir.")
    with tab2:
        option = st.selectbox("Ürünün Adet Analizini Görmek İstediğiniz Dönem:", ('Son 1 Ay', 'Son 3 Ay', 'Son 12 Ay'))
        option = re.findall(r'\d+', option)[0]
        st.pyplot(pareto_analysis_quantity(df, option)[0])
        df_pareto_cat = pareto_analysis_quantity(df, option)[1]
        df_pareto_cat2 = pareto_analysis_quantity(df, option)[2]
        total_cat = df_pareto_cat['SKU_Category'].nunique()
        pareto_cat = df_pareto_cat2['SKU_Category'].nunique()
        st.write(f"Toplam Kategori Sayısı: {total_cat}")
        st.write(f"Satışların %50'sini oluşturan kategori sayısı: {pareto_cat}")
        st.write(f"""Pareto analizine göre, satışların %50'sini oluşturan kategorilerin toplam kategori sayısına oranı: %{int((pareto_cat/total_cat)*100)}""")
        st.dataframe(pareto_analysis_quantity(df, option)[3].iloc[:20])
        st.write(f"Satışların % {round(pareto_analysis_quantity(df, option)[3].iloc[19, 2], 2)} Ürünlerin % {round((20 / pareto_analysis_quantity(df, option)[4]) * 100, 2)} 'üne denk gelmektedir.")


st.set_page_config(layout="centered")
st.title('Token Finansal Asistan :dart:')
page = st.sidebar.selectbox("Sayfalar", ["Finansal Durum", "Ürün Analizi"])
pos_image = Image.open("pos_machine.png")
st.sidebar.image(pos_image, use_column_width=True)

if page == "Finansal Durum":
    financial_status_page("ciro.xlsx")
elif page == "Ürün Analizi":
     product_analysis_page("scanner_data.csv")

