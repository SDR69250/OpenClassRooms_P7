# ======================== | Imports | ========================

#### GATHERING THE DATA ####

# imports
from email.policy import default
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import json
import requests

# ======================== | Page title & sidebar | ========================

st.markdown("# Customer details top 20 \U0001F50E")
st.sidebar.markdown("Customer details top 20 \U0001F50E")

# Imports data from Home page
# from Home import FLASK_URL, X_test

# URL parent du serveur Flask
# FLASK_URL = "http://127.0.0.1:5000/"
# URL parent du serveur Flask - Pythonanywhere
FLASK_URL = "https://sebastienderosa.eu.pythonanywhere.com/"

# ======================== | Interactions, API calls and decode | ========================

# API calls | GET data (used to select customer idx)
@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def load_data():
    url_data = FLASK_URL + "load_data/"
    response = requests.get(url_data)
    content = json.loads(response.content.decode('utf-8'))
    dict_data = content["data"]
    data = pd.DataFrame.from_dict(eval(dict_data), orient='columns')
    return data
data = load_data()


# GET X_test and cache it (heavy)
@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def load_X_test():
    url_X_test = FLASK_URL + "load_X_test/"
    response = requests.get(url_X_test)
    content = json.loads(response.content.decode('utf-8'))
    dict_X_test = content["X_test"]
    X_test = pd.DataFrame.from_dict(eval(dict_X_test), orient='columns')
    return X_test
X_test = load_X_test()


# Select Customer number SK_ID_CURR in data
idx = st.sidebar.selectbox(
    "Select Credit File", 
    data.SK_ID_CURR, key = "idx2")

# retrieve previously selected value from Home page
if 'idx' in st.session_state:
    idx = st.session_state['idx']
# st.write(idx)

# GET predict : prediction / prob_predict / ID_to_predict : 
url_predict_client = FLASK_URL + "predict/" + str(idx)
response = requests.get(url_predict_client)
content = json.loads(response.content.decode('utf-8'))
dict_ID_to_predict = content["ID_to_predict"]
ID_to_predict = pd.DataFrame.from_dict(eval(dict_ID_to_predict), orient='columns')

# API calls | GET top_20
url_top_20 = FLASK_URL + "load_top_20/"
response = requests.get(url_top_20)
content = json.loads(response.content.decode('utf-8'))
top_20 = content["top_20"]

#### INTERACTIONS IN THE STREAMLIT SESSION ####

# table with top 20 features of selected customer
st.write(f"customer number : {str(idx)}\n")
st.write(ID_to_predict)

# clean dataviz for top 20 features of selected customer 
def customer_details():
    # utiliser graph objects avec une boucle sur top_20
    # pour montrer uniquement des données chiffrées

    sel = top_20
    fig1 = go.Figure()

    for i, c in enumerate(sel,1):
        chaine = "Val / Var Mean :<br>" + c
        if ((i == 1) | (i == 2)):
            row = 0
            column = 1 - i%2
        elif i % 2 != 0:
            row = int(i/2)
            column = 0
        else:
            row = int((i-1)/2)
            column = 1
        fig1.add_trace(go.Indicator(
            mode = "number+delta",
            value = ID_to_predict[c].iloc[0],
            delta = {'reference': np.mean(X_test[c]),
                    'valueformat': '.0f',
                    'increasing': {'color': 'green'},
                    'decreasing': {'color': 'red'}},
            title = chaine,
            domain = {'row': row, 'column': column}))
    fig1.update_layout(
        grid = {'rows': 10, 'columns': 2, 'pattern': "independent", 'xgap' : 0.5, 'ygap' : 0.6})
    fig1.update_layout(
        autosize=False,
        width=800,
        height=1400,)
    plt.tight_layout()

    st.write(fig1)

customer_details()


