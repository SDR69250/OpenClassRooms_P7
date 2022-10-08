from pyexpat import model
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from zipfile import ZipFile
import joblib
import json


# Preprocessing, Imputing, Upsampling, Model Selection, Model Evaluation
import sklearn
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import shap
import streamlit.components.v1 as components

st.markdown("# Customer details top 20 \U0001F50E")
st.sidebar.markdown("Customer details top 20 \U0001F50E")

# left_column, right_column = st.columns(2)


# retrieve session state variables
model_load = st.session_state['model_load']
best_thresh = st.session_state['best_thresh']
X_test = st.session_state['X_test']
y_test = st.session_state['y_test']
y_pred = st.session_state['y_pred']
explainer = st.session_state['explainer']
shap_values = st.session_state['shap_values']
shap_values1 = st.session_state['shap_values1']
expected_value = st.session_state['expected_value']
columns = st.session_state['columns']
data = st.session_state['data']
idx = st.session_state['idx']
feature_importance = st.session_state['feature_importance']
top_10 = st.session_state['top_10']
top_20 = st.session_state['top_20']
ID_to_predict = st.session_state['ID_to_predict']
prob_predict = st.session_state['prob_predict']


# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
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

# with left_column:
if st.sidebar.checkbox('Show Customer Details'):
    customer_details()


