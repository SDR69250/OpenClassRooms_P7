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

st.markdown("# Home page \U0001F3E6")
st.sidebar.markdown("# Home page \U0001F3E6")

@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def load_models():
    model_load = joblib.load("../Models/model.joblib")
    best_thresh = joblib.load("../Models/best_thresh_LightGBM_NS.joblib")
    st.info("model loaded")
    return model_load, best_thresh

model_load, best_thresh = load_models()


@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def load_data():
    X_test = joblib.load("../Models/X_test.pkl")
    y_test = joblib.load("../Models/y_test.pkl")
    y_pred = joblib.load("../Models/y_pred.pkl")
    explainer = joblib.load("../Models/explainer")
    shap_values = joblib.load("../Models/shap_values.pkl")
    shap_values1 = joblib.load("../Models/shap_values1.pkl")
    expected_value = joblib.load("../Models/expected_values.pkl")
    st.info("data loaded")
    return X_test, y_test, y_pred, explainer, shap_values, shap_values1, expected_value

X_test, y_test, y_pred, explainer, shap_values, shap_values1, expected_value = load_data()

# feature names
columns = shap_values.feature_names
# data
data = pd.DataFrame(y_test, index=y_test.index).reset_index()
data["PRED"] = y_pred


# Select Customer number SK_ID_CURR in original X_test file
# idx = st.sidebar.multiselect(
#     "Select Credit File", 
#     options= data.SK_ID_CURR.values,
#     default=[416652]
# )[0]
idx = st.sidebar.selectbox(
    "Select Credit File", 
    data.SK_ID_CURR)

# Initialization
if 'idx' not in st.session_state:
    st.session_state['idx'] = idx

# Customer index in the corresponding array
data_idx = data.loc[data["SK_ID_CURR"]==idx].index[0]
# Customer data based on customer index in final X_test array
ID_to_predict = pd.DataFrame(X_test.iloc[data_idx,:]).T
# on réalise la prédiction de ID_to_predict avec le modèle 
prediction = sum((model_load.predict_proba(ID_to_predict)[:, 1]>best_thresh)*1)
prob_predict = float(model_load.predict_proba(ID_to_predict)[:, 1])

# Compute feature importance
# compute mean of absolute values for shap values
vals = np.abs(shap_values1).mean(0)
# compute feature importance as a dataframe containing vals
feature_importance = pd.DataFrame(list(zip(columns, vals)),\
    columns=['col_name','feature_importance_vals'])
# Define top 10 features for customer details
top_10 = feature_importance.sort_values(by='feature_importance_vals', ascending=False)[0:10].col_name.tolist()
# Define top 20 features for comparison vs group
top_20 = feature_importance.sort_values(by='feature_importance_vals', ascending=False)[0:20].col_name.tolist()
feat_tot = feature_importance.feature_importance_vals.sum()
feat_top = feature_importance.loc[feature_importance['col_name'].isin(top_20)].feature_importance_vals.sum()


# session state backup
st.session_state['model_load'] = model_load
st.session_state['best_thresh'] = best_thresh
st.session_state['X_test'] = X_test
st.session_state['y_test'] = y_test
st.session_state['y_pred'] = y_pred
st.session_state['explainer'] = explainer
st.session_state['shap_values'] = shap_values
st.session_state['shap_values1'] = shap_values1
st.session_state['expected_value'] = expected_value
st.session_state['columns'] = columns
st.session_state['data'] = data
st.session_state['feature_importance'] = feature_importance
st.session_state['top_10'] = top_10
st.session_state['top_20'] = top_20
st.session_state['ID_to_predict'] = ID_to_predict
st.session_state['prob_predict'] = prob_predict
st.session_state['feat_tot'] = feat_tot
st.session_state['feat_top'] = feat_top
st.session_state['data_idx'] = data_idx


# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def jauge():
    if prob_predict < best_thresh:
        title_auto = {'text':"<b>No probability of default detected</b><br>\
    <span style='color: forestgreen; font-size:0.9em'>Credit<br><b>Granted</b></span>", \
                    'font': {'color': 'forestgreen', 'size': 15}}
    else:
        title_auto = {'text':"<b>Probability of default detected</b><br>\
    <span style='color: crimson; font-size:0.9em'>Credit<br><b>Not granted</b></span>", \
                    'font': {'color': 'crimson', 'size': 15}}


    fig2 = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prob_predict,
        mode = "gauge+number+delta",
        title = title_auto,
        delta = {'reference': best_thresh},
        gauge = {'axis': {'range': [None, 1]},
                'bgcolor': "crimson",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps' : [
                    {'range': [0, best_thresh], 'color': "forestgreen"},
                    {'range': [best_thresh, 0.5], 'color': "crimson"}],
                'threshold' : {'line': {'color': "crimson", 'width': 2}, 'thickness': 1, 'value': best_thresh},
                'bar': {'color': "palegoldenrod"}}))

    if prob_predict < best_thresh:
        fig2.update_layout(paper_bgcolor = "honeydew", font = {'color': "darkgreen", 'family': "Arial"})
    else:
        fig2.update_layout(paper_bgcolor = "lavenderblush", font = {'color': "crimson", 'family': "Arial"})

    st.write(fig2)

# with right_column:
if st.sidebar.checkbox('Show Customer Jauge'):
    jauge()



# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def customer_details():
    # utiliser graph objects avec une boucle sur top_10
    # pour montrer uniquement des données chiffrées

    sel = top_10
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
        grid = {'rows': 5, 'columns': 2, 'pattern': "independent", 'xgap' : 0.5, 'ygap' : 0.6})

    fig1.update_layout(
        autosize=False,
        width=800,
        height=700,)

    plt.tight_layout()

    st.write(fig1)

# with left_column:
if st.sidebar.checkbox('Show Customer Details Top 10'):
    customer_details()


