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

st.markdown("# Customer vs Group \U00002696")
st.sidebar.markdown("# Customer vs Group \U00002696")



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
feat_tot = st.session_state['feat_tot']
feat_top = st.session_state['feat_top']
data_idx = st.session_state['data_idx']
viz = ""


# fig = plt.figure(figsize = (10, 20))

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if st.sidebar.checkbox('Show Feature Importance Analysis'):
    # Select type of feature explanation visualization
    viz = st.sidebar.selectbox("Select feature importance visualization", options=['feature importance', 'expected to predicted', 'feature impact', 'odds of default', 'probability of default'])

    # @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
    def sample_feature_importance(idx, type=viz):

        fig1 = plt.figure()
        if type=='odds of default':
            st_shap(shap.force_plot(shap_values.base_values[idx], shap_values1[idx, :], X_test.iloc[idx, :]))
        elif type=='expected to predicted':
            shap.waterfall_plot(shap_values[idx], max_display=10)
            st.pyplot(fig1)    
        elif type=='feature impact':
            shap.summary_plot(shap_values, X_test, max_display=10)
            st.pyplot(fig1)
        elif type=='feature importance':
            shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns)
            st.pyplot(fig1)
        elif type=='probability of default':
            st_shap(shap.force_plot(shap_values.base_values[idx], shap_values1[idx, :], X_test.iloc[idx, :], link='logit')) # choose between 'logit' or 'identity'
        else:
            return "Return valid visual ('feature importance', 'expected to predicted', 'feature impact', 'odds of default', 'probability of default')"


    sample_feature_importance(data_idx)




if st.sidebar.checkbox('Show Observation vs Group'):
    # @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
    def obs_vs_group():
        print('les 20 premières features représentent', round((feat_top/feat_tot)*100, 2),'% de l\'importance de toutes les features.')

        # Show boxplot for each feature with original units
        # selection of 20 most explicative features
        
        sel = top_20
        width = 20
        height = ((len(sel)+1)/2)*2

        fig2 = plt.figure(figsize=(width, height))

        # fig2 = plt.subplot(2,1,2,figsize=(width, height))
        for i, c in enumerate(sel,1):
            chaine = 'Distribution de : ' + c
            ax = fig2.add_subplot((len(sel)+2)//2, 2, i)
            plt.title(chaine)
            sns.boxplot(x=X_test[c],
                        orient='h',
                        color='lightgrey',
                        notch=True,
                        flierprops={"marker": "o"},
                        boxprops={"facecolor": (.4, .6, .8, .5)},
                        medianprops={"color": "coral"},
                        ax=ax)

        # show customer ID values for each feature
            plt.scatter(ID_to_predict[c], c, marker = 'D', c='r', s=200)

        # scaling automatique ('notebook', 'paper', 'talk', 'poster')
        sns.set_context("talk")
        fig2.tight_layout()

        st.pyplot(fig2)

    obs_vs_group()