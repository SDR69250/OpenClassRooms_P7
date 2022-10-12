# ======================== | Imports | ========================

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import json
import requests # for get requests on url
import warnings
warnings.filterwarnings("ignore")
import shap
import streamlit.components.v1 as components # to visualize shap plots

# Imports from Home page
from Home import X_test, FLASK_URL


# ======================== | Initializations | ========================

# initialize variable viz for use in selection menu
viz = ""


# ======================== | Page title & sidebar | ========================

st.markdown("# Customer vs Group \U00002696")
st.sidebar.markdown("# Customer vs Group \U00002696")


# ======================== | Interactions, API calls and decode | ========================


#### API CALLS ####

# API call | GET data (used to select customer idx)
@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def load_data():
    url_data = FLASK_URL + "load_data/"
    response = requests.get(url_data)
    content = json.loads(response.content.decode('utf-8'))
    dict_data = content["data"]
    data = pd.DataFrame.from_dict(eval(dict_data), orient='columns')
    return data
data = load_data()

# Select Customer number SK_ID_CURR in data
idx = st.sidebar.selectbox(
    "Select Credit File", 
    data.SK_ID_CURR, key = "idx3")

# retrieve previously selected value from Home page
if 'idx' in st.session_state:
    idx = st.session_state['idx']

# API call | GET top_20
url_top_20 = FLASK_URL + "load_top_20/"
response = requests.get(url_top_20)
content = json.loads(response.content.decode('utf-8'))
top_20 = content["top_20"]
feat_top = content["feat_top"]
feat_tot = content["feat_tot"]

# API call | GET data for shap plots : type / index / shap_values :
url_cust_vs_group = FLASK_URL + "cust_vs_group/" + str(idx)
response = requests.get(url_cust_vs_group)
content = json.loads(response.content.decode('utf-8'))
base_value = content["base_value"]
shap_values1_idx = np.array(content["shap_values1_idx"])
dict_ID_to_predict = content["ID_to_predict"]
ID_to_predict = pd.DataFrame.from_dict(eval(dict_ID_to_predict), orient='columns')

# API call | GET expected_to_predicted waterfall plot : figure
url_expected_to_predicted = FLASK_URL + "expected_to_predicted/" + str(idx) 


# ======================== | Interactions Streamlit & Plots | ========================

# recall customer number
st.write(f"Customer number : {str(idx)}")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if st.sidebar.checkbox('Show Feature Importance Analysis'):
    # Select type of feature explanation visualization
    viz = st.sidebar.selectbox("Select feature importance visualization", options=['probability of default', 'odds of default', 'feature importance', 'expected to predicted', 'feature impact'])
else : st.write(f"""To visualise feature importance for customer {str(idx)}\n
    Please select "Select Feature importance visualization" on the sidebar.
    """)

# recall customer data
st.write(ID_to_predict)

# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def sample_feature_importance(idx, viz):

    fig1 = plt.figure()
    if viz=='probability of default':
        return st.write(st_shap(shap.force_plot(base_value, shap_values1_idx, ID_to_predict, link='logit'))) # choose between 'logit' or 'identity'
    elif viz=='odds of default':
        return st.write(st_shap(shap.force_plot(base_value, shap_values1_idx, ID_to_predict)))
    # elif type=='expected to predicted':
    #     shap.waterfall_plot(shap_values[idx], max_display=10)
    #     return st.pyplot(fig1)    
    # # elif type=='feature impact':
    #     shap.summary_plot(shap_values, X_test, max_display=10)
    #     st.pyplot(fig1)
    # elif type=='feature importance':
    #     shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns)
    #     st.pyplot(fig1)
sample_feature_importance(idx, viz)


if st.sidebar.checkbox('Show Observation vs Group'):
    # @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
    def obs_vs_group():
        st.write(f'les 20 premières features représentent {round((feat_top/feat_tot)*100, 2)} % de l\'importance de toutes les features.')

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
else : st.write(f"""To visualize customer {str(idx)} vs Group :\n
    Please select "Show Observation vs Group" on the sidebar.
    """)