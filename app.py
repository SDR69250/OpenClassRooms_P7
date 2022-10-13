from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import numpy as np
import shap
import json
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import pickle



app = Flask(__name__)
# basepath = os.path.abspath("./OPC_P7/OpenClassRooms_P7/")

# load models, threshold, data and explainer
model_load = joblib.load("model.joblib") # sur PA pour chaque fichier remplacer le premier " par "basepath + "/
best_thresh = joblib.load("best_thresh_LightGBM_NS.joblib")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")
explainer = joblib.load("explainer")
shap_values = joblib.load("shap_values.pkl")
shap_values1 = joblib.load("shap_values1.pkl")
expected_value = joblib.load("expected_values.pkl")
columns = shap_values.feature_names
data = pd.DataFrame(y_test, index=y_test.index).reset_index()
data["PRED"] = y_pred


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


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict/<int:Client_Id>")
def predict(Client_Id: int):
    # Customer index in the corresponding array
    data_idx = data.loc[data["SK_ID_CURR"]==int(Client_Id)].index[0]
    # Customer data based on customer index in final X_test array
    ID_to_predict = pd.DataFrame(X_test.iloc[data_idx,:]).T  
    # on réalise la prédiction de ID_to_predict avec le modèle 
    prediction = sum((model_load.predict_proba(ID_to_predict)[:, 1]>best_thresh)*1)
    prob_predict = float(model_load.predict_proba(ID_to_predict)[:, 1])
    # on renvoie la prédiction
    return json.dumps({"prediction" : int(prediction), "prob_predict": prob_predict, "ID_to_predict" : ID_to_predict.to_json(orient='columns')})

    
@app.route("/load_top_10/", methods=['GET'])
def load_top_10():
    return json.dumps({"top_10" : top_10})

@app.route("/load_top_20/", methods=['GET'])
def load_top_20():
    return json.dumps({"top_20" : top_20, 'feat_tot': feat_tot, 'feat_top': feat_top})

@app.route("/load_best_thresh/", methods=['GET'])
def load_best_thresh():
    return {"best_thresh" : best_thresh} 

@app.route("/load_X_test/", methods=['GET'])
def load_X_test():
    return {"X_test" : pd.DataFrame(X_test).to_json(orient='columns')} 

@app.route("/load_data/", methods=['GET'])
def load_data():
    return {"data" : pd.DataFrame(data).to_json(orient='columns')} 


# provide dataviz for shap features importance on selected customer's credit decision 
@app.route("/cust_vs_group/<int:Client_Id>")
def cust_vs_group(Client_Id: int):
    # utiliser idx pour former le graph via Flask et l'importer dans streamlit
    data_idx = data.loc[data["SK_ID_CURR"]==int(Client_Id)].index[0] #string ou pas
    # customer data based on customer index in final X_test array
    ID_to_predict = pd.DataFrame(X_test.iloc[data_idx,:]).T
    # return json string
    return json.dumps({'base_value': shap_values.base_values[data_idx], 'shap_values1_idx': shap_values1[data_idx, :].tolist(), \
    "ID_to_predict": ID_to_predict.to_json(orient='columns')})


# def st_shap(plot, height=None):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#     components.html(shap_html, height=height)

# # provide dataviz for shap features importance on selected customer's credit decision 
# @app.route("/expected_to_predicted/<int:Client_Id>")
# def expected_to_predicted(Client_Id: int):
#     # fig1 = plt.figure()
#     data_idx = data.loc[data["SK_ID_CURR"]==int(Client_Id)].index[0] 
#     shap.waterfall_plot(shap_values[data_idx], max_display=10, show=False).savefig("/templates/waterfall.png")
#     return render_template('/templates/waterfall.png')




#     return json.dumps(Client_Id, viz)



# def sample_feature_importance(idx, type):

#     fig1 = plt.figure()
#     if type=='odds of default':
#         st_shap(shap.force_plot(shap_values.base_values[idx], shap_values1[idx, :], X_test.iloc[idx, :]))
#     elif type=='expected to predicted':
#         shap.waterfall_plot(shap_values[idx], max_display=10)
#         st.pyplot(fig1)    
#     elif type=='feature impact':
#         shap.summary_plot(shap_values, X_test, max_display=10)
#         st.pyplot(fig1)
#     elif type=='feature importance':
#         shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns)
#         st.pyplot(fig1)
#     elif type=='probability of default':
#         st_shap(shap.force_plot(shap_values.base_values[idx], shap_values1[idx, :], X_test.iloc[idx, :], link='logit')) # choose between 'logit' or 'identity'
#     else:
#         return "Return valid visual ('feature importance', 'expected to predicted', 'feature impact', 'odds of default', 'probability of default')"


if __name__ == "__main__":
    app.run()
