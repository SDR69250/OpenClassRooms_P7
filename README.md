# OC Data Scientist - Projet 7 : Implémentez un modèle de scoring

## Contexte

Une société financière, nommée "Prêt à dépenser", propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. Il s’agit de mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, de classifier la demande en crédit accordé ou refusé, et d’expliquer de manière transparente au client les décisions d’octroi de crédit. Ce modèle est destiné à être utilisé par des professionnels de la banque auprès des clients. Il faut être capable d’expliquer les décisions d’octroi de crédit de manière claire et transparente. Un tableau de bord interactif a été développé pour répondre à ce besoin.

## Données

https://www.kaggle.com/c/home-credit-default-risk/data

## Missions
-  Construire un modèle de scoring qui produit une prédiction de probabilité de défaut pour une demande client de façon automatique.
-  Construire un dashboard interactif à destination des gestionnaires de relation client, qui interprète les prédictions faites par le modèle, visualise les informations des clients, et permette d’expliquer aux clients les décisions d’octroi de crédit.

## Livrables

1. Développement du modèle de classification
-  DeRosa_Sebastien_1_eda_082022.ipynb : notebook Exploratory Data Analysis (source : de https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script) 
-  DeRosa_Sebastien_2_modelisation_082022.ipynb : notebook de modélisation, optimisation et sélection du meilleur modèle
-  DeRosa_Sebastien_3_interpretation_db_082022.ipynb : notebook préparation de l’interprétation et de la visualisation des résultats du modèle
2. Création d'un dashboard interactif : décision d'octroi de crédit
-  app_flask.py : code source de l’API Flask de mise à disposition des routes 
-  Dashboard.py : code du dashboard avec Streamlit
-  Lien vers le tableau de bord final : http://15.188.141.153:8501/
3. Documentation du projet
-  De_Rosa_Sebastien_3_note_methodologique_082022.pdf : description de la méthodologie du modèle de scoring, de son interprétation, de son déploiement, et des limites et améliorations identifiées
-  DeRosa_Sebastien_4_presentation_082022.pdf : support de présentation
