# Predicting Metastatic Cancer Diagnosis within 90 Days

## Overview

  This project, developed for the WiDS Datathon and sponsored by Gilead Sciences, focuses on predicting metastatic cancer diagnosis within 90 days of screening using a rich dataset. The aim is to identify factors influencing timely treatment for breast cancer patients and contribute to reducing healthcare disparities.

## Dataset

  The dataset, provided by Health Verity and enriched by NASA/Columbia University, includes demographics, diagnosis, treatment options, and insurance information for patients diagnosed with breast cancer from 2015-2018, along with socio-economic and toxicology data.

## Methodology

### Data Preprocessing
  The project begins with comprehensive data cleaning and imputation to prepare the dataset for analysis. This process involved:

#### Tableau Prep Builder: Used for initial cleaning steps, including removing duplicates, handling outliers, and standardizing column names for easier processing.
#### Python Script - Imputing.py: Applied to handle missing values in the dataset. Numerical columns were imputed with mean values, while categorical columns were filled with the most frequent values to preserve data integrity.
#### Feature Engineering: New features were created to enhance model performance, including indicators of healthcare access and environmental exposure, based on existing data.

### Clustering with k-Prototypes
  Given the mixed data types (numerical and categorical) in our dataset, k-Prototypes clustering was employed to identify distinct groups of patients. This method allowed us to explore underlying patterns related to demographics, diagnosis, and treatment options, highlighting factors that may influence the likelihood of receiving a timely metastatic cancer diagnosis.

#### Selection of Features: Key variables were chosen based on their potential relevance to healthcare outcomes, including socio-economic status, geographic location, and prior health conditions.
#### Optimization: The optimal number of clusters was determined through the evaluation of silhouette scores, ensuring meaningful segmentation of the patient population.

### Classification Model Optimization
  The goal of predicting whether patients would receive a metastatic cancer diagnosis within 90 days of screening required a careful selection and optimization of classification models:

#### Model Evaluation: Several machine learning models were tested, including Random Forest, Gradient Boosting, and CatBoost, among others.
#### Hyperparameter Tuning: Utilized gridsearch.py for extensive hyperparameter tuning with RandomizedSearchCV, focusing on maximizing accuracy while preventing overfitting.
#### Model Selection: The model with the highest accuracy on the validation set was selected for final predictions. This approach ensured the reliability and relevance of our predictive insights.

## Key Outcomes
  Through this methodology, we aimed to uncover significant predictors of timely diagnosis and treatment, offering potential pathways to mitigate healthcare disparities and improve outcomes for patients with aggressive forms of breast cancer.



Gilead Sciences and Health Verity for providing the dataset.
WiDS Datathon organizers.
NASA/Columbia University for enriching the dataset.
![image](https://github.com/arassiq/arassiq-CancerDiagnosisPrediction-WiDS/assets/143036773/900e3aa8-957c-449e-897f-a77062628e95)
