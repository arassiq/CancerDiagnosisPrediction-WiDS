import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataSet = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/training/Training cleaned.csv')

import numpy as np

# Identify numerical and categorical columns
numeric_cols = dataSet.select_dtypes(include=[np.number]).columns
categorical_cols = dataSet.select_dtypes(exclude=[np.number]).columns

print(numeric_cols, "\n", categorical_cols)

numeric_imputer = SimpleImputer(strategy='median')  
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation
dataSet[numeric_cols] = numeric_imputer.fit_transform(dataSet[numeric_cols])
dataSet[categorical_cols] = categorical_imputer.fit_transform(dataSet[categorical_cols])

# Confirm the imputation
print(dataSet.isnull().sum())

dataSet.to_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/training/TrainingCleaningImputed.csv', index=False)


#imputer = SimpleImputer(strategy='median')

#print(dataSet.isnull().sum())

#categoricalColumns = ['payer_type', 'patient_state', 'patient_zip3', 'patient_age', 'breast_cancer_diagnosis_code', 'breast_cancer_diagnosis_desc', 'metastatic_cancer_diagnosis_code', 'Region', 'Division']



