import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Load the training dataset
df_train = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/cleaned imputed data/train_C_I.csv')

# Identifying categorical and numerical columns
categorical_cols = df_train.select_dtypes(include=['object']).columns
numerical_cols = df_train.select_dtypes(include=['float64']).columns.drop(['DiagPeriodL90D', 'patient_id'])

# Preprocessing transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define target variable and predictors for training data
y_train = df_train['DiagPeriodL90D']
X_train = df_train.drop(['DiagPeriodL90D', 'patient_id'], axis=1)

# Define the parameter grid for CatBoost
param_distributions = {
    'model__depth': [4, 6, 8, 10],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'model__n_estimators': [100, 200, 300, 500],
    'model__l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Create and fit a CatBoost model with RandomizedSearchCV
cb_model = CatBoostClassifier()
cb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', cb_model)])

random_search = RandomizedSearchCV(
    estimator=cb_pipeline, 
    param_distributions=param_distributions, 
    n_iter=50,  # number of parameter settings sampled
    cv=5, 
    n_jobs=-1, 
    verbose=2
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", random_search.best_params_)

# Load the test dataset
df_test = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/cleaned imputed data/test_C_I.csv')

# Separate the patient_id for later use and drop it from features if it's not a feature
patient_ids = df_test['patient_id']
X_test = df_test.drop('patient_id', axis=1) if 'patient_id' in df_test.columns else df_test

# Predicting 90daydiagperiod for test data
predicted_90daydiagperiod = random_search.predict(X_test)

# Creating a new DataFrame with patient_id and predicted 90daydiagperiod
result_df = pd.DataFrame({
    'patient_id': patient_ids,
    '90daydiagperiod': predicted_90daydiagperiod
})

# Saving the result to a new CSV file
result_df.to_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/predictions (for test)/(1)TestPredRandomSearch.csv', index=False)
