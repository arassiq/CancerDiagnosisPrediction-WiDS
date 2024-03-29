import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes

# Load the dataset
originalDataSet = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/Data/cleaned_imputed_with_averages.csv')

ids = originalDataSet[['patient_id']].copy()  # Creating a new data frame with just the ID column

df_without_id = originalDataSet.drop('patient_id', axis=1)  # Removing the ID column from the original data frame 

categorical_cols = df_without_id.select_dtypes(exclude=[np.number]).columns

# Convert categorical columns to category type if they are not already
for col in categorical_cols:
    df_without_id[col] = df_without_id[col].astype('category')

# K-Prototypes clustering
kproto = KPrototypes(n_clusters=5, init='Cao', n_init=1, verbose=1)
clusters = kproto.fit_predict(df_without_id, categorical=[df_without_id.columns.get_loc(c) for c in categorical_cols])

# Adding the cluster labels to your dataset
df_without_id['Cluster'] = clusters

k = 4

for i in range(k):  # Assuming 'k' is your number of clusters writes the summaries to console
    print(f"Cluster {i} Summary:")
    print(df_without_id[df_without_id['Cluster'] == i].describe(include='all'), "\n")





def summariesToCsv():
    # Create a list to store DataFrames
    summary_dfs = []

    for i in range(k):  # Assuming 'k' is your number of clusters
        # Create a DataFrame with the cluster title
        cluster_title_df = pd.DataFrame({'Statistic': [f'Cluster {i} Summary'], 'Feature': [None], 'Value': [None], 'Cluster': [None]})

        # Get the summary for each cluster
        cluster_summary = df_without_id[df_without_id['Cluster'] == i].describe(include='all')

        # Convert multi-level index into columns (if necessary)
        cluster_summary = cluster_summary.unstack().reset_index()
        cluster_summary.columns = ['Statistic', 'Feature', 'Value']

        # Add a column to identify the cluster
        cluster_summary['Cluster'] = f"Cluster {i}"

        # Append the title and the summary DataFrame to the list
        summary_dfs.append(cluster_title_df)
        summary_dfs.append(cluster_summary)

    # Concatenate all summary DataFrames
    all_summaries = pd.concat(summary_dfs)

    # Save the concatenated DataFrame to a CSV file
    all_summaries.to_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/cluster_summaries.csv', index=False)

summariesToCsv()


import seaborn as sns
import matplotlib.pyplot as plt

'''
# Example: Box plot for a numerical feature
for i in range(k):
    sns.boxplot(x='Cluster', y='patient_age', data=df_without_id)
    plt.title(f"Feature Distribution for clusters")
    plt.show()
'''
# Replace 'actual_categorical_column' with the name of a categorical column from your df_without_id

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)

'''
for i in range(k):
    cluster_data = df_without_id[df_without_id['Cluster'] == i]
    cluster_data['actual_categorical_column'].value_counts().plot(kind='bar')
    plt.title(f"Category Counts in Cluster {i}")
    plt.show()
'''
# Reattach the patient_id to the clustered data
clustered_data = pd.concat([ids, df_without_id], axis=1)

# Export to CSV
output_file_path = '/Users/mob/Desktop/Folders/Junior Year Semester 2/BUSanalyticsImmersion/clusters.csv'  # change this to your desired output path
clustered_data.to_csv(output_file_path, index=False)

