import os
import pandas as pd
import numpy as np
import warnings
from colorama import Fore
warnings.filterwarnings('ignore')
scriptFolder = os.path.dirname(os.path.realpath(__file__))
csvFile = "Admission.csv"
csvFolder = scriptFolder + "\\" + csvFile
cleanedCsvFile = "Admission_cleaned.csv"
cleanedCsvFolder = scriptFolder + "\\" + cleanedCsvFile
df = pd.read_csv(csvFolder)
df.columns = df.columns.str.strip()
print("Duplicated values: \n", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print ("Dataset Null Values: \n", df.isnull().sum())

for col in df.columns:
    df[col].fillna(df[col].mean(), inplace=True)

print("Dataset Null Values: \n", df.isnull().sum())
num_cols = df.select_dtypes(include=[np.number]).columns

def outliersSummary(df):
    print("\nOutliers count per numerical column:")
    outlier_summary = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_summary[col] = outliers.shape[0]
        print(f"{col}: {outliers.shape[0]}")

print("Outliers summary (Before Handling): \n")
outliersSummary(df)
feature_num_cols = [col for col in num_cols if col != 'Chance_of_Admit' and df[col].nunique() > 2]
def cap_outliers_iqr(dataframe, columns):
    for col in columns:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        dataframe[col] = np.where(
            dataframe[col] < lower, lower,
            np.where(dataframe[col] > upper, upper, dataframe[col])
        )
    return dataframe

df = cap_outliers_iqr(df, feature_num_cols)

print("\nOutliers summary (After Handling): \n")
outliersSummary(df)

df.to_csv(cleanedCsvFolder, index=False)
print(Fore.MAGENTA, "\nCleaned DataFrame is saved:as ",Fore.RESET,cleanedCsvFile)