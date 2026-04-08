import os
import warnings
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
import seaborn as sns
scriptFolder = os.path.dirname(os.path.abspath(__file__))
csvFile = "Admission.csv"
csvFilePath = scriptFolder + "/" + csvFile
imagesDirectory = scriptFolder + "/images"
df = pd.read_csv(csvFilePath)
warnings.filterwarnings("ignore")

#Data Exploration
print(Fore.GREEN, "\nDataset Columns:\n", Fore.RESET, df.columns)
print(Fore.GREEN, "\nDataset Shape:\n", Fore.RESET, df.shape)
print(Fore.GREEN, "\nDataset description:\n", Fore.RESET, df.describe())
print(Fore.GREEN, "\nDataset Head:\n", Fore.RESET, df.head())
print(Fore.GREEN, "\nDataset Information:", Fore.RESET)
df.info()

df.hist(bins=30, edgecolor='black', color='#4da6ff', layout=(len(df.columns) // 3 + 1, 3),
                        figsize=(12,10))
plt.suptitle("Histograms of Features", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{imagesDirectory}/Histograms.png")
plt.show()


correlationMatrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlationMatrix, annot=True, cmap='Purples', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(f"{imagesDirectory}/Correlation Matrix.png")
plt.show()

for col in df.columns:
    sns.displot(df[col])
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"{imagesDirectory}/countPerCol/{col}.png")
    plt.show()

target = "Chance_of_Admit "

for col in df.columns:
    if col != target:
        plt.figure()
        plt.scatter(df[target], df[col], color="#C4B6F2")
        plt.xlabel(target)
        plt.ylabel(col)
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.savefig(f"{imagesDirectory}/ColVsTarget/{col}.png")
        plt.show()
