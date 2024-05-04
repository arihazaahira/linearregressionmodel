
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

df = pd.read_csv('ETH-USD2.csv')
float_columns = ['Close', 'Open', 'High','Low','Volume']  
for col in float_columns:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

df.head()
df.shape
print(df)
duplications = df.duplicated().sum()
valeurs_manquantes = df.isnull().values.any()
df.info()
df.describe()
df_combined = pd.concat([df], axis=1)
df_combined.to_csv('newdata.csv', index=False)





