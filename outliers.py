# %%
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('bank-full.csv', sep=';')
data_numeric = data
data.head()
#sns.boxplot(x='y', y='duration', data=data)

le = preprocessing.LabelEncoder()

def categorize(data):
    new_data = data.copy()
    le = preprocessing.LabelEncoder()
    new_data["job"] = le.fit_transform(new_data["job"])
    new_data["education"] = le.fit_transform(new_data["education"])
    return new_data

new_data = categorize(data)
new_data.head()
# %%


# %%
