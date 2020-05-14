# %%
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('bank-full.csv', sep=';')

#sns.boxplot(x='y', y='duration', data=data)
#sns.boxplot(x='y', y='balance', data=data)
#sns.boxplot(x='y', y='age', data=data)
#sns.boxplot(x='y', y='day', data=data)
sns.boxplot(x='y', y='pdays', data=data)


#sns.distplot(data["balance"])
#sns.distplot(data["age"])

# %%
