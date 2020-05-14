# %%Main code structure
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt

data = pd.read_csv('bank-full.csv', sep=';')
data.head()
data.describe(include='all')
data.info()

# Are values are non-null
#Add categorical values to list
categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# Add numerical values to list
numerical = [x for x in data.columns.to_list() if x not in categorical]
# Remove last category the result from the numerical category
numerical.remove('y')