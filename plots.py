# %%
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

print('Categorical features:', categorical)
print('Numerical features:', numerical)
# %% 
#Plot Answer variables of Answer attribute
sns.countplot(x=data['y'])
plt.title('Distribution of classes')
plt.xlabel('Target class')

# %%
#Plot Pdays versus Answer Attribute
sns.boxplot(y=data['pdays'], x=data['y'])
plt.title('Box plot of pdays vs y (target variable)')
plt.xlabel('y: target variable')




# %%
# Correlation matrix
data = pd.read_csv('bank-full.csv', sep=';')
data.y[data.y == 'yes'] = 1
data.y[data.y == 'no'] = 0
data['y'] = data['y'].astype('str').astype('int')
cmap = sns.diverging_palette(220, 10, as_cmap=True)
corr_data = data
corr = corr_data.corr()
cor_plot = sns.heatmap(corr,annot=True,cmap=cmap,linewidths=0.2,annot_kws={'size':10})
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.xticks(fontsize=12,rotation=-45)
plt.yticks(fontsize=12)
plt.title('Bank marketing dataset correlation matrix')
plt.show()

# %%
