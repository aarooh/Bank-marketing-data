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
#sns.boxplot(x='y', y='pdays', data=data)

# Meaningful columns
#sns.countplot(x='education',hue='y', data=data)

#sns.distplot(data["balance"])
#sns.distplot(data["age"])

# %%
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('bank-full.csv', sep=';')
#data[['pdays', 'campaign', 'previous','age','duration']].describe()
def remove_outliers(data, column, minimum, maximum):
    col_values = data[column].values
    data[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)
    return data

min_val = data["age"].min()
max_val = 80

data = remove_outliers(data=data, column='age' , minimum=min_val, maximum=max_val)

# Split data into train and test data

X = data.drop('y',axis = 1).values 
y = data['y'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=20)




# %%
