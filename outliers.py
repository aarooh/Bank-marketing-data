# %%
import seaborn as sns 
import pandas as pd

#Make Boxplots

data = pd.read_csv('bank-full.csv', sep=';')
sns.pairplot(data, hue='y')
#sns.boxplot(x='y', y='duration', data=data)
#sns.boxplot(x='y', y='balance', data=data)
#sns.boxplot(x='y', y='age', data=data)
#sns.boxplot(x='y', y='day', data=data)
#sns.boxplot(x='y', y='pdays', data=data)
#sns.boxplot(x='y', y='campaign', data=data)
# Meaningful columns
#sns.countplot(x='education',hue='y', data=data)
#sns.distplot(data["balance"])
#sns.distplot(data["age"])

# %%
import pandas as pd

data = pd.read_csv('bank-full.csv', sep=';')
data[['pdays', 'campaign', 'previous','age','duration']].describe()
# Lets calculate Outlier values
# Outliers in Age, Duration, pdays, campaign
# Remove values over 1.5 * IQR above third quartile
# Remove values less than 1.5 * IQR below first quartile

#Age
q3 = data['age'].quantile(0.75)
q1 = data['age'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
print(min, max)

data = data[data['age'] >= min]
data = data[data['age'] <= max]

#Duration
q3 = data['duration'].quantile(0.75)
q1 = data['duration'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
print(min, max)
data = data[data['duration'] >= min]
data = data[data['duration'] <= max]

# campaign
q3 = data['campaign'].quantile(0.75)
q1 = data['campaign'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
print(min, max)
data = data[data['campaign'] >= min]
data = data[data['campaign'] <= max]

data.describe()


#%%
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Split data into train and test data

X = data.drop('y',axis = 1).values 
y = data['y'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=20)




# %%
