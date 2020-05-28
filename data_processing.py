# %%
import seaborn as sns 
import pandas as pd
from matplotlib import pyplot as plt
#Make Boxplots

data = pd.read_csv('bank-full.csv', sep=';')
#sns.pairplot(data, hue='y')
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
#Pdays will be kept as is, because the values are skewed
# Remove values over 1.5 * IQR above third quartile
# Remove values less than 1.5 * IQR below first quartile

#Age
q3 = data['age'].quantile(0.75)
q1 = data['age'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
data = data[data['age'] >= min]
data = data[data['age'] <= max]

#Duration
q3 = data['duration'].quantile(0.75)
q1 = data['duration'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
data = data[data['duration'] >= min]
data = data[data['duration'] <= max]

# campaign
q3 = data['campaign'].quantile(0.75)
q1 = data['campaign'].quantile(0.25)
iqr = q3 - q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
data = data[data['campaign'] >= min]
data = data[data['campaign'] <= max]

data.head()


#%%
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

le = preprocessing.LabelEncoder()
data['job'] = le.fit_transform(data['job'])
data['marital'] = le.fit_transform(data['marital'])
data['education'] = le.fit_transform(data['education'])
data['housing'] = le.fit_transform(data['housing'])
data['default'] = le.fit_transform(data['default'])
data['loan'] = le.fit_transform(data['loan'])
data['contact'] = le.fit_transform(data['contact'])
data['month'] = le.fit_transform(data['month'])
data['poutcome'] = le.fit_transform(data['poutcome'])

data.y[data.y == 'yes'] = 1
data.y[data.y == 'no'] = 0
data['y'] = data['y'].astype('str').astype('int')



X = data.drop('y',axis = 1).values
y = data['y'].values

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

# %%
#LogisticRegression
model = RandomForestClassifier(n_jobs=-1,)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")



# %%
#RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")

# %%
#Support Vector Machine
from sklearn.svm import SVC
model = SVC(probability=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")

# %%
#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")


# %%
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")


# %%
#Naive Bayesian
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy",accuracy)

# Confusion Matrix for the tested data
conf_matr = confusion_matrix(y_test,model.predict(x_test))
df_cma = pd.DataFrame(conf_matr)
print("Confusion Matrix")
print(df_cma)


# ROC
probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## Draw ROC Curve
import matplotlib.pyplot as plt

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Draw Confusion Matrix
class_names = ['negative','positive']
df_heatmap = pd.DataFrame(confusion_matrix(y_test,model.predict(x_test)), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")

# %%
