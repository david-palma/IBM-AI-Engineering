#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
#
# Required libraries:

# In[585]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
#
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[586]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File

# In[587]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[588]:


df.shape


# ### Convert to date time object

# In[589]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
#
#

# Let’s see how many of each class is in our data set

# In[590]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
#

# Lets plot some columns to underestand data better:

# In[507]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[591]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[592]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan

# In[593]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4

# In[594]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[595]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
#

# Lets convert male to 0 and female to 1:
#

# In[596]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# #### How about education?

# In[597]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[598]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame

# In[599]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[600]:


X = Feature
X[0:5]


# What are our lables?

# In[601]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[602]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification

# Classification using the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[603]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[604]:


# split the dataset into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 4)

print('Size of train set: ' + str(X_train.shape[0]) + ' x ' + str(X_train.shape[1]))
print('Size of test set: ' + str(X_test.shape[0]) + ' x ' + str(X_test.shape[1]))


# In[605]:


# look for the best K
K        = 10
mean_acc = np.zeros((K - 1))
std_acc  = np.zeros((K - 1))

for k in range(1, K):

    # train model
    clf_knn = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

    # compute prediction
    y_hat = clf_knn.predict(X_test)

    # compute accuracy
    mean_acc[k - 1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[k - 1]  = np.std(y_hat == y_test)/np.sqrt(y_hat.shape[0])

    # print result
    print('K =', k, '\n', ' Accuracy:', '{0:.4f}'.format(mean_acc[k - 1]), '+/-', '{0:.6f}'.format(std_acc[k - 1]))


# In[606]:


# plot the results
plt.clf()
plt.plot(range(1,K), mean_acc, 'g')
plt.fill_between(range(1,K), mean_acc - std_acc, mean_acc + std_acc, color='k', alpha=0.05)
plt.legend(('Accuracy ', '+/- Std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of neighbours (K)')
plt.tight_layout()
plt.show()


# In[607]:


# create an instance of Neighbours Classifier and train the model with K=5
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(X, y)


# # Decision Tree

# In[609]:


from sklearn.tree import DecisionTreeClassifier


# In[610]:


# create an instance of Decision Tree Classifier and train the model
clf_dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
clf_dt.fit(X, y)


# # Support Vector Machine

# In[611]:


from sklearn.svm import SVC


# In[612]:


# create an instance of Support Vector Classifier and train the model
clf_svm = SVC(kernel='rbf')
clf_svm.fit(X, y)


# # Logistic Regression

# In[613]:


from sklearn.linear_model import LogisticRegression


# In[614]:


# create an instance of Logistic Regression Classifier and train the model
clf_lr = LogisticRegression(C = 0.01, solver = 'liblinear', random_state = 0)
clf_lr.fit(X, y)


# # Model Evaluation using Test set

# In[615]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[616]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation

# In[617]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# ### Pre-processing, feature extraction and normalisation

# In[618]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)
test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
test_df[['Principal','terms','age','Gender','education']].head()


# In[619]:


# suppress warning about implicit conversion
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# feature selection
Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature, pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1, inplace=True)
X_test = Feature
y_test = test_df['loan_status'].values

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
X_test[0:5]


# #### K Nearest Neighbor

# In[620]:


# compute prediction
y_hat = clf_knn.predict(X_test)

# compute various scores
y_t = y_test == 'PAIDOFF'
y_h = y_hat  == 'PAIDOFF'

knn_f1_score = f1_score(y_t, y_h)
knn_jaccard  = jaccard_similarity_score(y_t, y_h)

print('K Nearest Neighbor')
print('F1-score: %1.7f' % knn_f1_score)
print('Jaccard:  %1.7f' % knn_jaccard)

# plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_hat)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Spectral)
plt.ylabel('True label')
plt.xlabel('Predicted label')

class_names = ['PAIDOFF','COLLECTION']
tick_marks  = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + ' = ' + str(cm[i][j]))

plt.show()


# #### Decision Tree

# In[621]:


# compute prediction
y_hat = clf_dt.predict(X_test)

# compute various scores
y_t = y_test == 'PAIDOFF'
y_h = y_hat  == 'PAIDOFF'

dt_f1_score = f1_score(y_t, y_h)
dt_jaccard  = jaccard_similarity_score(y_t, y_h)

print('Decision Tree')
print('F1-score: %1.7f' % dt_f1_score)
print('Jaccard:  %1.7f' % dt_jaccard)

# plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_hat)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Spectral)
plt.ylabel('True label')
plt.xlabel('Predicted label')

class_names = ['PAIDOFF','COLLECTION']
tick_marks  = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + ' = ' + str(cm[i][j]))

plt.show()


# #### Support Vector Machine

# In[622]:


# compute prediction
y_hat = clf_svm.predict(X_test)

# compute various scores
y_t = y_test == 'PAIDOFF'
y_h = y_hat  == 'PAIDOFF'

svm_f1_score = f1_score(y_t, y_h)
svm_jaccard  = jaccard_similarity_score(y_t, y_h)

print('Support Vector Machine')
print('F1-score: %1.7f' % svm_f1_score)
print('Jaccard:  %1.7f' % svm_jaccard)

# plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_hat)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Spectral)
plt.ylabel('True label')
plt.xlabel('Predicted label')

class_names = ['PAIDOFF','COLLECTION']
tick_marks  = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + ' = ' + str(cm[i][j]))

plt.show()


# #### Logistic Regression

# In[623]:


# compute prediction
y_hat  = clf_lr.predict(X_test)
y_prob = clf_lr.predict_proba(X_test)

# compute various scores
y_t = y_test == 'PAIDOFF'
y_h = y_hat  == 'PAIDOFF'

lr_f1_score = f1_score(y_t, y_h)
lr_log_loss = log_loss(y_t, y_prob)
lr_jaccard  = jaccard_similarity_score(y_t, y_h)

print('Logistic Regression')
print('F1-score: %1.7f' % lr_f1_score)
print('Log-loss: %1.7f' % lr_log_loss)
print('Jaccard:  %1.7f' % lr_jaccard)

# plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_hat)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Spectral)
plt.ylabel('True label')
plt.xlabel('Predicted label')

class_names = ['PAIDOFF','COLLECTION']
tick_marks  = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + ' = ' + str(cm[i][j]))

plt.show()


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.74    | 0.83     | NA      |
# | Decision Tree      | 0.72    | 0.81     | NA      |
# | SVM                | 0.72    | 0.84     | NA      |
# | LogisticRegression | 0.74    | 0.85     | 0.56    |
