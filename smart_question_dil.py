import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import the dataset as csv then read as pandas dataframe
curr_dir = os.getcwd()
test = curr_dir + "\\test.csv"
train = curr_dir + "\\train.csv"

df_test = pd.read_csv(test, index_col = "Unnamed: 0")
df_train = pd.read_csv(train, index_col = "Unnamed: 0")
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)

# check the dataframe
df_test.shape
df_train.shape
df_train.isnull().sum()  # 310 null values for Arrival Delay in Minutes
test = df_train.describe()  # median value for Arrival Delay is 0 and very highly skewed towards 0, so we replace with 0
df_train['Arrival Delay in Minutes'].fillna(0, inplace = True)
df_train.isnull().sum()
df_test['Arrival Delay in Minutes'].fillna(0, inplace = True)
df_test.isnull().sum()

sns.set_palette("colorblind")
sns.distplot(df_train['Arrival Delay in Minutes'])
plt.title("Distribution of Arrival Delay in Minutes")
plt.show()

df_train.describe()
df_train.info()

# if i needed to convert things like gate location food and drink to categorical/objects
#columns =['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness']
#for col in columns:
#    df_train.loc[:,col] = df_train.loc[:,col].astype('object')


# looks good, now we can look at our data.
# first question is which of the categorical and which of the discrete influence satisfaction the most.

# gender vs satisfaction
sns.countplot(x='satisfaction', hue='Gender', data=df_train)
plt.title('Count plot for Gender')
plt.show()  # about a 50/50 split

# customertype vs satisfaction
sns.countplot(x='satisfaction', hue="Customer Type", data=df_train)
plt.title('Count plot for Customer Type')
plt.show()  # seems like loyal customers more likely to give feedback than non-loyal but were more often dissatisfied

# type of travel vs satisfaction
sns.countplot(x='satisfaction', hue="Type of Travel", data=df_train)
plt.title('Count plot for Type of Travel')
plt.show()  # those traveling for business are more likely to be satisfied

# Class vs satisfaction
sns.countplot(x='satisfaction', hue="Class", data=df_train)
plt.title('Count plot for Class')
plt.show()  # those traveling Eco are more likely not satisfied. Business class also more likely to be satisfied


# inflight wifi vs satisfaction
sns.countplot(x='satisfaction', hue="Inflight wifi service", data=df_train)
plt.title('Count plot for Inflight wifi service')
plt.show()  # multivariate might prefer a stacked graph. Looks like higher wifi service the low counts of dissatisfaction

# departure/arrival time convenient vs satisfaction
sns.countplot(x='satisfaction', hue="Departure/Arrival time convenient", data=df_train)
plt.title('Count plot for Departure/Arrival time convenient')
plt.show()  # same as above.

#for o in columns:
#    sns.countplot(x='satisfaction', hue=o, data=df_train)
#    plt.title('Count plot for' + o)
#plt.show()

#plt.figure(figsize=(10,10))
#list1=["Age",'Inflight wifi service',
#       'Departure/Arrival time convenient', 'Ease of Online booking',
#       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
#       'Inflight entertainment', 'On-board service', 'Leg room service',
#       'Baggage handling', 'Checkin service', 'Inflight service',
#       'Cleanliness',"satisfaction"]
#sns.heatmap(df_train[list1].corr(),annot=True,fmt=".2f")
#plt.show()

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Create Dummy variables for the variables
df_train=pd.get_dummies(df_train, columns=['Customer Type'])
df_train=pd.get_dummies(df_train, columns=['Type of Travel'])
df_train=pd.get_dummies(df_train, columns=['Class'])
df_train=pd.get_dummies(df_train, columns=['Gender'])
# Create the training and test datasets. At first we use all variables.
df_train_len=len(df_train)
train=df_train[:df_train_len]
X_train=train.drop(labels="satisfaction",axis=1)
y_train=train["satisfaction"]
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.33,random_state=42)

print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))
# Initiate the Logistic Regression Model and print the accuracy
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
acc_log_train=round(logreg.score(X_train,y_train)*100,2)
acc_log_test=round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Test Accuracy: % {}".format(acc_log_test))
# Print the coef's and the intercept's
print(logreg.coef_, logreg.intercept)




