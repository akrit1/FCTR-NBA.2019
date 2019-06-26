"""
Module: MLP_Creator.py
Use: Creates a multilayer perceptron along with testing sets and pickles them for use with MLP_Tester.py
Last Edited: Akrit Sinha, 06-26-2019
"""
# Packages
import pandas
import pickle
from datetime import datetime
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import data to pandas data frame
pandas.set_option('display.max_columns', None)
train_df = pandas.read_csv("training_set.csv", encoding="ISO-8859-1")

# Allocate features (X) and labels (y)
X = train_df.iloc[:, 1:]
y = train_df.iloc[:, :1]


# Returns time of post in terms of hours and minutes
def gettime(st):
    st = st[:-4]
    d = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    return int(d.hour * 60 + d.minute)


# Returns post date in terms of month and day of week
def getdate(st):
    st = st[:-4]
    d = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    return int(d.isoweekday() + d.month)


# Returns sentiment level of post caption
def getsent(st):
    if isinstance(st, str):
        t = TextBlob(st)
        return t.sentiment.polarity
    else:
        return 0


# Cleans the data, applies above functions to create complete numerical data frame
le = preprocessing.LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])
X['Date'] = X['Created'].map(getdate)
X['Created'] = X['Created'].map(gettime)
X['dFollowers'] = (X['Followers at Posting'].diff(periods=-1)) / (X['Created'].diff(periods=-1))
X['Sentiment'] = X['Description'].map(getsent)
X['Punctuation'] = X['Description'].str.count('!!!|ebron|rving|urry|iannis|arden|Why') \
                   + 2*X['Description'].str.count('@|#|ames')
X['Description'] = X['Description'].str.len()

# Splits the data into training and testing sets, and resolves NaNs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
X_test = X_test.fillna(0)
y_test = y_test.fillna(0)

# Scales the feature set for MLP sensitivity
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creates the MLP and fits it to training data
mlp = MLPRegressor(hidden_layer_sizes=(25,), max_iter=100000)
mlp.fit(X_train, y_train.values.ravel())

# Pickles MLP and testing sets for use with MLP_Tester
pickle.dump(X_test, open('X_test', 'wb'))
pickle.dump(y_test, open('y_test', 'wb'))
pickle.dump(mlp, open('MLP', 'wb'))
