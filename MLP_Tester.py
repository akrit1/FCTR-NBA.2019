"""
Module: MLP_Tester.py
Use: Reads in the pickled MLP and testing sets from MLP_Creator and determines MAPE
Last Edited: Akrit Sinha, 06-26-2019
"""
# Packages
import pickle
import numpy as np

# Import pickled testing sets
X_test = pickle.load(open('X_test', 'rb'))
y_test = pickle.load(open('y_test', 'rb'))


# Defines MAPE function for accuracy
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Determines the accuracy of single run using pickled MLP
def acc():
    load_mlp = pickle.load(open('MLP', 'rb'))
    predictions = load_mlp.predict(X_test)

    return 100 - mean_absolute_percentage_error(y_test.values.ravel(), predictions)


# Initiates 500 runs and determines average MAPE score
if __name__ == "__main__":
    run_sum = 0
    for i in range(500):
        run_sum += acc()
    print(run_sum/500)
