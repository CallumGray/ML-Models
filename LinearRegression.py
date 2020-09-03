import numpy as np
from Sarcos import SarcosData
from Toy import ToyData
import time

data = SarcosData()
#data = ToyData(25000)

# Calculate RMSE of the algorithm
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#
#
#
#   LINEAR REGRESSION
#
#
#

def reshape_train(x):
    return x.reshape(-1,1)

def add_ones(x):
    ones = np.ones(shape=x.shape[0]).reshape(-1,1)
    return np.concatenate((ones,x),1)

def train_coefficients(old_x,y):
    x = old_x

    if len(x.shape) == 1:
        x = reshape_train(x)

    x = add_ones(x)
    return np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

def predict(query):

    prediction = coefficients[0]

    for feature,coefficient in zip(query,coefficients[1:]):
        prediction += feature * coefficient

    return prediction


build_time = time.time()
coefficients = train_coefficients(data.x_train,data.y_train)
build_time = time.time() - build_time
print('Build Time on Length:', len(data.x_train), ':', build_time)

def test_quality():
    train_predictions = np.array([predict(x) for x in data.x_train])
    train_error = rmse(train_predictions,data.y_train)

    predict_all_time = time.time()
    test_predictions = np.array([predict(x) for x in data.x_test])
    predict_all_time = time.time() - predict_all_time
    print('Test Time on Length', len(data.x_test), ':', predict_all_time)
    test_error = rmse(test_predictions,data.y_test)

    return train_error,test_error

train_err,test_err = test_quality()

print('Train Standard Deviation: ',np.std(data.y_train))
print('Test Standard Deviation: ',np.std(data.y_test))
print('LINEAR REGRESSION TRAIN ERROR: ',train_err)
print('LINEAR REGRESSION TEST ERROR: ',test_err)