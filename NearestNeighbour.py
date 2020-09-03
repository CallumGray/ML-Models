import numpy as np
import time
from Toy import ToyData
from Sarcos import SarcosData

data = SarcosData()
#data = ToyData(n=25000)


# Calculate RMSE of the algorithm
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def knn(x_query,x_train,k):

    # Vectorised euclidean distance
    subbed_squared = np.power(np.subtract(x_query,x_train),2)
    summed = np.sum(subbed_squared,axis=1)
    distances = np.sqrt(summed)
    index_distance = [(index,distance) for index,distance in enumerate(distances)]

    sorted_index_distance = sorted(index_distance,key=lambda t: t[1])
    neighbours = sorted_index_distance[:k]

    neighbours_index = [index for (index,distance) in neighbours]
    y_neighbours = data.y_train[neighbours_index]
    y_query = np.mean(y_neighbours)

    return y_query


training_time = time.time()

k_range = np.linspace(1,10,num=4,dtype=int)
best_k = 5
best_rmse = np.inf


for k in k_range:
    knn_results = []
    testlength = len(data.x_validation)
    count = 0
    
    for x_query in data.x_validation:
        result = knn(x_query,data.x_train,k)
        knn_results.append(result)
        count+=1
        print(count,'/',testlength)
     
    rmse_result = rmse(np.array(knn_results),data.y_validation)

    if rmse_result < best_rmse:
        best_k = k
        best_rmse = rmse_result

    print('RMSE FOR ',k,' is: ',rmse_result)

training_time = time.time() - training_time
print('Tuning Time on Length', len(data.x_train), ':', training_time)
print('Best K:',best_k)
print()

predict_all_time = time.time()

knn_results = []
testlength = len(data.x_test)
count = 0

for x_query in data.x_test:
    result = knn(x_query, data.x_train, best_k)
    knn_results.append(result)
    count += 1
    #print(count, '/', testlength)

predict_all_time = time.time() - predict_all_time
print('Test Time on Length', len(data.x_test), ':', predict_all_time)

knn_test_err = rmse(np.array(knn_results),data.y_test)

print('Test Standard Deviation: ', np.std(data.y_test))
print('KNN REGRESSION TEST ERROR: ', knn_test_err)


knn_results = []
testlength = len(data.x_train)
count = 0
for x_query in data.x_train:
    result = knn(x_query, data.x_train, best_k)
    knn_results.append(result)
    count += 1
    #print(count, '/', testlength)

knn_train_err = rmse(np.array(knn_results),data.y_train)

print('Train Standard Deviation: ', np.std(data.y_train))
print('KNN REGRESSION TRAIN ERROR: ', knn_train_err)
