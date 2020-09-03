import csv
import numpy as np

class SarcosData:

    x_train = None
    y_train = None
    x_validation = None
    y_validation = None
    x_test = None
    y_test = None

    def __init__(self,proportion=1.0,split=0.8,valid_split=0.9):

        assert 0 < proportion <= 1

        data = []

        with open('sarcos_inv.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data.append(row)

        data = np.array(data).astype(np.float)

        # Erase any ordering bias
        np.random.seed(0)
        data = self.shuffle_data(data)

        rows = int(len(data) * proportion)
        data = data[:rows]

        main_split = int(split*rows)
        training_validation = data[:main_split]
        test = data[main_split:]

        validation_split = int(valid_split * len(training_validation))
        training = training_validation[:validation_split]
        validation = training_validation[validation_split:]

        self.x_train = training[:,:-1]
        self.y_train = training[:,-1]
        self.x_validation = validation[:, :-1]
        self.y_validation = validation[:, -1]
        self.x_test = test[:,:-1]
        self.y_test = test[:,-1]

    def get_train(self):
        return self.x_train,self.y_train

    def get_validation(self):
        return self.x_validation,self.y_validation

    def get_test(self):
        return self.x_test, self.y_test


    def shuffle_data(self,data):
        total_length = len(data)
        shuffle = np.random.permutation(total_length)
        return data[shuffle]

    '''
    def random_sub_sample(self,split=0.8):

        temp_training = self.shuffle_data(self.training)
        split_point = int(split * len(temp_training))
        train = temp_training[:split_point]
        validation = temp_training[split_point:]

        x_train = train[:,:-1]
        y_train = train[:,-1]

        x_validation = validation[:,:-1]
        y_validation = validation[:,-1]

        return x_train,y_train,x_validation,y_validation 
    '''