import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from model import MyModel

if __name__ == '__main__':
    np.random.seed(1234)

    PATH = 'normal_1700.csv'

    data = pd.read_csv(PATH)
    index = data.iloc[:,1]

    # K Fold
    kf = KFold(n_splits=3, shuffle=False)

    k = 0

    for train, valid in kf.split(index):

        data = pd.read_csv(PATH)

        f_total = data.iloc[:, 1]  # Frequency
        I_total = data.iloc[:, 0]  # Intensity
        label_total = data.iloc[:, 2]  # Labels

        train_index = train
        valid_index = valid

        f_data = []
        I_data = []
        label_data = []

        f_data_t = []
        I_data_t = []
        label_data_t = []

        # Create train data
        for i in train_index:
            dev_f = f_total[i]
            dev_I = I_total[i]
            dev_label = label_total[i]
            f_data.append(dev_f)
            I_data.append(dev_I)
            label_data.append(dev_label)

        f = np.transpose(np.array([f_data]))
        I = np.transpose(np.array([I_data]))
        label = np.transpose(np.array([label_data]))
        train_data = np.hstack((f, I))
        train_label = label

        data_train = np.hstack((train_data, train_label))
        train = pd.DataFrame(data_train, columns=['I', 'f', 'label'])
        train.to_csv('datasets/train/train_{}.csv'.format(k),sep=",",index=False)

        # Create test data
        for n in valid_index:
            val_f = f_total[n]
            val_I = I_total[n]
            val_label = label_total[n]
            f_data_t.append(val_f)
            I_data_t.append(val_I)
            label_data_t.append(val_label)

        f_test = np.transpose(np.array([f_data_t]))
        I_test = np.transpose(np.array([I_data_t]))
        label_test = np.transpose(np.array([label_data_t]))
        test_data = np.hstack((f_test, I_test))
        test_label = label_test

        data_test = np.hstack((test_data, test_label))
        test = pd.DataFrame(data_test, columns=['I', 'f', 'label'])
        test.to_csv('datasets/test/test_{}.csv'.format(k), sep=",", index=False)

        k = k+1

        # Parameters
        ite_num =1000

        # Model
        layers = [2, 20,20,20, 3]
        model = MyModel(layers, train_data, train_label, test_data, test_label)

        # Train
        model.train(ite_num)

        # Test
        model.predict(test_data, test_label)

