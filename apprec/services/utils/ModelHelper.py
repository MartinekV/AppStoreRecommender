import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class ModelHelper(object):
    @staticmethod
    def fit_model_and_transform(model, table, file_name):
        features = np.array(list(table.values()))
        transformed = model.fit_transform(features)

        data = {}
        for id, feature in zip(table.keys(), transformed):
            data[id] = feature

        print(sum(model.explained_variance_ratio_))

        with open(file_name, "wb") as ft:
            pickle.dump(data, ft)

        matplotlib.use('TkAgg')

        # Plotting the Cumulative Summation of the Explained Variance
        plt.figure()
        plt.plot(np.cumsum(model.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title('Explained Variance')
        plt.show()
