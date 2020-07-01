import pandas as pd
import numpy as np
from model.gosdt import GOSDT

dataframe = pd.DataFrame(pd.read_csv("datasets/monk-1/train.csv"))
# dataframe = pd.DataFrame(pd.read_csv("error.csv"))

X = dataframe[dataframe.columns[:-1]]
y = dataframe[dataframe.columns[-1:]]

hyperparameters = {
    "objective": "acc",
    "regularization": 0.1,
    "time_limit": 3600,
    "verbose": True,
    "similar_support": False,
    "strong_indifference": False
}

hyperparameters = {
    "regularization": 0.0001,
    "time_limit": 3600,
    "continuous_feature_exchange": False,
    "feature_exchange": False
}


model = GOSDT(hyperparameters)
model.fit(X, y)
# model.load("models/iris_error.json")
print("Execution Time: {}".format(model.time))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print(model.tree)