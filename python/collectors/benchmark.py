import numpy as np
import pandas as pd
import json
import time
import random
import sys
import os  

from math import ceil
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from time import sleep

from model.dl85 import DL85
from model.gosdt import GOSDT
from model.osdt import OSDT
from model.encoder import Encoder
from model.corels import CORELS

def file_exists(path):
    exists = False
    file_descriptor = None
    try:
        file_descriptor = open(path)
        # Do something with the file
        exists = True
    except IOError:
        # print("File '{}' not found.".format(path))
        pass
    finally:
        if not file_descriptor is None:
            file_descriptor.close()
    return exists

def benchmark(datasets, algorithms):
    configuration_file_path = "python/configurations/benchmark.json"
    if not file_exists(configuration_file_path):
        exit(1)
    with open(configuration_file_path, 'r') as configuration_source:
        configuration = configuration_source.read()
        configuration = json.loads(configuration)

    for dataset in datasets:
        dataset_file_path = "datasets/{}/data.csv".format(dataset)
        if not file_exists(dataset_file_path):
            exit(1)
        for algorithm in algorithms:
            for regularization in configuration["regularization"]:
                trial_path = "results/benchmark/trials_{}_{}_{}".format(dataset,algorithm,regularization)
                result_file_path = "{}.csv".format(trial_path)
                temp_file_path = "{}.tmp".format(trial_path)

                if file_exists(result_file_path):
                    continue # This set of trials is already complete

                temp_file = open(temp_file_path, "w")
                temp_file.write("algorithm,regularization,fold,train,test,depth,leaves,nodes,time\n")

                dataframe = pd.DataFrame(pd.read_csv(dataset_file_path)).dropna()
                X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
                y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
                (n, m) = X.shape

                def trial(generator):
                    kfolds = KFold(n_splits=configuration["trials"], random_state=0)
                    fold = 0
                    for train_index, test_index in kfolds.split(X):
                        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
                        try:
                            model = generator()
                            model.fit(X_train, y_train)
                        except Exception as e:
                            print(str(e))
                            default_row = [algorithm,regularization,fold,0,0,0,0,0,0]
                            temp_file.write(",".join(str(e) for e in default_row) + "\n")
                            print(default_row)
                            print("The following data subset caused an error: error.csv")
                            print(X_train, y_train)
                            X_train.insert(m, "class", y_train) # It is expected that the last column is the label column
                            output = pd.DataFrame(X_train)
                            output.to_csv("error.csv", index=False)
                            exit(1)
                        else:
                            train = model.error(X_train, y_train)
                            test = model.error(X_test, y_test)
                            row = [
                                algorithm,regularization,fold,
                                train,test,
                                model.max_depth(), model.leaves(), model.nodes(),
                                model.time
                            ]
                            print(row)
                            temp_file.write(",".join(str(e) for e in row) + "\n")
                        fold += 1

                if algorithm == "dl85":
                    trial(lambda : DL85(regularization=regularization, time_limit=configuration["time_limit"], preprocessor="complete"))                

                elif algorithm == "osdt":
                    config = {
                        "regularization": regularization,
                        "time_limit": configuration["time_limit"]
                    }
                    trial(lambda : OSDT(config, preprocessor="complete"))                

                elif algorithm == "gosdt":
                    config = {
                        "regularization": regularization,
                        "similar_support": False,
                        "strong_indifference": False,
                        "time_limit": configuration["time_limit"]
                    }
                    trial(lambda : GOSDT(config))

                elif algorithm == "corels":
                    trial(lambda : CORELS(regularization=regularization))

                temp_file.close() # temp file is complete
                os.rename(temp_file_path, result_file_path) # commit this file
                print("Trials Completed:", result_file_path)
