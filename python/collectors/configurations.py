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

def file_exists(path):
    exists = False
    file_descriptor = None
    try:
        file_descriptor = open(path)
        # Do something with the file
        exists = True
    except IOError:
        print("File '{}' not found.".format(path))
        pass
    finally:
        if not file_descriptor is None:
            file_descriptor.close()
    return exists

def configurations(datasets, algorithms):
    configuration_file_path = "python/configurations/configurations.json"
    if not file_exists(configuration_file_path):
        exit(1)
    with open(configuration_file_path, 'r') as configuration_source:
        configuration = configuration_source.read()
        configuration = json.loads(configuration)

    

    for dataset in datasets:
        dataset_file_path = "datasets/{}/data.csv".format(dataset)
        if not file_exists(dataset_file_path):
            exit(1)

        dataframe = pd.DataFrame(pd.read_csv(dataset_file_path)).dropna()
        X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
        y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
        (n, m) = X.shape
        classes = len(set(dataframe.iloc[:,-1]))
        binary_features = sum(len(set(dataframe.iloc[:,j])) for j in range(m-1)) # Note: This is a rough estimate

        for algorithm in algorithms:
            trial_path = "results/tradeoff/{}_{}".format(dataset,algorithm)
            result_file_path = "{}.csv".format(trial_path)
            temp_file_path = "{}.tmp".format(trial_path)

            if file_exists(result_file_path):
                continue # This set of trials is already complete

            temp_file = open(temp_file_path, "w")
            # temp_file.write("algorithm,regularization,fold,train,test,depth,leaves,nodes,time\n")
            temp_file.write("Data Set,Preprocessing,Classes,Samples,Features,Binary Features,Algorithm,Depth Limit,Width Limit,Regularization,Time Limit,Machine,Threads,Fold Index,Subsample,Subfeatures,Binary subfeatures,Training Time,Max Depth,# Leaves,# Nodes,Training Accuracy,Test Accuracy,Latex,Source\n")

            failures = [0] # Awkward forcing of pass-by-reference into python closures

            def trial(generator, row_generator, default_row_generator):
                kfolds = KFold(n_splits=configuration["trials"], random_state=0)
                fold = 0

                for train_index, test_index in kfolds.split(X):
                    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
                    try:
                        if failures[0] < 3:
                            model = generator()
                            model.fit(X_train, y_train)
                            train = (1.0 - model.error(X_train, y_train)) * 100
                            test = (1.0 - model.error(X_test, y_test)) * 100
                            row = row_generator(fold, model.time, model.max_depth(), model.leaves(), model.nodes(), train, test, model.latex(), '')
                            print(row)
                            temp_file.write(",".join(str(e) for e in row) + "\n")
                        else:
                            default_row = default_row_generator(fold)
                            temp_file.write(",".join(str(e) for e in default_row) + "\n")
                    except Exception as e:
                        failures[0] += 1
                        print(str(e))
                        default_row = default_row_generator(fold)
                        temp_file.write(",".join(str(e) for e in default_row) + "\n")

                        print(default_row)
                        print("The following data subset caused an error: error.csv")
                        print(X_train, y_train)
                        X_train.insert(m, "class", y_train) # It is expected that the last column is the label column
                        output = pd.DataFrame(X_train)
                        output.to_csv("error.csv", index=False)
                        # exit(1)
                        
                    fold += 1

            if algorithm == "dl85":
                for limit in configuration["limits"]:
                    def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

                    def default_row(fold_index):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            -1, -1, -1, -1, "NA", "NA"]
                    trial(lambda : DL85(depth=limit["depth"], time_limit=configuration["time_limit"], preprocessor="complete"))  

            elif algorithm == "osdt":
                for regularization in configuration["regularization"]:
                    config = {
                        "regularization": regularization,
                        "time_limit": configuration["time_limit"]
                    }
                    def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

                    def default_row(fold_index):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            -1, -1, -1, -1, "NA", "NA"]
                    trial(lambda : OSDT(config, preprocessor="complete"))                

            elif algorithm == "gosdt":
                for regularization in configuration["regularization"]:
                    config = {
                        "regularization": regularization,
                        "time_limit": configuration["time_limit"]
                    }

                    def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

                    def default_row(fold_index):
                        return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
                            -1, -1, -1, -1, "NA", "NA"]

                    trial(lambda : GOSDT(config), row, default_row)                

            temp_file.close() # temp file is complete
            os.rename(temp_file_path, result_file_path) # commit this file
            print("Trials Completed:", result_file_path)
