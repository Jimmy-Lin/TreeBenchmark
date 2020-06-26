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


########################
## SAMPLE SCALABILITY ##
########################

def samples(dataset, algorithms): # Section 5.2.1
    configuration_file_path = "python/configurations/scale.json"
    if not file_exists(configuration_file_path):
        exit(1)
    with open(configuration_file_path, 'r') as configuration_source:
        configuration = configuration_source.read()
        configuration = json.loads(configuration)

    for algorithm in algorithms:
        trial_path = "results/scalability/samples/samples_{}_{}".format(dataset,algorithm)
        result_file_path = "{}.csv".format(trial_path)
        temp_file_path = "{}.tmp".format(trial_path)

        if file_exists(result_file_path):
            continue # This set of trials is already complete

        temp_file = open(temp_file_path, "w")
        temp_file.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

        dataframe = shuffle(pd.DataFrame(pd.read_csv("datasets/{}/data.csv".format(dataset))).dropna(), random_state=0)
        X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
        y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
        (n, m) = X.shape

        timeout = [0]
        time_limit = configuration["time_limit"]

        def record(generator, train_index, test_index):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            (samples, features) = X_train.shape
            binary_features = len(Encoder(X_train.values[:,:], header=X_train.columns[:]).headers)
            if timeout[0] >= 3:
                temp_file.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
                return
            try:
                model = generator()
                model.fit(X_train, y_train)
                if model.time > time_limit:
                    timeout[0] += 1
            except Exception as e:
                print(str(e))
                temp_file.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
                timeout[0] += 1

                print("The following data subset caused an error: error.csv")
                print(X_train, y_train)
                X_train.insert(m, "class", y_train) # It is expected that the last column is the label column
                output = pd.DataFrame(X_train)
                output.to_csv("error.csv", index=False)
                # exit(1)
            else:
                training_accuracy = model.score(X_train, y_train) * 100
                test_accuracy = model.score(X_test, y_test) * 100
                row = [
                    samples, features, binary_features,
                    model.time,
                    model.max_depth(), model.leaves(), model.nodes(),
                    training_accuracy, test_accuracy,
                    model.latex()
                ]
                temp_file.write("NA,{},{},{},{},{},{},{},{},{},{}\n".format(*row))
                print(*row)    

        for sample_size in configuration["samples"]:
            if sample_size > n:
                break
            train_index = [ i for i in range(sample_size) ]
            test_index = [ i for i in range(sample_size, n) ]
            if algorithm == "cart":
                depth, width = configuration["limits"]["depth"], configuration["limits"]["width"]
                record(lambda : CART(depth=depth, width=width), train_index, test_inde)
                    
            elif algorithm == "binoct":
                depth = configuration["limits"]["depth"]
                record(lambda : BinOCT(depth=depth, time_limit=time_limit), train_index, test_index,)                
                
            elif algorithm == "dl85":
                depth = configuration["limits"]["depth"]
                record(lambda : DL85(depth=depth, time_limit=time_limit), train_index, test_inde)                

            elif algorithm == "osdt":
                regularization = configuration["regularization"]
                record(lambda : OSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), train_index, test_inde)                

            elif algorithm == "pygosdt":
                regularization = configuration["regularization"]
                record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), train_index, test_index, )                

            elif algorithm == "gosdt":
                regularization = configuration["regularization"]
                record(lambda : GOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), train_index, test_index)                

        temp_file.close() # temp file is complete
        os.rename(temp_file_path, result_file_path) # commit this file
        print("Trials Completed:", result_file_path)

# #########################
# ## FEATURE SCALABILITY ##
# #########################

def features(dataset, algorithms): # Section 5.2.1
    configuration_file_path = "python/configurations/scale.json"
    if not file_exists(configuration_file_path):
        exit(1)
    with open(configuration_file_path, 'r') as configuration_source:
        configuration = configuration_source.read()
        configuration = json.loads(configuration)

    for algorithm in algorithms:
        trial_path = "results/scalability/features/features_{}_{}".format(dataset,algorithm)
        result_file_path = "{}.csv".format(trial_path)
        temp_file_path = "{}.tmp".format(trial_path)

        if file_exists(result_file_path):
            continue # This set of trials is already complete

        timeout = [0]
        time_limit = configuration["time_limit"]

        temp_file = open(temp_file_path, "w")
        temp_file.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

        dataframe = shuffle(pd.DataFrame(pd.read_csv("datasets/{}/data.csv".format(dataset))).dropna(), random_state=0)
        X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
        y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
        (n, m) = X.shape

        encoder = Encoder(X.values[:,:], header=X.columns[:])
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        (n, z) = X.shape

        sample_size = int( 0.9 * n )
        train_index = [ i for i in range(sample_size) ]
        test_index = [ i for i in range(sample_size, n) ]
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        def record(generator, feature_index):
            (samples, features) = X_train.shape
            binary_features = len(feature_index)
            if timeout[0] >= 3:
                temp_file.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
                return
            try:
                model = generator()
                model.fit(X.iloc[train_index,feature_index].reset_index(drop=True), y.iloc[train_index].reset_index(drop=True))
                if model.time > time_limit:
                    timeout[0] += 1
            except Exception as e:
                print(str(e))
                temp_file.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
                timeout[0] += 1

                print("The following data subset caused an error: error.csv")
                X_error = X.copy().iloc[train_index,feature_index].reset_index(drop=True)
                X_error.insert(binary_features, "class", y.iloc[train_index].reset_index(drop=True))
                output = pd.DataFrame(X_error)
                output.to_csv("error.csv", index=False)


            else:
                training_accuracy = model.score(X.iloc[train_index,feature_index], y.iloc[train_index]) * 100
                test_accuracy = model.score(X.iloc[test_index,feature_index], y.iloc[test_index]) * 100
                row = [
                    samples, features, binary_features,
                    model.time,
                    model.max_depth(), model.leaves(), model.nodes(),
                    training_accuracy, test_accuracy,
                    model.latex()
                ]
                temp_file.write("NA,{},{},{},{},{},{},{},{},{},{}\n".format(*row))
                print(*row)    

        for k in configuration["features"]:
            if k > z:
                k = z

            feature_index = [ i for i in range(k) ]

            if algorithm == "cart":
                depth, width = configuration["limits"]["depth"], configuration["limits"]["width"]
                record(lambda : CART(depth=depth, width=width), feature_index)
                    
            elif algorithm == "binoct":
                depth = configuration["limits"]["depth"]
                record(lambda : BinOCT(depth=depth, time_limit=time_limit), feature_index)                
                
            elif algorithm == "dl85":
                depth = configuration["limits"]["depth"]
                record(lambda : DL85(depth=depth, time_limit=time_limit), feature_index)                

            elif algorithm == "osdt":
                regularization = configuration["regularization"]
                record(lambda : OSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), feature_index)                

            elif algorithm == "pygosdt":
                regularization = configuration["regularization"]
                record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), feature_index)

            elif algorithm == "gosdt":
                regularization = configuration["regularization"]
                record(lambda : GOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": 1 }), feature_index)                

            if k == z:
                break

        temp_file.close() # temp file is complete
        os.rename(temp_file_path, result_file_path) # commit this file
        print("Trials Completed:", result_file_path)



# def features(datasets, algorithms):
#     configuration_file_path = "python/configurations/configurations.json"
#     if not file_exists(configuration_file_path):
#         exit(1)
#     with open(configuration_file_path, 'r') as configuration_source:
#         configuration = configuration_source.read()
#         configuration = json.loads(configuration)

#     for dataset in datasets:
#         dataset_file_path = "datasets/{}/data.csv".format(dataset)
#         if not file_exists(dataset_file_path):
#             exit(1)

#         dataframe = pd.DataFrame(pd.read_csv(dataset_file_path)).dropna()
#         X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
#         y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
#         (n, m) = X.shape
#         classes = len(set(dataframe.iloc[:,-1]))
#         binary_features = sum(len(set(dataframe.iloc[:,j])) for j in range(m-1)) # Note: This is a rough estimate

#         for algorithm in algorithms:
#             trial_path = "results/tradeoff/{}_{}".format(dataset,algorithm)
#             result_file_path = "{}.csv".format(trial_path)
#             temp_file_path = "{}.tmp".format(trial_path)

#             if file_exists(result_file_path):
#                 continue # This set of trials is already complete

#             temp_file = open(temp_file_path, "w")
#             temp_file.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

#             failures = [0] # Awkward forcing of pass-by-reference into python closures

#             def trial(generator, row_generator, default_row_generator):
#                 kfolds = KFold(n_splits=configuration["trials"], random_state=0)
#                 fold = 0

#                 for train_index, test_index in kfolds.split(X):
#                     X_train, y_train = X.iloc[train_index], y.iloc[train_index]
#                     X_test, y_test = X.iloc[test_index], y.iloc[test_index]
#                     try:
#                         if failures[0] < 3:
#                             model = generator()
#                             model.fit(X_train, y_train)
#                             train = (1.0 - model.error(X_train, y_train)) * 100
#                             test = (1.0 - model.error(X_test, y_test)) * 100
#                             row = row_generator(fold, model.time, model.max_depth(), model.leaves(), model.nodes(), train, test, model.latex(), '')
#                             print(row)
#                             temp_file.write(",".join(str(e) for e in row) + "\n")
#                         else:
#                             default_row = default_row_generator(fold)
#                             temp_file.write(",".join(str(e) for e in default_row) + "\n")
#                     except Exception as e:
#                         failures[0] += 1
#                         print(str(e))
#                         default_row = default_row_generator(fold)
#                         temp_file.write(",".join(str(e) for e in default_row) + "\n")

#                         print(default_row)
#                         print("The following data subset caused an error: error.csv")
#                         print(X_train, y_train)
#                         X_train.insert(m, "class", y_train) # It is expected that the last column is the label column
#                         output = pd.DataFrame(X_train)
#                         output.to_csv("error.csv", index=False)
#                         # exit(1)
                        
#                     fold += 1

#             if algorithm == "dl85":
#                 for limit in configuration["limits"]:
#                     def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

#                     def default_row(fold_index):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             -1, -1, -1, -1, "NA", "NA"]
#                     trial(lambda : DL85(depth=limit["depth"], time_limit=configuration["time_limit"], preprocessor="complete"))  

#             elif algorithm == "osdt":
#                 for regularization in configuration["regularization"]:
#                     config = {
#                         "regularization": regularization,
#                         "time_limit": configuration["time_limit"]
#                     }
#                     def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

#                     def default_row(fold_index):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             -1, -1, -1, -1, "NA", "NA"]
#                     trial(lambda : OSDT(config, preprocessor="complete"))                

#             elif algorithm == "gosdt":
#                 for regularization in configuration["regularization"]:
#                     config = {
#                         "regularization": regularization,
#                         "time_limit": configuration["time_limit"]
#                     }

#                     def row(fold_index, training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             training_time, max_depth, leaves, nodes, training_accuracy, test_accuracy, latex, source]

#                     def default_row(fold_index):
#                         return [dataset,"complete",classes,n,m,binary_features,algorithm,"None","None",regularization,configuration["time_limit"],"leviathan",1,fold_index,n,m,binary_features,
#                             -1, -1, -1, -1, "NA", "NA"]

#                     trial(lambda : GOSDT(config), row, default_row)                

#             temp_file.close() # temp file is complete
#             os.rename(temp_file_path, result_file_path) # commit this file
#             print("Trials Completed:", result_file_path)
