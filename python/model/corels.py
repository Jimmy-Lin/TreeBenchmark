import time
import os.path, os, subprocess, sys, re
# from subprocess32 import check_output
import pandas as pd
import time
from corels import *
from model.encoder import Encoder
from model.tree_classifier import TreeClassifier

class CORELS:
    def __init__(self, preprocessor="complete", search="-b", regularization=0.01, max_nodes=10000, symmetry=" -p 1"):
        self.search = search
        self.regularization = regularization
        self.max_nodes = max_nodes
        self.symmetry = symmetry
        self.preprocessor = preprocessor

    def fit(self, X, y):
        self.shape = X.shape

        encoder = Encoder(X.values[:, :], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers

        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)
        self.encoder = encoder

        c = CorelsClassifier(self.regularization, self.max_nodes, verbosity=['rulelist', 'rule'], max_card=1)
        start = time.perf_counter()
        c.fit(X, y.values.ravel())
        self.time = time.perf_counter() - start
        # train = c.score(X, y.values.ravel())

        source = self.__translate__(c.rl().rules)
        self.tree = TreeClassifier(source, encoder=encoder)

        # fargs = fname + ".out " + fname + ".label "
        # command = "../src/corels " + str(self.search) + self.symmetry + " -r " + str(self.regularization)
        # + " -n " + str(self.max_nodes) +  fargs
        # command2 = "./corels -c 1 -p 1 -r 0.01 -n 100000 ../../../bbcache/data/votes.out ../../../bbcache/data/votes.label"
        # output = check_output(command2, stderr=subprocess.STDOUT, timeout=21600, shell=True)
        # time = re.search("(?<=final total time: ).*", output)
        # parsed_tree = re.findall("if \(\{(.*?)\}\) then \(\{(.*?)\}\)|else \(\{(.*?)\}\)", output)
        # print(parsed_tree)

    def __translate__(self, node):
        if len(node) == 1:
            return {
                "name": "class",
                "prediction": int(node[0]["prediction"]),
                "loss": 0,
                "complexity": self.regularization
            }
        else:
            feature = node[0]["antecedents"][0]
            # name = self.encoder.headers[node[0]["antecedents"][0]]
            # for rule in node[0]["antecedents"][1:]:
            #    name += " && " + self.encoder.headers[abs(rule)-1]
            if feature < 0:
                reference = 0
            else:
                reference = 1
            return {
                "feature": abs(feature)-1,
                "name": self.encoder.headers[feature],
                "relation": "==",
                "reference": reference,
                "true": self.__translate_leaf__(node[0]["prediction"]),
                "false": self.__translate__(node[1:])
            }

    def __translate_leaf__(self, prediction):
        return {
            "name": "class",
            "prediction": int(prediction),
            "loss": 0,
            "complexity": self.regularization
        }


    def error(self, X, y, weight=None):
        return self.tree.error(X, y, weight=weight)

    def predict(self, X):
        return self.tree.predict(X)

    def features(self):
        return self.tree.features

    def leaves(self):
        return self.tree.leaves()

    def nodes(self):
        return self.tree.nodes()

    def max_depth(self):
        return self.tree.maximum_depth()



