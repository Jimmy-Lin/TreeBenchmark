import sys
import os
import re
import numpy as np
import pandas as pd

input_path = sys.argv[1]
output_path = sys.argv[2]
number_of_rows = 0
number_of_features = 0

features = []
feature_names = []
targets = []
target_names = []

with open(input_path + '.label', 'r') as file:
    # data = file.read()
    for line in file:
        match = re.search('(\{.*?\})(.*)', line)
        target_names.append(match.group(1).replace(',', ';'))
        targets.append(match.group(2).strip().split(' '))
        number_of_rows = len(targets[0])

with open(input_path + '.out', 'r') as file:
    for line in file:
        number_of_features += 1
        match = re.search('(\{.*?\})(.*)', line)
        feature_names.append(match.group(1).replace(',', ';'))
        features.append(match.group(2).strip().split(' '))

print("Number of Rows:", number_of_rows)
print("Number of Features:", number_of_features)

header = feature_names + [ target_names[0] ]
data = pd.DataFrame(np.array([ [ features[j][i] for j in range(number_of_features) ] + [ targets[0][i] ] for i in range(number_of_rows) ]))
data.to_csv(output_path, header = header, index= False)
