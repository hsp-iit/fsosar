# read splits/diving/testlist07.txt and splits/diving/trainlist07.txt, where there is a value for each line

import os
import numpy as np
import pandas as pd

# Read the testlist07.txt file with open
testlist07 = []
with open('splits/diving/testlist07.txt', 'r') as f:
    for line in f:
        testlist07.append(line.split('/')[0])
print(set(testlist07))

trainlist07 = []
with open('splits/diving/trainlist07.txt', 'r') as f:
    for line in f:
        trainlist07.append(line.split('/')[0])
print(set(trainlist07))