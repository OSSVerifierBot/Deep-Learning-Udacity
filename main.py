# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression

import retrieveData

loadData = retrieveData.getData()
trainData = np.reshape(loadData['train_dataset'], (200000, 784))
trainLabels = loadData['train_labels']

testData = np.reshape(loadData['test_dataset'], (10000, 784))
testLabels = loadData['test_labels']

logreg = LogisticRegression()
logreg.fit(trainData, trainLabels)
predData = logreg.predict(testData)

comparison = (predData == testLabels)
nCorrect = 0
for i in comparison:
    if (i):
        nCorrect += 1
print(nCorrect / 100.) # 89.33