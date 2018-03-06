# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression

import retrieveData

loadData = retrieveData.getData()
n = 50
trainData = np.reshape(loadData['train_dataset'][0:n], (n, 784))
trainLabels = loadData['train_labels'][0:n]

testData = np.reshape(loadData['test_dataset'][0:50], (50, 784))
testLabels = loadData['test_labels'][0:50]

logreg = LogisticRegression()
logreg.fit(trainData, trainLabels)
predData = logreg.predict(testData)

print(trainLabels)
print(testLabels)

comparison = (predData == testLabels)
nCorrect = 0
for i in comparison:
  if (i):
    nCorrect += 1
print(nCorrect * 100. / n)