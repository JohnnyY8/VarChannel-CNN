#coding=utf-8
import os
import numpy as np

basePath = "/home/xlw/second/CNN_ensemble/4training_ensemble/"

pnu = 1
if pnu == 0:
  positiveData = np.load(os.path.join(basePath, "positiveData.npy"))
  positiveNamePairs = np.load(os.path.join(basePath, "positiveNamePairOneCol.npy"))
  print "The shape of positive data and positive namepairs:", positiveData.shape, positiveNamePairs.shape
  data = positiveData
elif pnu == 1:
  negativeData = np.load(os.path.join(basePath, "negativeData.npy"))
  negativeNamePairs = np.load(os.path.join(basePath, "negativeNamePairOneCol.npy"))
  print "The shape of positive data and positive namepairs:", negativeData.shape, negativeNamePairs.shape
  data = negativeData

def ind2Arr(ind):
  return np.array(ind)

length = np.array([0, 0, 0, 0, 0, 0, 0])

for ind, data in enumerate(data):
  shape = np.array(data[ind2Arr([ind])].tolist()).shape
  #if 1 <= shape[-1] <= 7:
  #  print "true"
  #elif shape[0] != 1 or shape[1] != 1956:
  #  print "false"
  length[shape[-1] - 1] += 1
print length
