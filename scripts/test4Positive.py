#coding=utf-8
import os
import numpy as np

basePath = "/home/xlw/second/CNN_ensemble/4training_ensemble/"

negativeData = np.load(os.path.join(basePath, "positiveData.npy"))
negativeNamePairs = np.load(os.path.join(basePath, "positiveNamePairOneCol.npy"))
#negativeData = np.load(os.path.join(basePath, "negativeData.npy"))
#negativeNamePairs = np.load(os.path.join(basePath, "negativeNamePairOneCol.npy"))

#print "The shape of positive data and positive namepairs:", positiveData.shape, positiveNamePairs.shape
print "The shape of positive data and positive namepairs:", negativeData.shape, negativeNamePairs.shape

def ind2Arr(ind):
  return np.array(ind)

length = np.array([0, 0, 0, 0, 0, 0, 0])

for ind, data in enumerate(negativeData):
  shape = np.array(negativeData[ind2Arr([ind])].tolist()).shape
  #if 1 <= shape[-1] <= 7:
  #  print "true"
  #elif shape[0] != 1 or shape[1] != 1956:
  #  print "false"
  length[shape[-1] - 1] += 1
print length
  #if len(np.array(positiveData[ind2Arr([ind])].tolist()).shape) != 3:
  #  print "exp"
  #print data
  #print type(data)
  #print data.shape
  #print data[np.array([0])].shape
#print "------"
#print np.array(positiveData[0].tolist()).shape
#print "------"
#print np.array(positiveData[np.array([0])].tolist()).shape
