#coding=utf-8
import os
import numpy as np

cls = ["VCAP", "PC3", "A375", "A549", "HA1E", "HCC515", "HEPG2"]

basePath = "/home/xlw/second/CNN_ensemble/after_merge/"
basePath = "/home/xlw/second/CNN_ensemble/4training/"

res = {}

for cl in cls:
  clPath = os.path.join(basePath, cl, "rngene_name.npy")
  clPath = os.path.join(basePath, cl, "positiveNamePairOneCol.npy")
  clPath = os.path.join(basePath, cl, "negativeNamePairOneCol.npy")
  drugNames = np.load(clPath)
  print "num of drugs in cl:", cl, drugNames.shape
  for drugName in drugNames:
    if not res.has_key(drugName):
      res[drugName] = 1
    else:
      res[drugName] += 1

length = np.array([0, 0, 0, 0, 0, 0, 0])
for key in res:
  #print "The key and value:", key, res[key]
  length[res[key] - 1] += 1
print len(res)
print "The total results:", length
