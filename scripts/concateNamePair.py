import os
import numpy as np

cls = ["VCAP", "PC3", "A375", "A549", "HA1E", "HCC515", "HEPG2"]

basePath = "/home/xlw/second/CNN_ensemble/4training/"

def mergeString(cl, namePairs, fileName):
  resNamePair = np.array([])
  print namePairs.shape
  for i in namePairs:
    iele, jele = i[0], i[1]
    #print iele, jele, "..."
    temp = iele + jele
    if resNamePair.shape[0] == 0:
      resNamePair = np.array([temp])
    else:
      resNamePair = np.append(resNamePair, temp)
  savePath = os.path.join(basePath, cl, fileName)
  print savePath
  np.save(savePath, resNamePair)

if __name__ == "__main__":
  for cl in cls:
    print cl
    clPath = os.path.join(basePath, cl, "negativeNamePair.npy")
    mergeString(cl, np.load(clPath), "negativeNamePairOneCol.npy")
    clPath = os.path.join(basePath, cl, "positiveNamePair.npy")
    mergeString(cl, np.load(clPath), "positiveNamePairOneCol.npy")
    clPath = os.path.join(basePath, cl, "unlabeledNamePair.npy")
    mergeString(cl, np.load(clPath), "unlabeledNamePairOneCol.npy")
