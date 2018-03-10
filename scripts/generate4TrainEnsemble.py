#coding=utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split

cls = ["VCAP", "HCC515", "A375", "A549", "HA1E", "PC3", "HEPG2"]

dataPath = "../files/4training/"
savePath = "../files/4training_ensemble/"
path4Prediction = "../files/newMap4UnlabeledData"

def filterDataAndNamePair(data, namePair, keepList):
  resData, resNamePair = [], []
  for ind, ele in enumerate(namePair):
    if ele in keepList:
      resData.append(data[ind])
      resNamePair.append(ele)
  return np.array(resData), np.array(resNamePair)

def ind2Arr(index):
  return np.array(index)

def putTogether(oldData, oldNamePairs, newData, newNamePairs):
  resData, resNamePairs = [], []
  while(newNamePairs.shape[0] != 0):
    namePair = newNamePairs[0]
    index = np.where(oldNamePairs == namePair)
    if index[0].shape[0] == 0:
      print "Not found..."
      resData.append(newData[ind2Arr([0])].reshape(-1, 1).tolist())
    else:
      print "Found..."
      tempData = np.array(np.array(oldData[index]).tolist())
      if tempData.size == 1956:
        print "  The size is 1956..."
        tempData = tempData.reshape(1, tempData.size, 1)
      else:
        print "  The size is:", tempData.size
        tempData = tempData.reshape(1, 1956, tempData.size / 1956)
      tempData = np.dstack((tempData, newData[0])).tolist()
      resData.append(tempData[0])
      oldData = np.delete(oldData, index, axis = 0)
      oldNamePairs = np.delete(oldNamePairs, index, axis = 0)
    resNamePairs.append(namePair)
    newData = np.delete(newData, 0, axis = 0)
    newNamePairs = np.delete(newNamePairs, 0, axis = 0)
    print "The shape of resData and resNamePairs:", np.array(resData).shape, np.array(resNamePairs).shape

  while(oldNamePairs.shape[0] != 0):
    tempData = np.array(np.array(oldData[ind2Arr(0)]).tolist())
    if tempData.size == 1956:
      tempData = tempData.reshape(1, tempData.size, 1)
    else:
      tempData = tempData.reshape(1, 1956, tempData.size / 1956)
    resData.append(tempData[0].tolist())
    resNamePairs.append(oldNamePairs[0])
    oldData = np.delete(oldData, 0, axis = 0)
    oldNamePairs = np.delete(oldNamePairs, 0, axis = 0)
    print "The shape of resData and resNamePairs:", np.array(resData).shape, np.array(resNamePairs).shape

  return np.array(resData), np.array(resNamePairs)

if __name__ == "__main__":
  # 1.load all data from cell lines
  # 2.stack all drug-gene pairs in all cell lines
  # 3.train test split
  pnu = 2
  flag = 0
  for cl in cls:
    print cl
    if flag == 0:
      if pnu == 0:
        clNDPath = os.path.join(dataPath, cl, "negativeData.npy")
        negativeData = np.load(clNDPath)
        print "negativeData is done..."

        clNNPath = os.path.join(dataPath, cl, "negativeNamePairOneCol.npy")
        negativeNamePair = np.load(clNNPath)
        print "negativeNamePair is done..."

      elif pnu == 1:
        clPDPath = os.path.join(dataPath, cl, "positiveData.npy")
        positiveData = np.load(clPDPath)
        print "positiveData is done..."

        clPNPath = os.path.join(dataPath, cl, "positiveNamePairOneCol.npy")
        positiveNamePair = np.load(clPNPath)
        print "positiveNamePair is done..."

      elif pnu == 2:
        clUDPath = os.path.join(dataPath, cl, "unlabeledData.npy")
        unlabeledData = np.load(clUDPath)
        print "unlabeledData is done..."

        clUNPath = os.path.join(dataPath, cl, "unlabeledNamePairOneCol.npy")
        unlabeledNamePair = np.load(clUNPath)
        print "unlabeledNamePair is done..."

        keepList = np.load(os.path.join(path4Prediction, "unlabeledNamePairOneCol4Prediction_" + cl + ".npy"))
        unlabeledData, unlabeledNamePair = filterDataAndNamePair(unlabeledData, unlabeledNamePair, keepList)

      flag += 1

    elif flag == 1:
      if pnu == 0:
        clNDPath = os.path.join(dataPath, cl, "negativeData.npy")
        newNegativeData = np.load(clNDPath)
        print "negativeData is done..."

        clNNPath = os.path.join(dataPath, cl, "negativeNamePairOneCol.npy")
        newNegativeNamePair = np.load(clNNPath)
        print "negativeNamePair is done..."

        negativeData, negativeNamePair = putTogether(negativeData, negativeNamePair, newNegativeData, newNegativeNamePair)
        print "Put negative data and name pairs together is done..."

      elif pnu == 1:
        clPDPath = os.path.join(dataPath, cl, "positiveData.npy")
        newPositiveData = np.load(clPDPath)
        print "positiveData is done..."

        clPNPath = os.path.join(dataPath, cl, "positiveNamePairOneCol.npy")
        newPositiveNamePair = np.load(clPNPath)
        print "positiveNamePair is done..."

        positiveData, positiveNamePair = putTogether(positiveData, positiveNamePair, newPositiveData, newPositiveNamePair)
        print "Put positive data and name pairs together is done..."

      elif pnu == 2:
        keepList = np.load(os.path.join(path4Prediction, "unlabeledNamePairOneCol4Prediction_" + cl + ".npy"))

        clUDPath = os.path.join(dataPath, cl, "unlabeledData.npy")
        newUnlabeledData = np.load(clUDPath)
        print "unlabeledData is done..."

        clUNPath = os.path.join(dataPath, cl, "unlabeledNamePairOneCol.npy")
        newUnlabeledNamePair = np.load(clUNPath)
        print "unlabeledNamePair is done..."

        #print newUnlabeledData.shape, newUnlabeledNamePair.shape
        keepList = np.load(os.path.join(path4Prediction, "unlabeledNamePairOneCol4Prediction_" + cl + ".npy"))
        newUnlabeledData, newUnlabeledNamePair = filterDataAndNamePair(newUnlabeledData, newUnlabeledNamePair, keepList)
        #print newUnlabeledData.shape, newUnlabeledNamePair.shape
        #raw_input("....")

        unlabeledData, unlabeledNamePair = putTogether(unlabeledData, unlabeledNamePair, newUnlabeledData, newUnlabeledNamePair)
        print "Put unlabeled data and name pairs together is done..."

  if pnu == 0:
    np.save(savePath + "negativeData.npy", negativeData)
    np.save(savePath + "negativeNamePairOneCol.npy", negativeNamePair)
  elif pnu == 1:
    np.save(savePath + "positiveData.npy", positiveData)
    np.save(savePath + "positiveNamePairOneCol.npy", positiveNamePair)
  elif pnu == 2:
    np.save(savePath + "unlabeledData.npy", unlabeledData)
    np.save(savePath + "unlabeledNamePairOneCol.npy", unlabeledNamePair)
