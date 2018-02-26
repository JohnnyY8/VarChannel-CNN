import os
import random
import numpy as np

cls = ["VCAP", "PC3", "A375", "A549", "HA1E", "HCC515", "HEPG2"]
baseDataPath = "/home/xlw/second/CNN_ensemble/after_merge/"
baseSavePath = "/home/xlw/second/CNN_ensemble/4training/"

def mergeDataAndName(tpData, drugData, tpName, drugName, num4Nonzero, dic4DN):
  resData, resName = [], []
  num4Neg = num4Nonzero * 2
  num4TP, num4Drug = tpData.shape[0], drugData.shape[0]
  tpName, drugName = tpName.reshape(-1), drugName.reshape(-1)

  newDrugName, newDrugData = np.array([]), np.array([])
  for i, dn in enumerate(drugName):
    if dic4DN[dn] == 7:
      newDrugName = np.append(newDrugName, dn)
      if newDrugData.shape[0] == 0:
        newDrugData = drugData[i]
      else:
        newDrugData = np.vstack((newDrugData, drugData[i]))

  sTpName = np.sort(tpName, axis = 0)
  sDrugName = np.sort(newDrugName, axis = 0)
  for stn in sTpName[: 90]:
    ind4TP = np.where(tpName == stn)
    for sdn in sDrugName[: 90]:
      ind4Drug = np.where(newDrugName == sdn)
      tempData = np.append(tpData[ind4TP], newDrugData[ind4Drug])
      resData.append(tempData)
      resName.append(np.append(stn, sdn))
  resData, resName = np.array(resData), np.array(resName)
  print "The shape of resData and resName:", resData.shape, resName.shape

  return resData[: num4Neg], resName[: num4Neg]

def countOverlap():
  res = {}
  for cl in cls:
    #clPath = os.path.join(baseDataPath, cl, "rngene_name.npy")
    #clPath = os.path.join(baseSavePath, cl, "positiveNamePairOneCol.npy")
    #clPath = os.path.join(baseSavePath, cl, "negativeNamePairOneCol.npy")
    clPath = os.path.join(baseDataPath, cl, "drug_name.npy")
    drugNames = np.load(clPath)
    for drugName in drugNames:
      if not res.has_key(drugName):
        res[drugName] = 1
      else:
        res[drugName] += 1
  
  length = np.array([0, 0, 0, 0, 0, 0, 0])
  for key in res:
    length[res[key] - 1] += 1
  print len(res)
  print length
  return res

if __name__ == "__main__":

  dic4DN = countOverlap()

  for cl in cls:
    print(cl)
    clDataPath = os.path.join(baseDataPath, cl)
    clSavePath = os.path.join(baseSavePath, cl)
    label = np.load(clDataPath + "/new_label.npy")
    drugData, drugName = np.load(clDataPath + "/drug_data.npy"), np.load(clDataPath + "/drug_name.npy").reshape(-1, 1)
    tpData, tpName = np.load(clDataPath + "/tp_data.npy"), np.load(clDataPath + "/tp_name.npy").reshape(-1, 1)
    rngData, rngName = np.load(clDataPath + "/rngene_data_sampled.npy"), np.load(clDataPath + "/rngene_name_sampled.npy").reshape(-1, 1)
    
    ind4one4tp, ind4one4drug = np.where(label == 1)[0], np.where(label == 1)[1]
    ind4zero4tp, ind4zero4drug = np.where(label == 0)[0], np.where(label == 0)[1]
    
    positiveData = np.hstack((tpData[ind4one4tp], drugData[ind4one4drug]))
    np.save(clSavePath + "/positiveData.npy", positiveData)
    print("positiveData is done...")
    positiveNamePair = np.hstack((tpName[ind4one4tp], drugName[ind4one4drug]))
    np.save(clSavePath + "/positiveNamePair.npy", positiveNamePair)
    print("positiveNamePair is done...")

    negativeData, negativeNamePair = mergeDataAndName(rngData, drugData, rngName, drugName, ind4one4tp.shape[0], dic4DN)
    np.save(clSavePath + "/negativeData.npy", negativeData)
    print("negativeData is done...")
    np.save(clSavePath + "/negativeNamePair.npy", negativeNamePair)
    print("negativeNamePair is done...")

    unlabeledData = np.hstack((tpData[ind4zero4tp], drugData[ind4zero4drug]))
    np.save(clSavePath + "/unlabeledData.npy", unlabeledData)
    print("unlabeledData is done...")
    unlabeledNamePair = np.hstack((tpName[ind4zero4tp], drugName[ind4zero4drug]))
    np.save(clSavePath + "/unlabeledNamePair.npy", unlabeledNamePair)
    print("unlabeledNamePair is done...")
