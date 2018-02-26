import os
import random
import numpy as np

cls = ["PC3", "VCAP", "A375", "A549", "HA1E", "HCC515", "HEPG2"]
baseDataFilePath = "/home/xlw/second/CNN_ensemble/after_merge/"
baseSavePath = "/home/xlw/second/CNN_ensemble/4training/"

if __name__ == "__main__":
  dic4RNG = {}
  for cl in cls:
    clPath = os.path.join(baseDataFilePath, cl, "rngene_name.npy")
    rngeneNames = np.load(clPath)
    print "num of rngenes in cl:", cl, rngeneNames.shape
    for rngeneName in rngeneNames:
      if not dic4RNG.has_key(rngeneName):
        dic4RNG[rngeneName] = 1
      else:
        dic4RNG[rngeneName] += 1

  #length = np.array([0, 0, 0, 0, 0, 0, 0])
  #for key in res:
  #  length[res[key] - 1] += 1
  #print len(res)
  #print "The total results:", length

  for cl in cls:
    print(cl)
    clPath = os.path.join(baseDataFilePath, cl)
    label = np.load(clPath + "/new_label.npy")
    print "The shape of label and nonzero:", label.shape, np.nonzero(label)[0].shape[0]
    num4RNG = np.nonzero(label)[0].shape[0] * 2
    print "num4RNG:", num4RNG
    rngNamePath = os.path.join(clPath + "/rngene_name.npy")
    rngDataPath = os.path.join(clPath + "/rngene_data.npy")
    rngNames = np.load(rngNamePath)
    rngData = np.load(rngDataPath)

    flag = 0
    newRNGName = []
    for ind, rngName in enumerate(rngNames):
      if dic4RNG[rngName] == 7:
        if flag == 0:
          newRNGData = rngData[np.array([ind])]
          flag = 1
        else:
          newRNGData = np.vstack((newRNGData, rngData[np.array([ind])]))
        newRNGName.append(rngName)
    print "The shape of newRNGData and newRNGName:", newRNGData.shape, np.array(newRNGName).shape

    np.save(clPath + "/rngene_data_sampled.npy", newRNGData)
    np.save(clPath + "/rngene_name_sampled.npy", newRNGName)
