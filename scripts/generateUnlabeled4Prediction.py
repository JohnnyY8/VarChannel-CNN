import os
import numpy as np

cls = ["A375", "A549", "HA1E", "HCC515", "HEPG2", "PC3", "VCAP"]

basePath = "../files/newMap4UnlabeledData"

if __name__ == "__main__":

  for cl in cls:
    names4Prediction = []
    with open(os.path.join(basePath, cl + "_forpredict.txt")) as filePointer:
      fileLines = filePointer.readlines()
      drugNames = fileLines[0][: -2].split('\t')
      tpNames = [fileLine.split('\t')[0] for fileLine in fileLines[1: ]]
      newLabels = [fileLine[: -2].split('\t')[1: ] for fileLine in fileLines[1: ]]
    
    for rind, row in enumerate(newLabels):
      for cind, col in enumerate(row):
        if int(col) == 1:
          names4Prediction.append(tpNames[rind] + drugNames[cind])

    np.save(os.path.join(basePath, "unlabeledNamePairOneCol4Prediction_" + cl), names4Prediction)
