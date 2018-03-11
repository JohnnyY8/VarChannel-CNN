import os
import numpy as np

cls = ["A375", "A549", "HA1E", "HCC515", "HEPG2", "PC3", "VCAP"]

basePath = "../files/newMap4UnlabeledData"
ensemblePath = "../files/4training_ensemble"

if __name__ == "__main__":
  res, resSorted = [], []
  positiveNamePairOneCol, unlabeledNamePairOneCol = \
      np.load(os.path.join(ensemblePath, "positiveNamePairOneCol.npy")), \
      np.load(os.path.join(ensemblePath, "unlabeledNamePairOneCol.npy"))

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
          namePair = tpNames[rind] + drugNames[cind]
          if namePair not in positiveNamePairOneCol:
            res.append([tpNames[rind], drugNames[cind]])

  for iele in unlabeledNamePairOneCol:
    for jele in res:
      namePair = jele[0] + jele[1]
      if namePair == iele:
        resSorted.append(jele)
        break

  np.save(os.path.join(ensemblePath, "unlabeledNamePairTwoCol.npy"), resSorted)
