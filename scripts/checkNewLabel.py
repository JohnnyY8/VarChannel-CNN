import os
import numpy as np

cls = ["A375", "A549", "VCAP", "HCC515", "PC3", "HEPG2", "HA1E"]
basePath = "/home/xlw/Git-Repo/VarChannel-CNN/files/after_merge/"
path4Prediction = "/home/xlw/Git-Repo/VarChannel-CNN/files/newMap4UnlabeledData/"

for cl in cls:
  print cl, "......"
  clPath = os.path.join(basePath, cl)
  drugNames = np.load(os.path.join(clPath, "drug_name.npy"))
  tpNames = np.load(os.path.join(clPath, "tp_name.npy"))
  #newLabel = np.load(os.path.join(clPath, "new_label.npy"))

  with open(os.path.join(path4Prediction, cl + "_forpredict.txt")) as file_p:
    file_lines = file_p.readlines()
    newDrugNames = file_lines[0][: -2].split('\t')
    #print newDrugName
    newTpNames = [file_line.split('\t')[0] for file_line in file_lines[1: ]]
    #print newTpNames
    print "drugName.shape:", drugNames.shape
    print "tpName.shape:", tpNames.shape
    print len(newDrugNames)
    print len(newTpNames)

    print "DRUG..."
    count = 0
    for drugName in drugNames:
      if drugName in newDrugNames:
        count += 1
      else:
        print "Wrong..."
    print count

    print "TP..."
    count = 0
    for tpName in tpNames:
      if tpName in newTpNames:
        count += 1
      else:
        print "Wrong..."
    print count
