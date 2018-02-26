import os
import numpy as np

cls = ["A375", "A549", "VCAP", "HCC515", "PC3", "HEPG2", "HA1E"]
basePath = "/home/xlw/Git-Repo/VarChannel-CNN/files/after_merge/"
path4Prediction = "/home/xlw/Git-Repo/VarChannel-CNN/files/newMap4UnlabeledData/"

for cl in cls:
  print cl, "......"
  clPath = os.path.join(basePath, cl)
  drugName = np.load(os.path.join(clPath, "drug_name.npy"))
  tpName = np.load(os.path.join(clPath, "tp_name.npy"))
  #newLabel = np.load(os.path.join(clPath, "new_label.npy"))

  with open(os.path.join(path4Prediction, cl + "_forpredict.txt")) as file_p:
    file_lines = file_p.readlines()
    print file_lines
    raw_input("...")
    print "drugName.shape:", drugName.shape
    print "tpName.shape:", tpName.shape
    flag = 0
    for ind, file_line in enumerate(file_lines):
      if flag == 0:
        flag = 1
        continue
      file_line = file_line[: -1].split("\t")
      print "len(file_line):", len(file_line)
      if file_line[0] != tpName[ind - 1]:
        print "Error...name"
        break
      for jnd, ele in enumerate(file_line[1: ]):
        if float(ele) != newLabel[ind - 1][jnd]:
          print "Error..."
          break
