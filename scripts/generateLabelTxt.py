import os
import numpy as np

cls = ["A375", "A549", "VCAP", "HCC515", "PC3", "HEPG2", "HA1E"]
basePath = "/home/xlw/second/CNN_ensemble/after_merge"

for cl in cls:
  print cl, "......"
  clPath = os.path.join(basePath, cl)
  drugName = np.load(os.path.join(clPath, "drug_name.npy"))
  tpName = np.load(os.path.join(clPath, "tp_name.npy"))
  newLabel = np.load(os.path.join(clPath, "new_label.npy"))

  with open(os.path.join(clPath, "newLabel.txt"), 'w') as file_p:
    # For drug names
    flag = 0
    for iele in drugName:
      if flag == 0:
        strline = iele
        flag = 1
      else:
        strline += '\t' + iele
    strline += '\n'
    file_p.write(strline)
    print "drug names is done..."
    # For tp names and label
    for ind, iele in enumerate(tpName):
      strline = iele
      for jele in newLabel[ind]:
        strline += '\t' + str(jele)
      strline += '\n'
      file_p.write(strline)
      print "tp name is done..."
