import os
import numpy as np

cls = ["A375", "A549", "VCAP", "HCC515", "PC3", "HEPG2", "HA1E"]
basePath = "/home/xlw/second/CNN_ensemble/after_merge"

for cl in cls:
  clPath = os.path.join(basePath, cl)
  tpName = np.load(os.path.join(clPath, "tp_name.npy"))
  for ind, iele in enumerate(tpName[: -2]):
    if iele in tpName[ind + 1: ]:
      print "false"
      raw_input("....")
    else:
      print "true"
