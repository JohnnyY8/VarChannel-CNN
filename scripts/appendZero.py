import os
import numpy as np

basePath = "/home/xlw/second/CNN_ensemble/4training_ensemble/"

positiveData = np.load(os.path.join(basePath, "positiveData.npy"))
negativeData = np.load(os.path.join(basePath, "negativeData.npy"))

def appendZero(data, flag):
  res = []
  for ind, ele in enumerate(data):
    tempData = np.array(data[np.array([ind])].tolist())
    if tempData.shape[2] != 7:
      print "Not 7...", tempData.shape
      num = 7 - tempData.shape[2]
      appendData = np.zeros((num, 1956)).reshape(-1, 1956, num)
      tempData = np.dstack((tempData, appendData))
      print "tempData.shape:", tempData.shape
    else:
      print "7......."
    res.append(tempData[0])
  res = np.array(res)
  print "res.shape", res.shape
  if flag == 0:
    np.save(os.path.join(basePath, "positiveDataAppendZeros.npy"), res)
  else:
    np.save(os.path.join(basePath, "negativeDataAppendZeros.npy"), res)
  
if __name__ == "__main__":
  appendZero(positiveData, 0)
  appendZero(negativeData, 1)
