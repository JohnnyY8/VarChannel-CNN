#coding=utf-8

import os
import random
import numpy as np
import tensorflow as tf

from dataPro import *
from baseCNNModel import *
from modelTrainer import *
from generateEggs import *

flags = tf.app.flags

flags.DEFINE_string(
    "gpuId",
    "0",
    "Which gpu is assigned.")

flags.DEFINE_string(
    "fileRootPath",
    "./files",
    "File path for all files.")

flags.DEFINE_string(
    "dataRootPath",
    "./files",
    "Data file path for all data.")

flags.DEFINE_string(
    "path4SaveModel",
    "./files/trainedModel",
    "The path for saving model.")

flags.DEFINE_string(
    "path4SaveFinalValue",
    "./files/finalValue",
    "The path for saving final value of validation set.")

flags.DEFINE_string(
    "accuSavePath",
    "./files/accuAndLoss",
    "The path for npy files of training and valdation accuracy and loss value.")

flags.DEFINE_string(
    "path4Summaries",
    "./files/summaries",
    "The path for saving summaries.")

flags.DEFINE_string(
    "path4SaveEggsFile",
    "./files/eggFiles/",
    "The path for saving eggs file.")

flags.DEFINE_string(
    "savePath",
    "./files/1_fold/",
    "The path for saving loss and accuracy.")

flags.DEFINE_string(
    "ensembleDataPath",
    "./files/4training_ensemble/",
    "The path for enmsemble data.")

flags.DEFINE_string(
    "oneCLDataPath4Training",
    "./files/4_training/PC3",
    "The path for training in one cell line.")

flags.DEFINE_string(
    "oneCLDataPath4GenerateEggs",
    "./files/after_merge/PC3",
    "The path for generating eggs in one cell line.")

flags.DEFINE_float(
    "testSize",
    0.1,
    "The threshold for validation data.")

flags.DEFINE_float(
    "dropOutRate",
    0.5,
    "The threshold for validation data.")

flags.DEFINE_float(
    "learningRate",
    0.0001,
    "The learning rate for training.")

flags.DEFINE_float(
    "threshold4Val",
    0.5,
    "The threshold for validation data.")

flags.DEFINE_float(
    "threshold4Convegence",
    1e-40,
    "The threshold for training convegence.")

flags.DEFINE_integer(
    "num4Features",
    1956,
    "The number of original features.")

flags.DEFINE_integer(
    "num4FirstFC",
    200,
    "The number of neurons in first fully connected layer.")

flags.DEFINE_integer(
    "num4SecondFC",
    10,
    "The number of neurons in second fully connected layer.")

flags.DEFINE_integer(
    "conv1KWidth",
    1,
    "The width of convolutional kernel.")

flags.DEFINE_integer(
    "conv1SWidth",
    1,
    "The width of convolutional kernel stride.")

flags.DEFINE_integer(
    "num4InputChannels",
    7,
    "The number of input channels.")

flags.DEFINE_integer(
    "batchSize",
    146,
    "How many samples are trained in each iteration.")

flags.DEFINE_integer(
    "trainEpoches",
    1000,
    "How many times training through all train data.")

flags.DEFINE_integer(
    "nWeight",
    10,
    "The weighted for negative samples in objective funcion.")

flags.DEFINE_integer(
    "num4OutputChannels",
    2,
    "The number of output channels in first convolutional layer.")

FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuId
  insDataPro = DataPro(FLAGS)

  # For CNN
  insDataPro.loadEnsembleDataAndLabel()
  #print insDataPro.positiveData.shape, insDataPro.negativeData.shape, insDataPro.allTrainData.shape
  #print insDataPro.allTrainLabel.shape
  insCNNModel = BaseCNNModel(FLAGS, insDataPro)
  insCNNModel.buildBaseCNNModelGraph()
  insModelTrainer = ModelTrainer(FLAGS, insDataPro, insCNNModel)
  modelSavePath = insModelTrainer.trainCNN()
  insGenerateEggs = GenerateEggs(FLAGS, insDataPro, modelSavePath)
  insGenerateEggs.generateEggs2FilesIn3Col()
