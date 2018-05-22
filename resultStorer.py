#coding=utf-8
import os
import numpy as np

class ResultStorer:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.loss = np.array([])
    self.trainAccu = np.array([])
    self.valAccu = np.array([])

  # Add loss
  def addLoss(self, loss):
    self.loss = np.append(self.loss, loss)

  # Save loss
  def saveLoss(self):
    np.save(os.path.join(self.FLAGS.baseSavePath, "loss.npy"), self.loss)

  # Add train accuracy
  def addTrainAccu(self, trainAccu):
    self.trainAccu = np.append(self.trainAccu, trainAccu)

  # Save train accuracy
  def saveTrainAccu(self):
    np.save(os.path,join(self.FLAGS.baseSavePath, "trainAccu.npy"), self.trainAccu)

  # Add validation accuracy
  def addValAccu(self, valAccu):
    self.valAccu = np.append(self.valAccu, valAccu)

  # Save validation accuracy
  def saveValAccu(self):
    np.save(os.path.join(self.FLAGS.baseSavePath, "valAccu.npy"), self.valAccu)

  # Save preActOutput value of validation set
  def savePreActOutput(self, preActOutput):
    np.save(os.path.join(self.FLAGS.path4SavePreActOutput, "preActOutput.npy"), preActOutput)

  # Save current validation set
  def saveValidationSet(self, validationSet):
    np.save(os.path.join(self.FLAGS.path4SavePreActOutput, "validationSet.npy"), validationSet)
