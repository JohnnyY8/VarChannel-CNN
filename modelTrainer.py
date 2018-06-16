#coding=utf-8
import os
import math
import time
import random
import numpy as np
import tensorflow as tf

from dataSpliter import *
from resultStorer import *

class ModelTrainer:

  def __init__(self, FLAGS, insDataPro, insModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insModel = insModel
    self.insDataSpliter = DataSpliter(FLAGS, insDataPro)
    self.insResultStorer = ResultStorer(FLAGS)

  # Training and validation for DNN
  def trainDNN(self):
    self.xTrain, self.xTest, self.yTrain, self.yTest = self.insDataSpliter.splitData2TrainAndVal()

    # Set values of xTest and yTefrom YJ
    #self.xTest, self.yTest = 

    with tf.Session() as sess:
      oldTrainAccu, newTrainAccu, bestValAccu = 0.0, 0.0, 0.0
      flag = num4Epoches = 0
      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      sess.run(init)
      while True:
        trainIndex = np.array(range(self.xTrain.shape[0]))
        random.shuffle(trainIndex)
        print "No.%d epoch is starting..." % (num4Epoches)
        for ind in xrange(0, self.xTrain.shape[0], self.FLAGS.batchSize):
          batchXs, batchYs = self.xTrain[trainIndex[ind: ind + self.FLAGS.batchSize]], \
              self.yTrain[trainIndex[ind: ind + self.FLAGS.batchSize]]

          newTrainLoss, newTrainAccu, tempTS = sess.run(
              [self.insModel.loss,
               self.insModel.accuracy,
               self.insModel.trainStep],
              feed_dict = {
                  self.insModel.xData: batchXs,
                  self.insModel.yLabel: batchYs,
                  self.insModel.keepProb: self.FLAGS.dropOutRate})

          self.insResultStorer.addLoss(newTrainLoss)
          self.insResultStorer.addTrainAccu(newTrainAccu)
          print "  The loss is %.6f. The training accuracy is %.6f..." % (newTrainLoss, newTrainAccu)
          if flag == 0:
            flag = 1
          else:
            if abs(newTrainAccu - oldTrainAccu) <= self.FLAGS.threshold4Convegence:
              flag = 2
          oldTrainAccu = newTrainAccu

        newValAccu = sess.run(
            self.insModel.accuracy,
            feed_dict = {
                self.insModel.xData: self.xTest,
                self.insModel.yLabel: self.yTest,
                self.insModel.keepProb: 1.0})

        self.insResultStorer.addValAccu(newValAccu)
        print "    The validation accuracy is %.6f..." % (newValAccu)
        if newValAccu > bestValAccu:
          bestValAccu = newValAccu
          savePath = saver.save(sess, os.path.join(self.FLAGS.path4SaveModel, "model.ckpt"))
        if flag == 2 and num4Epoches >= self.FLAGS.trainEpoches:
          print "The training process is done..."
          print "The model saved in file:", savePath
          break
        num4Epoches += 1
    return savePath

  # Training and validation for CNN
  def trainCNN(self):
    self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataSpliter.splitData2TrainAndVal()

    self.insResultStorer.saveTrainSet(self.xTrain)
    self.insResultStorer.saveTrainLabel(self.yTrain)
    self.insResultStorer.saveValidationSet(self.xTest)
    self.insResultStorer.saveValidationLabel(self.yTest)

    with tf.Session() as sess:
      oldTrainAccu, newTrainAccu, bestValAccu = 0.0, 0.0, 0.0
      flag, num4Epoches = 0, 0

      path4TrainWriter = "train_convK" + str(self.FLAGS.conv1KWidth) + \
          "_convS" + str(self.FLAGS.conv1SWidth) + \
          "_numOC" + str(self.FLAGS.num4OutputChannels) + \
          "_numFFC" + str(self.FLAGS.num4FirstFC) + \
          "_numSFC" + str(self.FLAGS.num4SecondFC) + \
          "_nWeight" + str(self.FLAGS.nWeight)
      self.trainWriter = tf.summary.FileWriter(
          os.path.join(
              self.FLAGS.path4Summaries,
              path4TrainWriter),
          sess.graph)

      path4TestWriter = "test_convK" + str(self.FLAGS.conv1KWidth) + \
          "_convS" + str(self.FLAGS.conv1SWidth) + \
          "_numOC" + str(self.FLAGS.num4OutputChannels) + \
          "_numFFC" + str(self.FLAGS.num4FirstFC) + \
          "_numSFC" + str(self.FLAGS.num4SecondFC) + \
          "_nWeight" + str(self.FLAGS.nWeight)
      self.testWriter = tf.summary.FileWriter(
          os.path.join(
              self.FLAGS.path4Summaries,
              path4TestWriter))

      saver = tf.train.Saver()
      sess.run(self.insModel.init)

      while True:
        trainIndex = np.array(range(self.xTrain.shape[0]))
        random.shuffle(trainIndex)
        print("No.%d epoch is starting." % (num4Epoches))

        for ind in xrange(0, self.xTrain.shape[0], self.FLAGS.batchSize):
          batchXs, batchYs = \
              self.xTrain[trainIndex[ind: ind + self.FLAGS.batchSize]], \
              self.yTrain[trainIndex[ind: ind + self.FLAGS.batchSize]]

          ind4Summary = num4Epoches * math.ceil(
              self.xTrain.shape[0] * 1.0 / self.FLAGS.batchSize) + \
                  ind / self.FLAGS.batchSize

          if ind4Summary % 100 == 99:
            # Record training execution states
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            newTrainLoss, newTrainAccu, summary, tempTS = sess.run(
                 [self.insModel.loss,
                  self.insModel.accuracy,
                  self.insModel.merged,
                  self.insModel.trainStep],
                 feed_dict = {
                     self.insModel.xData: batchXs,
                     self.insModel.yLabel: batchYs,
                     self.insModel.keepProb: self.FLAGS.dropOutRate},
                 options = run_options,
                 run_metadata = run_metadata)

            self.trainWriter.add_run_metadata(
                 run_metadata,
                 "step%d" % ind4Summary)
            print("Adding run metadat for", ind4Summary)
            self.trainWriter.add_summary(summary, ind4Summary)

          else:
            # Record a training summary
            newTrainLoss, newTrainAccu, summary, tempTS = sess.run(
                [self.insModel.loss,
                 self.insModel.accuracy,
                 self.insModel.merged,
                 self.insModel.trainStep],
                feed_dict = {
                    self.insModel.xData: batchXs,
                    self.insModel.yLabel: batchYs,
                    self.insModel.keepProb: self.FLAGS.dropOutRate})
            self.trainWriter.add_summary(summary, ind4Summary)

            self.insResultStorer.addLoss(newTrainLoss)
            self.insResultStorer.addTrainAccu(newTrainAccu)
            print("  The loss is %.6f. The training accuracy is %.6f." % \
                (newTrainLoss, newTrainAccu))

          if flag == 0:
              flag = 1
          else:
            if abs(newTrainAccu - oldTrainAccu) <= \
                self.FLAGS.threshold4Convegence:
              flag = 2
          oldTrainAccu = newTrainAccu

        summary, preActOutput, hOutput, newValAccu = sess.run(
            [self.insModel.merged,
             self.insModel.preActOutput,
             self.insModel.hOutput,
             self.insModel.accuracy],
             feed_dict = {
                 self.insModel.xData: self.xTest,
                 self.insModel.yLabel: self.yTest,
                 self.insModel.keepProb: 1.0})
        self.testWriter.add_summary(summary, num4Epoches)
        self.insResultStorer.addValAccu(newValAccu)
        print("    The validation accuracy is %.6f." % (newValAccu))

        if newValAccu > bestValAccu:
          bestValAccu = newValAccu
          self.insResultStorer.savePreActOutput(preActOutput)
          self.insResultStorer.saveScore(hOutput)
          savePath = saver.save(
              sess,
              os.path.join(self.FLAGS.path4SaveModel, "model.ckpt"))

        if flag == 2 and num4Epoches >= self.FLAGS.trainEpoches:
          print("The training process is done.")
          print("The model saved in file:", savePath)
          break
        num4Epoches += 1

    self.trainWriter.flush()
    self.testWriter.flush()

    return savePath
