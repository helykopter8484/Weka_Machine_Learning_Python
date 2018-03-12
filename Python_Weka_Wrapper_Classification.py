
import timeit
import os
import csv
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import FilteredClassifier, Classifier, \
    SingleClassifierEnhancer, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.plot.classifiers import plot_roc


def J48(data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_J48_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_J48_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_J48_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_J48_ROC.png', wait=False)
    value_J48 = str(evl.percent_correct)
    return value_J48


def Bag_J48(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname = "weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"] )
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_J48_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_J48_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_j48_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_Bag_J48_ROC.png', wait=False)
    value_Bag_J48 = str(evaluation.percent_correct)
    return value_Bag_J48


def Boost_J48(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Boost_J48_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_J48_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_j48_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_Boost_J48_ROC.png', wait=False)
    value_Boost_J48 = str(evaluation.percent_correct)
    return value_Boost_J48


def J48graft (data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname="weka.classifiers.trees.J48graft", options=["-C", "0.25", "-M", "2"])
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"] )
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_J48graft_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_J48graft_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_J48graft_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_J48graft_ROC.png', wait=False)
    value_J48graft = str(evl.percent_correct)
    return value_J48graft


def Bag_J48graft(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.J48graft", options=["-C", "0.25", "-M", "2"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_J48graft_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_J48graft_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_j48graft_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_Bag_J48graft_ROC.png', wait=False)
    value_Bag_J48graft = str(evaluation.percent_correct)
    return value_Bag_J48graft


def Boost_J48graft(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.J48graft", options=["-C", "0.25", "-M", "2"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Boost_J48graft_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_J48graft_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_j48graft_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_Boost_J48graft_ROC.png', wait=False)
    value_Boost_J48graft = str(evaluation.percent_correct)
    return value_Boost_J48graft


def IBK(data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "1", "-W", "0"])
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_IBK_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_IBK_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_IBK_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_IBK_ROC.png', wait=False)
    value_IBK = str(evl.percent_correct)
    return value_IBK


def Bag_IBK(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname = "weka.classifiers.lazy.IBk", options = ["-K", "1", "-W", "0"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_IBK_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_IBK_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_IBK_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm + '_Bag_IBK_ROC.png', wait=False)
    value_Bag_IBK = str(evaluation.percent_correct)
    return value_Bag_IBK


def Boost_IBK(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "1", "-W", "0"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open (rnm + '_Boost_IBK_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_IBK_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_IBK_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Boost_IBK_ROC.png', wait=False)
    value_Boost_IBK = str(evaluation.percent_correct)
    return value_Boost_IBK


def NaiveBayes(data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname = "weka.classifiers.bayes.NaiveBayes")
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_NB_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_NB_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_NB_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_NB_ROC.png', wait=False)
    value_NB = str(evl.percent_correct)
    return value_NB


def Bag_NaiveBayes(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_NB_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_NB_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_NB_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Bag_NB_ROC.png', wait=False)
    value_Bag_NB = str(evaluation.percent_correct)
    return value_Bag_NB


def Boost_NB(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open (rnm + '_Boost_NB_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_NB_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_NB_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Boost_NB_ROC.png', wait=False)
    value_Boost_NB = str(evaluation.percent_correct)
    return value_Boost_NB


def RandomForest(data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I", "100", "-K", "0", "-S", "1"])
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_RF_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_RF_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_RF_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_RF_ROC.png', wait=False)
    value_RF = str(evl.percent_correct)
    return value_RF


def Bag_RandomForest(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I", "100", "-K", "0", "-S", "1"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname = "weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_RF_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_RF_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_RF_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Bag_RF_ROC.png', wait=False)
    value_Bag_RF = str(evaluation.percent_correct)
    return value_Bag_RF


def Boost_RandomForest(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I", "100", "-K", "0", "-S", "1"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Boost_RF_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_RF_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_RF_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Boost_RF_ROC.png', wait=False)
    value_Boost_RF = str(evaluation.percent_correct)
    return value_Boost_RF


def RandomTree(data, rnm):
    data.class_is_last()
    fc = FilteredClassifier()
    fc.classifier = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"])
    fc.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, folds, Random(1), pred_output)
    fc.build_classifier(data)
    f0 = open(rnm + '_RT_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc)
    f0.close()
    f1 = open(rnm + '_RT_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_RT_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evl.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evl.class_details())
    f2.close()
    plot_roc(evl, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_RT_ROC.png', wait=False)
    value_RT = str(evl.percent_correct)
    return value_RT


def Bag_RandomTree(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Bag_RT_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Bag_RT_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Bag_RT_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Bag_RT_ROC.png', wait=False)
    value_Bag_RT = str(evaluation.percent_correct)
    return value_Bag_RT


def Boost_RandomTree(data, rnm):
    data.class_is_last()
    fc1 = FilteredClassifier()
    fc1.classifier = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"])
    fc1.filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    fc2 = SingleClassifierEnhancer(classname="weka.classifiers.meta.AdaBoostM1", options=["-P", "100", "-S", "1", "-I", "10"])
    fc2.classifier = fc1
    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-p", "1"])
    folds = 10
    fc2.build_classifier(data)
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(fc2, data, folds, Random(1), pred_output)
    f0 = open(rnm + '_Boost_RT_Tree.txt', 'w')
    print >> f0, "Filename: ", rnm
    print >> f0, '\n\n'
    print >> f0, str(fc2)
    f0.close()
    f1 = open(rnm + '_Boost_RT_Prediction.txt', 'w')
    print >> f1, 'Filename:', rnm
    print >> f1, 'Prediction Summary:', (pred_output.buffer_content())
    f1.close()
    f2 = open(rnm + '_Boost_RT_Evaluation.txt', 'w')
    print >> f2, 'Filename:', rnm
    print >> f2, 'Evaluation Summary:', (evaluation.summary())
    print >> f2, '\n\n\n'
    print >> f2, (evaluation.class_details())
    f2.close()
    plot_roc(evaluation, class_index=[0,1], title=rnm, key_loc='best', outfile=rnm+'_Boost_RT_ROC.png', wait=False)
    value_Boost_RT = str(evaluation.percent_correct)
    return value_Boost_RT


def Get_CSV(directory):
    for root, dirs, files in os.walk(directory):
        CSV = []
        for filename in files:
            if filename.endswith(('.csv')):
                CSV.append(filename)
        return CSV


def Driver(directory):
    """This program uses the Python Weka Wrapper for Classification.
Details of Python Weka Wrapper: http://pythonhosted.org/python-weka-wrapper/
The Prediction Summary is resented in a txt file. Overall the code presents
the following output: Classification Tree, Prediction, Evaluation and the ROC
image. Input: The program takes all the csv files in a folder as an input and
performs classification. Output: Values of "Correctly Classified Instances"
presented as a csv file."""
    Data = []
    Head_str = ['Filename', 'J48', 'Boost_J48', 'Bag_J48',
                'J48_graft', 'Boost_J48graft', 'Bag_J48graft',
                'NaiveBayes', 'Boost_NaiveBayes', 'Bag_NaiveBayes',
                'RandomForest', 'Boost_RandomForest', 'Bag_RandomForest',
                'RandomTree', 'Boost_RandomTree', 'Bag_RandomTree',
                'IBK', 'Boost_IBK', 'Bag_IBK']
    Data.append(Head_str)
    csv_files = Get_CSV(directory)
    print "\n"
    print "="*60
    print "Performing Classification using Python Weka Wrapper:"
    print "http://pythonhosted.org/python-weka-wrapper/index.html#"
    print "="*60
    print "\n"
    print "Files:"

    for item in csv_files:
        print item
        fpath = directory + item
        rnm = item[0:-4]
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(fpath)

        j48 = J48(data, rnm)
        j48 = round(float(j48), 1)

        bag_j48 = Bag_J48(data, rnm)
        bag_j48 = round(float(bag_j48), 1)

        boost_j48 = Boost_J48(data, rnm)
        boost_j48 = round(float(boost_j48), 1)

        j48graft = J48graft(data, rnm)
        j48graft = round(float(j48graft), 1)

        bag_j48graft = Bag_J48graft(data, rnm)
        bag_j48graft = round(float(bag_j48graft), 1)

        boost_j48graft = Boost_J48graft(data, rnm)
        boost_j48graft = round(float(boost_j48graft), 1)

        ibk = IBK(data, rnm)
        ibk = round(float(ibk), 1)

        bag_ibk = Bag_IBK(data, rnm)
        bag_ibk = round(float(bag_ibk), 1)

        boost_ibk = Boost_IBK(data, rnm)
        boost_ibk = round(float(boost_ibk), 1)

        nb = NaiveBayes(data, rnm)
        nb = round(float(nb), 1)

        bag_nb = Bag_NaiveBayes(data, rnm)
        bag_nb = round(float(bag_nb), 1)

        boost_nb = Boost_NB(data, rnm)
        boost_nb = round(float(boost_nb), 1)

        rf = RandomForest(data, rnm)
        rf = round(float(rf), 1)

        bag_rf = Bag_RandomForest(data, rnm)
        bag_rf = round(float(bag_rf), 1)

        boost_rf = Boost_RandomForest(data, rnm)
        boost_rf = round(float(boost_rf), 1)

        rt = RandomTree(data, rnm)
        rt = round(float(rt), 1)

        bag_rt = Bag_RandomTree(data, rnm)
        bag_rt = round(float(bag_rt), 1)

        boost_rt = Boost_RandomTree(data, rnm)
        boost_rt = round(float(boost_rt), 1)

        res_list = list(
            (rnm, j48, boost_j48, bag_j48, j48graft,
            boost_j48graft, bag_j48graft, nb, boost_nb, bag_nb,
            rf, boost_rf, bag_rf, rt, boost_rt, bag_rt, ibk,
            boost_ibk, bag_ibk)
            )
        Data.append(res_list)
    return Data

if __name__ == '__main__':

    # Get the working directory.
    directory = os.getcwd() + '/'

    # Get the start time.
    start_time = timeit.default_timer()

    # Instantiate the JVM.
    jvm.start(packages=True, max_heap_size="4g")
    Data = Driver(directory)
    # Write results to csv file.
    resF = open("Classification_Results.csv", 'wb')
    wr = csv.writer(resF, dialect='excel')
    wr.writerows(Data)
    resF.close()
    print "Classification Completed!"

    # Stop the JVM.
    jvm.stop()

    # Record and display the time.
    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print 'Time taken for execution:', "%d:%02d:%02d" % (h, m, s)
