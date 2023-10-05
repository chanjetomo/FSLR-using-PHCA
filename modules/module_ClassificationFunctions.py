import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

def shuffle_per_class(FSL_dataset, data_per_class, classes):
    """ Shuffles data per class from a sorted dataset (acc to target) """
    FSLData = FSL_dataset['data']
    FSLTarget = FSL_dataset['target']

    curr_idx = 0
    for idx in range(classes):
        FSLData[curr_idx:data_per_class*(idx+1)], FSLTarget[curr_idx:data_per_class*(idx+1)] = shuffle(FSLData[curr_idx:data_per_class*(idx+1)], FSLTarget[curr_idx:data_per_class*(idx+1)], random_state=np.random.randint(classes))
        curr_idx += data_per_class*(idx+1)
    return FSLData, FSLTarget


def fivefoldDivideData(FSLData, FSLTarget, n_folds=5):
    """ Partitions the dataset for each class into five folds."""
    fivefoldList_features, fivefoldList_labels = [], []
    feat, lab = [], []
    data_per_fold = math.ceil(len(FSLTarget)/n_folds)
    cut = data_per_fold

    for i in range(len(FSLTarget)):
        if i < cut:
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
        else:
            fivefoldList_features.append(feat), fivefoldList_labels.append(lab)
            feat, lab = [], []
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
            cut += data_per_fold

    fivefoldList_features.append(feat), fivefoldList_labels.append(lab)
    return fivefoldList_features, fivefoldList_labels

def kfoldDivideData(FSLData, FSLTarget, data_per_class, folds=5):
    """ Partitions the dataset into five folds, each fold with equal number of data from the same class."""
    fivefoldData, fivefoldTarget = [[] for i in range(folds)], [[] for i in range(folds)]
    feat, lab = [], []
    fold = 0
    dpcpf = math.ceil(data_per_class/folds)
    cut = dpcpf

    for i in range(len(FSLTarget)):
        if i < cut:
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
        else:
            fivefoldData[fold] += feat
            fivefoldTarget[fold] += lab

            feat, lab = [], []
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
            fold += 1
            cut += dpcpf

        if fold == folds:
            fold = 0
    fivefoldData[folds-1] += feat
    fivefoldTarget[folds-1] += lab

    for f in range(folds):
        fivefoldData[f], fivefoldTarget[f] = shuffle(fivefoldData[f], fivefoldTarget[f], random_state=np.random.randint(folds))
    return fivefoldData, fivefoldTarget 

def get_specificity(confusionMatrix):
    label_lists = np.arange(len(confusionMatrix[0]))
    specificity = {}
    for label in label_lists:
        tp, tn, fp, fn =0, 0, 0, 0
        tp = confusionMatrix[label, label]
        fn = sum(confusionMatrix[label]) - tp
        for i in label_lists:
            for j in label_lists:
                if i == label or j == label:
                    continue
                else:
                    tn += confusionMatrix[i,j]
        for i in label_lists:
            if i==label:
                continue
            else:
                fp += confusionMatrix[label][i]
        specificity[str(label)] = tn/(tn+fp)
    return specificity


def get_classificationReport(method_labels):
    classificationReport = {}
    classificationReport['phca'] = classification_report(method_labels['true_labels'], method_labels['phca'], output_dict=True)
    cf = confusion_matrix(method_labels['true_labels'], method_labels['phca'])
    specificity = get_specificity(cf)

    avg = 0
    for label in specificity:
        avg += specificity[label]
        classificationReport['phca'][label]['specificity'] = specificity[label]
    
    classificationReport['phca']['macro avg']['specificity'] = avg/len(specificity)
    classificationReport['phca']['weighted avg']['specificity'] = avg/len(specificity)
    return classificationReport


def generate_graphs(classificationReport, classificationMeasurements, label_list, data_name, save=0):
    fig = plt.figure(1, 2, figsize=(8,5))
    for measurement in classificationMeasurements:
        x = 2*np.arange(len(label_list))
        

