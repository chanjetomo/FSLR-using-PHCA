import ripser
import numpy as np
import copy
import math

class PersistentHomologyClassifier:
    def __init__(self, dim=0):
        self.dim=dim
    
    def fit(self, x, y):
        self.byClass_data = byClassData(x, y)
        self.persistentDiagram =  getPersistentDiagram(self.byClass_data, self.dim)
   
    def predict(self, x_new):
        self.byClass_data_plus = copy.deepcopy(self.byClass_data)
        self.persistentDiagram_plus={}
        for clss in self.byClass_data.keys():
            self.byClass_data_plus[clss].append(x_new)
            self.persistentDiagram_plus[clss]=ripser.ripser(np.array(self.byClass_data_plus[clss]),maxdim=self.dim)['dgms']
        classPredicted=classSelector(self.persistentDiagram, self.persistentDiagram_plus, self.dim)
        return classPredicted
    

def byClassData(x_data,y_data):
    byClass_data={}
    for clss in set(y_data):
        byClass_data[clss]=[]
    for i in range(len(x_data)):
        byClass_data[y_data[i]].append(x_data[i])
    return byClass_data

def getPersistentDiagram(data,dim):
    dataPH={}
    dataClass=data.keys()
    for dClass in dataClass:
        dataPH[dClass]=ripser.ripser(np.array(data[dClass]),maxdim=dim)['dgms']
    return dataPH

def classSelector(PHCtrain, PHCpredict, dim):
    byClass_score={}
    n_train, n_predict = 0, 0
    for clss in PHCtrain.keys():
        #difference in the life span
        s2=0
        s2_train=0
        s2_predict=0
        
        for d in range(0,dim+1):
            for i in range(len(PHCtrain[clss][d])):
                if not math.isinf(PHCtrain[clss][d][i][1]):
                    s2_train+=PHCtrain[clss][d][i][1]-PHCtrain[clss][d][i][0]
                    n_train += 1
            for i in range(len(PHCpredict[clss][d])):
                if not math.isinf(PHCpredict[clss][d][i][1]):
                    s2_predict+=PHCpredict[clss][d][i][1]-PHCpredict[clss][d][i][0]
                    n_predict += 1
        s2=s2_predict-s2_train
        byClass_score[clss]=s2
    classPredicted = min(byClass_score, key=byClass_score.get)
    return classPredicted      
