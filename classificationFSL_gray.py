# ---------- Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# self-made modules
from modules.module_GenerateDataset import fslDataset as fsl
from modules.module_SumFrameDiff import sumFrameDiff as sfd
from modules.module_ClassificationFunctions import shuffle_per_class, kfoldDivideData, get_classificationReport
from modules.module_PHCA import PersistentHomologyClassifier

import warnings
warnings.filterwarnings("ignore")

# ------------ Initial Parameters
data_name = "Filipino Sign Language"
folds = 5
classes = 3
data_per_class = 20
zip_path = 'clips.zip'
classificationMeasurements = ['precision', 'f1-score', 'specificity', 'recall', 'support', 'accuracy']

# ------------ Dataset Collection and Saving
print('\n Collecting data from dataset ...')
dataset_paths = fsl.generate_dataset(fsl, zip_path, numclass=classes)
num_data = len(dataset_paths['data'])

# Performing SFD
FSL_dataset = {'data': [], 'target': dataset_paths['target']}
for n in range(num_data):
    vidpath = dataset_paths['data'][n]
    sumframe = sfd.SumFrameDiff(sfd, vidpath)
    FSL_dataset['data'].append(sumframe)

np.save(f"FSLdataset_{str(classes)}classes.npy", FSL_dataset)
print('Data collected and saved in an npy file.')

# ----------------- Loading Dataset from npy file
print('Loading dataset from npy file ....')
FSL_dataset = np.load(f"FSLdataset_{str(classes)}classes.npy", allow_pickle=True)
FSL_dataset = {'data': FSL_dataset.item().get('data'),
               'target': FSL_dataset.item().get('target')}

# --------------- Data Preprocessing
# data sorting acc to target
sorted_inds = np.array(FSL_dataset['target']).argsort()
FSL_dataset['data'], FSL_dataset['target'] = [FSL_dataset['data'][i] for i in sorted_inds], [FSL_dataset['target'][i] for i in sorted_inds]

# shuffling dataset per class
FSLData, FSLTarget = shuffle_per_class(FSL_dataset, data_per_class, classes)
print('\n Data collection finished. Features extracted.')

# five fold partitioning of data
fivefoldData, fivefoldTarget = kfoldDivideData(FSLData, FSLTarget, data_per_class)

# --------------- Five Fold Validation
method_labels = {'true_labels': [], 'phca': []}
print("Starting fivefold validation")
for val in range(folds):
    print(f"\nRunning validation {val}")

    x_train, y_train, x_test, y_test = [], [], [], []
    for j in range(folds):
        if j == val:
            x_test += fivefoldData[j]
            y_test += fivefoldTarget[j]
        else:
            x_train += fivefoldData[j]
            y_train += fivefoldTarget[j]
    
    print("\nThe PHCA model is learning from the data...")

    # Persistent Homology Classifier
    PHCmodel = PersistentHomologyClassifier()
    PHCmodel.fit(x_train, y_train)

    print("Model finished learning.")
    print("The model is now predicting new data.")

    method_labels['true_labels'] += y_test
    for y in range(len(y_test)):
        method_labels["phca"].append(PHCmodel.predict(x_test[y]))
    
    print("Model is finished predicting.")

# ---------------- Classification Report
print(classification_report(method_labels['true_labels'], method_labels['phca']))