import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Import Data
features_path = os.path.join(os.getcwd(),'bank_transaction_features.csv')
features = pd.read_csv(features_path, delimiter = ',')
labels_path = os.path.join(os.getcwd(), 'bank_transaction_labels.csv')
labels = pd.read_csv(labels_path, delimiter = ',')

# Add necessary columns from labels to features
features['category'] = labels['bank_transaction_category']
features['dataset'] = labels['bank_transaction_dataset']

# -------------------------------- Data Preprocessing
# single out useful columns
relevant_data = features[['bank_transaction_description', 'category', 'dataset']].copy()
# convert descriptions to lowercase
relevant_data['bank_transaction_description'] = relevant_data['bank_transaction_description'].str.lower()
# get rid of entries with descriptions that only occur once
data_filtered = relevant_data.groupby('bank_transaction_description').filter(lambda x: len(x) > 1)
# use one hot encoding to convert categorical data to numerical
data_filtered = pd.get_dummies(data_filtered, columns=["bank_transaction_description"])

# -------------------------------- Split into Testing and Validation Sets
training = data_filtered[data_filtered["dataset"] == "TRAIN"]
validation = data_filtered[data_filtered["dataset"] == "VAL"]

X_train = training.drop(['dataset','category'],axis=1)
y_train = pd.DataFrame(training['category'])

X_val = validation.drop(['dataset','category'],axis=1)
y_val = pd.DataFrame(validation['category'])

# -------------------------------- Using the Model
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
print('Classification accuracy on training set: {:.3f}'.format(lr.score(X_train, y_train)))
print('Classification accuracy on validation set: {:.3f}'.format(lr.score(X_val, y_val)))

# -------------------------------- Confusion Matrices

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

prediction_train = lr.predict(X_train)

train_cm = confusion_matrix(y_train, prediction_train)
train_cm_norm = train_cm/train_cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15,15))
plot_confusion_matrix(train_cm_norm, classes=lr.classes_, title='Training confusion')
plt.savefig("testconf.png")
plt.show()

prediction_val = lr.predict(X_val)

val_cm = confusion_matrix(y_val, prediction_val)
val_cm_norm = val_cm/val_cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(val_cm_norm, classes=lr.classes_, title='Validation confusion')
plt.savefig("valconf.png")
plt.show()
