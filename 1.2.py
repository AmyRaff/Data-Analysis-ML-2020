import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
features_path = os.path.join(os.getcwd(),'bank_transaction_features.csv')
features = pd.read_csv(features_path, delimiter = ',')
labels_path = os.path.join(os.getcwd(), 'bank_transaction_labels.csv')
labels = pd.read_csv(labels_path, delimiter = ',')

# Add dataset to features file
features['dataset'] = labels['bank_transaction_dataset']

training = features[features["dataset"] == "TRAIN"]
validation = features[features["dataset"] == "VAL"]

# Visualise types
plt.plot(figsize=(30,8))
sns.countplot(x='bank_transaction_type', hue='dataset', data=features)
plt.tight_layout()
plt.savefig("types2.jpg")
plt.show()

# Visualise amounts
train_amounts = features[features['dataset'] == 'TRAIN']['bank_transaction_amount']
val_amounts = features[features['dataset'] == 'VAL']['bank_transaction_amount']

amounts = [train_amounts, val_amounts]
fig, ax = plt.subplots(figsize=(10,7))
bp = ax.boxplot(amounts) 
plt.xticks(np.arange(3), ('', 'TRAIN', 'VAL'))
plt.xlabel("Dataset")
plt.savefig("amounts2.png")
plt.show() 

# Visualise descriptions
train_desc = features[features['dataset'] == 'TRAIN']['bank_transaction_description']
val_desc = features[features['dataset'] == 'VAL']['bank_transaction_description']

top_5_train = train_desc.value_counts().head(5)
top_5_val = val_desc.value_counts().head(5)

unique_train = train_desc.unique().shape[0]
unique_val = val_desc.unique().shape[0]