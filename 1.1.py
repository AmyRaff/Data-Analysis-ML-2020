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

# Add category to features file
features['category'] = labels['bank_transaction_category']
categories = features['category'].unique()

# Visualising bank transaction types as countplots
plt.plot(figsize=(30,8))
sns.countplot(x='bank_transaction_type', hue='category', data=features)
plt.tight_layout()
plt.savefig("types.jpg")
plt.show()

# Visualiisng bank transaction amounts as boxplots
travel_amounts = features[features['category'] == 'TRAVEL']['bank_transaction_amount']
motor_amounts = features[features['category'] == 'MOTOR_EXPENSES']['bank_transaction_amount']
accom_meals_amounts = features[features['category'] == 'ACCOMMODATION_AND_MEALS']['bank_transaction_amount']
finance_amounts = features[features['category'] == 'BANK_OR_FINANCE_CHARGES']['bank_transaction_amount']
insurance_amounts = features[features['category'] == 'INSURANCE']['bank_transaction_amount']

amounts = [travel_amounts, motor_amounts, accom_meals_amounts, finance_amounts, insurance_amounts]
fig, ax = plt.subplots(figsize=(10,7))
bp = ax.boxplot(amounts) 
plt.xticks(np.arange(6), ('', 'TRAVEL', 'MOTOR', 'ACCOM_MEALS', 'BANK_FINANCE', 'INSURANCE'))
plt.xlabel("Category")
plt.savefig("amounts.png")
plt.show() 

# Visualising bank transaction descriptions as most 5 common descriptions and number of unique descriptions
travel_desc = features[features['category'] == 'TRAVEL']['bank_transaction_description']
motor_desc = features[features['category'] == 'MOTOR_EXPENSES']['bank_transaction_description']
accom_meals_desc = features[features['category'] == 'ACCOMMODATION_AND_MEALS']['bank_transaction_description']
finance_desc = features[features['category'] == 'BANK_OR_FINANCE_CHARGES']['bank_transaction_description']
insurance_desc = features[features['category'] == 'INSURANCE']['bank_transaction_description']

top_5_travel = travel_desc.value_counts().head(5)
top_5_motor = motor_desc.value_counts().head(5)
top_5_accom_meals = accom_meals_desc.value_counts().head(5)
top_5_finance = finance_desc.value_counts().head(5)
top_5_insurance = insurance_desc.value_counts().head(5)

unique_travel = travel_desc.unique().shape[0]
unique_motor = motor_desc.unique().shape[0]
unique_accom_meal = accom_meals_desc.unique().shape[0]
unique_finance = finance_desc.unique().shape[0]
unique_insurance = insurance_desc.unique().shape[0]

n_descriptions = features['bank_transaction_description'].unique().shape[0]