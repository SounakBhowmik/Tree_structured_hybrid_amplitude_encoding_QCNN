# -*- coding: utf-8 -*-


#%% Step 1: Set up the environment and necessary variables
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Variables for train and test dataset file paths
train_csv_path = '../Datasets/MNIST/mnist_train.csv'  # Replace with your training CSV path
test_csv_path = '../Datasets/MNIST/mnist_test.csv'    # Replace with your test CSV path

# Variable for new folder to save PCA results
pca_results_folder = '../Datasets/MNIST/pca_45'  # Replace with your new folder path

# Create the folder if it doesn't exist
if not os.path.exists(pca_results_folder):
    os.makedirs(pca_results_folder)
#%%
def standardize_dataframe(df):

  # Select only numeric columns
  numeric_cols = df.select_dtypes(include='number').columns.tolist()
  # Create a StandardScaler object
  scaler = StandardScaler()
  # Fit the scaler to the data and transform it
  df_scaled = scaler.fit_transform(df[numeric_cols])
  # Create a new DataFrame with the scaled data
  df_standardized = pd.DataFrame(df_scaled, columns=numeric_cols, index=df.index)
  # Combine the standardized numeric columns with the original non-numeric columns
  df_standardized = pd.concat([df_standardized, df.select_dtypes(exclude='number')], axis=1)
  return df_standardized




#%% Step:2 Perform PCA to reduce dimension to 45
# Load train dataset
train_df =pd.read_csv(train_csv_path)


# Separate features and labels
X_train = standardize_dataframe( train_df.drop(columns=['label']))  # Assuming 'label' column exists
y_train = train_df['label']

# Perform PCA to reduce to 45 dimensions
pca = PCA(n_components=45)
X_train_pca = pca.fit_transform(X_train.values)

# Save the PCA reduced data with labels
pca_train_data = pd.DataFrame(X_train_pca)
pca_train_data['label'] = y_train
pca_train_data.to_csv(os.path.join(pca_results_folder, 'train_pca.csv'), index=False)

# Plot the first two principal components
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 3], hue=y_train, palette='tab10', s=60)
plt.title('PCA: First Two Dimensions Colored by Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

#%% Step 3: Load the newly created dataset and split into train/validation sets

# Load the newly created PCA train dataset
pca_train_df = pd.read_csv(os.path.join(pca_results_folder, 'train_pca.csv'))

# Separate features and labels again
X_pca = pca_train_df.drop(columns=['label'])
y_pca = pca_train_df['label']

# Split into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)

# Output split sizes
print(f"Train set size: {X_train_split.shape}, Validation set size: {X_val_split.shape}")



#%% Step 4: Train a neural network model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from Model import QCNN


'''
# Define a simple feedforward neural network for binary classification #Testing
class BinaryClassifierNN(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output is 1 for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid for binary output
        return x
'''
# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    
    val_accuracies = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Evaluate model every epoch
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)
        print(f"Class {class_label} vs others test accuracy: {val_accuracy:.4f}")
        
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
    return val_accuracies

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    return accuracy_score(all_labels, all_preds)

# Function to prepare binary classification dataset
def prepare_binary_classification_dataset(X, y, target_class):
    # Select the data points where the label is equal to target_class (class 1)
    X_class_1 = X[y == target_class]
    y_class_1 = np.ones(X_class_1.shape[0])
    
    # Select data points from other classes (class 0)
    X_class_0 = X[y != target_class]
    y_class_0 = np.zeros(X_class_0.shape[0])
    
    # Randomly sample an equal number of class 0 examples using np.random.choice
    # We need to make sure that the number of samples from class 0 is equal to class 1
    idx_class_0 = np.random.choice(X_class_0.shape[0], size=len(X_class_1), replace=False)
    X_class_0 = X_class_0[idx_class_0]
    y_class_0 = y_class_0[idx_class_0]
    
    # Combine both classes into a single dataset
    X_combined = np.vstack([X_class_1, X_class_0])
    y_combined = np.hstack([y_class_1, y_class_0])
    
    return X_combined, y_combined

# Load the test dataset and apply PCA
test_df = pd.read_csv(test_csv_path)

X_test = standardize_dataframe(test_df.drop(columns=['label'])).values
y_test = test_df['label'].values

X_test_pca = pca.transform(X_test)



# Prepare the dataset and create DataLoader
def prepare_binary_classification_loader(X, y, target_class, batch_size=64):
    # Prepare binary dataset
    X_binary, y_binary = prepare_binary_classification_dataset(X, y, target_class)

    # Convert to torch tensors
    X_binary_tensor = torch.tensor(X_binary, dtype=torch.float32)
    y_binary_tensor = torch.tensor(y_binary, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(X_binary_tensor, y_binary_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

# Replace SVC with a neural network model for binary classification
input_size = 45  # Since PCA reduced the dataset to 45 dimensions
batch_size = 32
epochs = 50
learning_rate = 0.001



test_accuracies = []
val_scores = []
# Perform binary classification for each class
for class_label in range(10):
    # Prepare DataLoader for training data
    train_loader = prepare_binary_classification_loader(X_train_split.values, y_train_split.values, class_label)
    val_loader = prepare_binary_classification_loader(X_val_split.values, y_val_split.values, class_label)
    # Initialize the neural network model, loss function, and optimizer
    model = QCNN()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the neural network model
    val_scores.append(train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs))
    
    # Prepare DataLoader for test data
    test_loader = prepare_binary_classification_loader(X_test_pca, y_test, class_label)
    
    # Evaluate the model
    test_accuracy = evaluate_model(model, test_loader)
    test_accuracies.append(test_accuracy)
    print(f"Class {class_label} vs others test accuracy: {test_accuracy:.4f}")

# Calculate average test accuracy across all classes
average_accuracy = np.mean(test_accuracies)
print(f"Average test accuracy across all classes: {average_accuracy:.4f}")


#%% Saving results
import numpy as np
val_scores = np.array(val_scores)
test_accuracies = np.array(test_accuracies)

np.save('Results/val_scores.npy', val_scores, allow_pickle=True)
np.save('Results/test_accuracies.npy', test_accuracies, allow_pickle=True)

