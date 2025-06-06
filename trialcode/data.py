import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_and_preprocess_data(path="adults.csv", machine_id=0, total_machines=4):
    # ----------------------------
    # Load the dataset
    # ----------------------------
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    try:
        df = pd.read_csv(path, index_col=False, skipinitialspace=True, header=None, names=header)
    except:
        url = "https://raw.githubusercontent.com/aliakbarbadri/mlp-classifier-adult-dataset/master/adults.csv"
        df = pd.read_csv(url, index_col=False, skipinitialspace=True, header=None, names=header)

    # ----------------------------
    # Clean missing values
    # ----------------------------
    df = df.replace('?', np.nan)
    df.dropna(inplace=True)

    # ----------------------------
    # Drop unnecessary column
    # ----------------------------
    df.drop('education-num', axis=1, inplace=True)

    # ----------------------------
    # Convert label to int (salary)
    # ----------------------------
    df['salary'] = df['salary'].map({'>50K': 1, '<=50K': 0}).astype(int)

    # ----------------------------
    # One-hot encode categorical features
    # ----------------------------
    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation', 'relationship', 'race', 'sex', 'native-country']
    df = pd.get_dummies(df, columns=categorical_columns)

    # ----------------------------
    # Normalize numerical features
    # ----------------------------
    normalize_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = preprocessing.StandardScaler()
    df[normalize_columns] = scaler.fit_transform(df[normalize_columns])

    # Split data based on machine ID
    total_samples = len(df)
    samples_per_machine = total_samples // total_machines
    start_idx = machine_id * samples_per_machine
    end_idx = start_idx + samples_per_machine if machine_id < total_machines - 1 else total_samples
    
    df = df.iloc[start_idx:end_idx]

    # ----------------------------
    # Split into train/test
    # ----------------------------
    X = df.drop('salary', axis=1).values.astype(np.float32)
    y = df['salary'].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def get_data_loaders(path="adults.csv", machine_id=0, total_machines=4, batch_size=32):
    X_train, y_train, X_test, y_test = load_and_preprocess_data(path, machine_id, total_machines)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

