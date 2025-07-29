import os
import pandas as pd

def load_data(path):
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))
    return train, test
