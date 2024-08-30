import torch
import numpy as np

from torch.utils.data import Dataset


class UNSW_NB15Dataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14, transform=None):
        super(UNSW_NB15Dataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.transform = transform

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.reshape(self.train_data[index], (1, self.image_size, self.image_size))
        img = self.transform(torch.from_numpy(img)) if self.transform else torch.from_numpy(img)

        return index, img, label


class NSL_KDDDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14):
        super(NSL_KDDDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = image_size**2 - train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return index, img, label


class KDD_CUPDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14):
        super(KDD_CUPDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = image_size ** 2 - train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return index, img, label


class CIC_IDS2017Dataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14):
        super(CIC_IDS2017Dataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = image_size ** 2 - train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return index, img, label


class CIDDSDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14):
        super(CIDDSDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = image_size ** 2 - train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return index, img, label


class BoTIoTDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=14):
        super(BoTIoTDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = image_size ** 2 - train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return index, img, label



if __name__ == "__main__":
    import pandas as pd
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from util import data_preprocess_bot_iot

    base_csv_path = r'../my_project/datasets/IoT_Botnet.csv'
    df = pd.read_csv(base_csv_path)

    df = data_preprocess_bot_iot(df)
    df_Y = LabelEncoder().fit_transform(df.pop('category').values)
    df_X = df.values.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.25, random_state=666)
    print(Counter(y_train))
    print(Counter(y_test))

