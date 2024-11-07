import PIL
import cv2
import torch
import torch.utils.data as data
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



class RafDataSet(data.Dataset):
    def __init__(self, root, transform=None):
        super(RafDataSet, self).__init__()
        self.root = root
        self.transform = transform
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0] - 1

    def __getitem__(self, item):
        img = cv2.imread(self.root + '\\' + self.path[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # path = self.root + '\\' + self.path[item]
        # img = Image.open(path).convert('RGB')
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        pil_img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(pil_img)

        label = torch.tensor(self.label[item]).type(torch.long)
        return img, label

    def __len__(self):
        return self.path.shape[0]



# aff数据集的dataset
# 标签标注
# 0:Neutral
# 1:Happy
# 2:Sad
# 3:Surprise
# 4:Fear
# 5:Disgust
# 6:Anger
# 7:Contempt
class AffectDataSet(data.Dataset):
    def __init__(self, root, transform=None):
        super(AffectDataSet, self).__init__()
        self.root = root
        self.transform = transform
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    def __getitem__(self, item):
        img = cv2.imread(self.root + '\\' + self.path[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # path = self.root + '\\' + self.path[item]
        # img = Image.open(path).convert('RGB')
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        pil_img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(pil_img)

        label = torch.tensor(self.label[item]).type(torch.long)
        return img, label

    def __len__(self):
        return self.path.shape[0]
