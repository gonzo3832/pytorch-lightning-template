"""
データセットで以下を共通とすること
- pandasのDataFrame objectを読み込む
- column名はimage:画像名、y:教師データとする
引数は
必須：
df
xdir
phase
optional:
それ以下



"""
import os
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir, phase: str):
        assert phase in {'train', 'valid', 'test'}
        self.df = df
        self.datadir = datadir
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self):
        raise NotImplementedError


class ResizeDataset(Dataset):
    def __init__(self, df, datadir, phase,
                 height, width, convert_mode=None):
        """

        Args:
            df (pd.DataFrame): [description]
            datadir ([type]): [description]
            phase (str): [description]
            height ([type]): [description]
            width ([type]): [description]
            mode ([type], optional): [description]. Defaults to None.
        """
        super().__init__(df, datadir, phase)
        self.convert_mode = convert_mode
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert(self.convert_mode)),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx: int):
        image_file_name = self.df['x'].values[idx]
        x = Image.open(os.path.join(self.datadir, image_file_name))
        x = self.transform(x)

        y = self.df['y'].values[idx]

        return x.float(), torch.tensor(y).float()


class CropResizeDataset(ResizeDataset):

    def __getitem__(self, idx: int):

        image_file_name = self.df['x'].values[idx]
        x = Image.open(os.path.join(self.datadir, image_file_name))
        region = (self.df['x1'].values[idx], self.df['y1'].values[idx],
                  self.df['x2'].values[idx], self.df['y2'].values[idx])
        x = x.crop(region)
        x = self.transform(x)

        y = self.df['y'].values[idx]

        return x.float(), torch.tensor(y).float()


if __name__ == '__main__':
    data_dir = './inputs/MNIST_small/train'
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    ds = ResizeDataset(df, data_dir, 'train', 28, 28, 'L')
    for data, target in ds:
        print(data.size())
        break
