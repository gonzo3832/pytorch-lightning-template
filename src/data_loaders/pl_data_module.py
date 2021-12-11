# -*- coding: utf-8 -*-
"""**Custom Pytorch_ligntning data module**
| *pytorch_lightning.LightningDataModule*\を定義するmodule。  
| instance化の際に、*dataのパス, dataset, dataloader*\ の設定値を格納したDictConfと、
| *train, valid, test のpandas.DataFrame*\を与えることで、**動的にmodule定義**\を行う。
   
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import sklearn.model_selection as sms
import src.data_loaders.ds_image as ds_image
import hydra


class MyDataModule(pl.LightningDataModule):

    def __init__(self, cfg_data_path, cfg_dataset, cfg_dataloader,train_df,valid_df,test_df):
        super().__init__()
        self.cfg_data_path = cfg_data_path
        self.cfg_dataset = cfg_dataset
        self.cfg_dataloader = cfg_dataloader
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
 
    def get_loader(self, phase):
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=True if phase == 'train' else False,
            num_workers=self.cfg_dataloader.num_workers,
        )
 
    def get_dataset(self, phase):
        assert phase in {'train', 'valid', 'test'}
        name_dataset = self.cfg_dataset.name
        kwargs_dataset = self.cfg_dataset.kwargs
        cls_ds = ds_image.__getattribute__(name_dataset)
        ds = cls_ds(
            datadir=self.get_datadir(phase),
            phase=phase,
            df=self.get_dataframe(phase),
            **kwargs_dataset
        )
        return ds
      
    def get_dataframe(self, phase):
        assert phase in {'train', 'valid', 'test'}
        if phase == 'train':
            return self.train_df
        elif phase == 'valid':
            return self.valid_df
        elif phase == 'test':
            return self.test_df

    def get_datadir(self, phase):
        assert phase in {'train', 'valid', 'test'}
        if phase == 'train' or phase == 'valid':
            return self.cfg_data_path.train_data_dir
        elif phase == 'test':
            return self.cfg_data_path.test_data_dir


    def train_dataloader(self):
        return self.get_loader(phase='train')

    def val_dataloader(self):
        return self.get_loader(phase='valid')

    def test_dataloader(self):
        return self.get_loader(phase='test')
