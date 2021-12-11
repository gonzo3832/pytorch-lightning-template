# -*- coding: utf-8 -*-
"""**Custom Pytorch_ligntning model module**

| *pytorch_lightning.LightningModule*\を定義するmodule。  
| instance化の際に、*model, loss, optimizer, scheduler*\ の設定値を格納したDictConfを与えることで、**動的にmodule定義**\を行う。

Examples::

    @hydra.main(config_path=config_path, config_name=config_name)
    def run(cfg: DictConfig) -> None
        # 中略
        model = MyLightningModule_reg(cfg.model, cfg.loss, cfg.optimizer, cfg.scheduler)

ToDo:
    * 分類タスク用のLightning Moduleの作成
"""

import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import src.loss_funcs.loss as loss
import src.models.conv_net as models


class MyLightningModule_reg(pl.LightningModule):
    """回帰問題用のpl.LightningModule

    

    Attributes:
        cfg_model(DictConf):使用したいmodelのclass名とkwargsが格納されたDictConf
        cfg_loss(DictConf):使用したいloss関数のclass名とkwargsを格納.
        cfg_optim(DictConf):使用したいoptimizerのclass名とkwargsを格納.
        model(pytoch.module):モデルのインスタンス
        loss(pytorch.module):loss関数のインスタンス

    Note:
        - DictConf:OmegaConfにより生成される辞書型のオブジェクト。
        - とりあえずhydraによるパラメータはこの型式で渡される、ということだけ押さえておけばOK       
        
    """
    def __init__(self, cfg_model, cfg_loss, cfg_optim, cfg_scheduler):
        """
        Dictconf型式の設定値を受け取り、以下を行う。
        - モデルのインスタンス化
        - loss関数のインスタンス化
        Args:
            cfg_model ([type]): [description]
            cfg_loss ([type]): [description]
            cfg_optim ([type]): [description]
            cfg_scheduler ([type]): [description]
        
        """
        super().__init__()
        self.cfg_model = cfg_model
        self.model = self.get_model()
        self.cfg_loss = cfg_loss
        self.loss_func = self.get_loss_func()
        self.cfg_optim = cfg_optim
        self.cfg_scheduler = cfg_scheduler

    def forward(self, *args):
        """forward

        modelの順伝播処理。インスタンス化したmodelのforwardメソッドをcallしてるだけ。
        """
        return self.model(*args)
        

    def training_step(self, batch, batch_idx):
        """学習の最小単位
        
        | 内部でpl.LightningModuleのlogメソッドを呼んでいる。
        | これによりstep毎のloss、epoch毎のlossを記録する。

        ::

            self.log(
                'loss_train',loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True)            

        Args:
            batch (tuple): dataloaderが返すバッチデータ。(x,y)
            batch_idx (str): batch のindex番号。
        Return:
            output(dict): key:loss名、value:loss値
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        output = {'loss': loss}  
        self.log(
            'loss_train',loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True)
        return output

    def validation_step(self, batch, batch_idx):
        """検証の最小単位

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        output = {'loss_valid': loss}
        self.log_dict(
            output,
            on_epoch=True,
            on_step=False,
            prog_bar=True)
        return output

    def validation_end(self, outputs):
        """検証stepの最後に行う処理

        Args:
            outputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        loss_mean = 0
        y_list = []
        y_hat_list = []
        for output in outputs:
            loss_mean += output['loss_val']
            y_list.extend(output['y'])
            y_hat_list.extend(output['y_hat'])
        loss_mean /= len(outputs)
        results = {'log': {'loss_valid': loss_mean.item()}}
        return results
    
    def test_step(self, batch, batch_idx):
        """[summary]

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        output = {'loss_test': loss}
        self.log_dict(
            output,
            on_epoch=True,
            on_step=False,
            prog_bar=True)

        return output

    def configure_optimizers(self):
        """optimizer取ってくる

        Returns:
            [type]: [description]
        """
        name_optimizer = self.cfg_optim.name
        kwargs_optimizer = self.cfg_optim.kwargs
        cls_optimizer = optim.__getattribute__(name_optimizer)
        optimizer = cls_optimizer(self.model.parameters(), **kwargs_optimizer)

        if self.cfg_scheduler.name != None:
            name_scheduler = self.cfg_scheduler.name
            kwargs_scheduler = self.cfg_scheduler.kwargs
            cls_schesuler = optim.lr_scheduler.__getattribute__(name_scheduler)
            scheduler = cls_schesuler(optimizer, **kwargs_scheduler)

            return [optimizer], [scheduler]

        return optimizer

    def get_model(self):
        """hydraのformatのdictから、動的にmodelのinstance化を行う

        Returns:
            model: modelのinstance
        ToDo:
            moduleの動的呼び出し（cfgに書かせる）
        """
        name_model = self.cfg_model.name
        kwargs_model=self.cfg_model.kwargs
        cls_model = models.__getattribute__(name_model)
        if kwargs_model != None:
            return cls_model(**kwargs_model)
        return cls_model()

    def get_loss_func(self):
        """loss関数とってくる

        Returns:
            [type]: [description]
        """
        name_loss = self.cfg_loss.name

        if hasattr(nn, name_loss):
            cls_loss_func = nn.__getattribute__(name_loss)
            loss_func = cls_loss_func()
        else:
            cls_loss_func = loss.__getattribute__(name_loss)
            loss_func = cls_loss_func()

        return loss_func
