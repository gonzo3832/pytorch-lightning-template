import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import hydra
import sklearn.model_selection as sms


def get_log_parameters(cfg):
    """hydra cfgからロギングするパラメータのみを抜き足す

    Args:
        cfg : 

    Returns:
        dict: key = parameterの名前
    ToDo:
        lightningが使うデバイス名の取得
    """

    params = {
        "model": cfg.model.name,
        "dataset.": cfg.dataset.name,
        "dataset": {**cfg.dataset.kwargs},
        "loader.": cfg.dataloader.batch_size,
        "optimizer.": cfg.optimizer.name,
        "optimizer": {**cfg.optimizer.kwargs},
        "scheduler.": cfg.scheduler.name,
        "scheduler": {**cfg.scheduler.kwargs},
        "loss": cfg.loss.name,
        "seed": cfg.globals.seed,
        #        "device" :pl.LightningModule.device()
    }

    return params


def get_callback(cfg_callback):
    name_callback = cfg_callback.name
    kwargs_callback = cfg_callback.kwargs
    cls_callback = callbacks.__getattribute__(name_callback)
    return cls_callback(**kwargs_callback)


def path_fix(cfg_data_path):
    ori_path = hydra.utils.get_original_cwd()
    for k, v in cfg_data_path.items():
        cfg_data_path[k] = f'{ori_path}/{v}'

def get_splitter(cfg_splitter: dict):
    name_splitter= cfg_splitter.name 
    kwargs_splitter = cfg_splitter.kwargs
    return sms.__getattribute__(name_splitter)(**kwargs_splitter)
