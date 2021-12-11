from pytorch_lightning.loggers import MLFlowLogger
from src.models.pl_model_module import MyLightningModule_reg
from src.data_loaders.pl_data_module import MyDataModule
import pytorch_lightning as pl
import os
import hydra

from omegaconf import DictConfig
import pandas as pd
import warnings
import src.utils as utils
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
config_path = "./config"
config_name = "run_config.yaml"

@hydra.main(config_path=config_path, config_name=config_name)
def run(cfg: DictConfig) -> None:
    print(type(cfg))
    pl.seed_everything(cfg.globals.seed)
    cwd = hydra.utils.get_original_cwd()

    df = pd.read_csv(os.path.join(cwd, cfg.data_path.train_df_path))
    test_df = pd.read_csv(os.path.join(cwd, cfg.data_path.test_df_path))
    utils.path_fix(cfg.data_path)

    splitter = utils.get_splitter(cfg.splitter)
    for fold_i, (trn_idx, val_idx) in enumerate(splitter.split(df)):
        print(f'::::::: fold{fold_i} ::::::::')
        train_df = df.loc[trn_idx, :].reset_index(drop=True)
        valid_df = df.loc[val_idx, :].reset_index(drop=True)
        mlf_logger = MLFlowLogger(experiment_name=cfg.globals.ex_name,
                                  save_dir=os.path.join(cwd, 'mlruns'),
                                  run_name=f'fold{fold_i}-{cfg.globals.run_name}',
                                  )
        params = utils.get_log_parameters(cfg)
        mlf_logger.log_hyperparams(params)

        model = MyLightningModule_reg(cfg.model, cfg.loss, cfg.optimizer, cfg.scheduler)
        datamodule = MyDataModule(cfg.data_path, cfg.dataset,
                                  cfg.dataloader, train_df, valid_df, test_df)
        callback = utils.get_callback(cfg.callback)
        trainer = pl.Trainer(
            max_epochs=cfg.globals.max_epochs,
            logger=mlf_logger,
            log_every_n_steps=1,
            callbacks=[callback]
        )

        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    run()
