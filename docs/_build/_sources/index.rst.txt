.. project documentation master file, created by
   sphinx-quickstart on Fri Dec 10 08:45:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
pytorch lightning template
===================================
このrepositoryについて
------------------------
pytorchを使用した機械学習の実験を行う過程で

- 「あれ、このclassってどうやって使うんだっけ...?」
- 「チームの方にも自分のmoduleを使って欲しいけど、１からコードを読んでもらうのは申し訳ない...」

| となることが多かった。

- チーム間で実験ノウハウの共有
- コードの可読性向上

| のため、pytorch lightningでコードのテンプレート化を行なった。

なぜpytorch lightning？
-----------------------
| pytorchは自由度の高い記述が可能な点が魅力だが、それ故にmain scriptが冗長になりがち。
| 結果として、コードを読み返して流用する作業に時間がかかる。
| pytorch lightningは、**pytorchの自由度を残しつつ、複雑な処理（主に学習ループ）をラップして提供する。**\
| これにより、**本質的な作業（model開発や実験）に費やす時間が増える**\ことを期待している。

project構成
-------------
| pytorch-lightning-template
| ┣ config・・・・・・・・・・・・・<設定ファイル>.yamlを置く場所
| ┃ ┗ run_config.yaml       
| ┣ inputs・・・・・・・・・・・・・読み込むデータを置く場所
| ┃ ┣ data1.jpg
| ┃ ┣ data2.jpg
| ┃ ┃ :
| ┃ ┣ train.csv
| ┃ ┗ test.csv
| ┣ src・・・・・・・・・・・・・・「Sorce」の略！
| ┃ ┣ models                
| ┃ ┃ ┣ :doc:`pl_model_module.py <src.models.pl_model_module>` ・・・pytorch_lightningのmodel moduleを記述
| ┃ ┃ ┗ :doc:`conv_net.py <src.models.conv_net>` ・・・・・・・pytorchの純model classを記述。モデル種類毎にmodule分ける。
| ┃ ┣ data_loaders          
| ┃ ┃ ┣ :doc:`pl_data_module.py <src.data_loaders.pl_data_module>` ・・・pytorch_lightningのdata moduleを記述
| ┃ ┃ ┗ :doc:`ds_image.py <src.data_loaders.ds_image>` ・・・・・・・pytorchの純dataset classを記述。
| ┃ ┣ loss_funcs            
| ┃ ┃ ┗ :doc:`loss.py <src.loss_funcs.loss>` ・・・・・・・・・loss関数を記述。現状パッケージとして管理する必要ない。要修正。
| ┃ ┗  :doc:`utils <src.utils>` ・・・・・・・・・その他必要な関数を記述。
| ┗ run.py・・・・・・・・・・・・実際に実行するのはこれ！

- 学習に必要な要素はsrc内に記述
- run.pyには学習フローのみを書く



使用方法
-------------
run.pyは以下のように記述する

.. code-block:: python
   
   # import文は省略.なぜかsyntax highlightされない

   config_path = "./config"
   config_name = "run_config.yaml"

   @hydra.main(config_path=config_path, config_name=config_name)
   def run(cfg: DictConfig) -> None:
      pl.seed_everything(cfg.globals.seed)
      cwd = hydra.utils.get_original_cwd()

      # ----- dfの分割（train,valid) --------
      df = pd.read_csv(os.path.join(cwd, cfg.data_path.train_df_path))
      test_df = pd.read_csv(os.path.join(cwd, cfg.data_path.test_df_path))
      utils.path_fix(cfg.data_path)
      splitter = utils.get_splitter(cfg.splitter)
      
      for fold_i, (trn_idx, val_idx) in enumerate(splitter.split(df)):
         print(f'::::::: fold{fold_i} ::::::::')
         train_df = df.loc[trn_idx, :].reset_index(drop=True)
         valid_df = df.loc[val_idx, :].reset_index(drop=True)

      # ---------mlflow loggerのインスタンス化  -----
         mlf_logger = MLFlowLogger(experiment_name=cfg.globals.ex_name,
                                    save_dir=os.path.join(cwd, 'mlruns'),
                                    run_name=f'fold{fold_i}-{cfg.globals.run_name}',
                                    )
         params = utils.get_log_parameters(cfg)
         mlf_logger.log_hyperparams(params)

         model = MyLightningModule_reg(cfg.model, cfg.loss, cfg.optimizer, cfg.scheduler)

         # -------- 分割したdfを使ってdata moduleのインスタンス化 ---------
         datamodule = MyDataModule(cfg.data_path, cfg.dataset,
                                     cfg.dataloader, train_df, valid_df, test_df)
         callback = utils.get_callback(cfg.callback)
         trainer = pl.Trainer(
            max_epochs=cfg.globals.max_epochs,
            logger=mlf_logger,
            log_every_n_steps=1,
            callbacks=[callback]
         )

         # -----学習と推論-----
         trainer.fit(model, datamodule=datamodule)
         trainer.test(model, datamodule=datamodule)


このprojectでは、
 - hydraによるハイパーパラメータ管理
 - mlflow trackingによる実験log管理

を行うことを前提としている。

hydraによるハイパーパラメータ管理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
| hydraとは、Facebook researchが開発した設定ファイル管理packageのこと。
| パラメータを.yamlで記述し、main関数をhydra.mainでラップすることで、
| パラメータをmain関数にDict型式で与えることが出来る。

config file 記述例

.. code-block:: yaml
   :caption: run_config.yaml

   globals:
      ex_name: hogehoge 


main scriptでの動作

.. code-block:: python
   :caption: run.py

   config_path = "./config"
   config_name = "run_config.yaml"

   @hydra.main(config_path=config_path, config_name=config_name)
   def run(cfg: DictConfig) -> None:
      print( cfg.globals.ex_name) # 通常のDictとは異なり、name space記法でアクセス可

output::

   >>> hogehoge

| hydraの強力な機能の一つに「Multi-run」がある。
| script実行時に-mオプションをつけ、config.yamlに記載した実験条件をkey=valueの形で複数渡すと、
| 自動的に全組み合わせで実験を行なってくれる。
| 例えば、modelA,modelBそれぞれについて、datasetA,datasetBを試したいときは、
| 以下の形で実行すると良い。


.. code-block:: bash 
   :caption: bash

   $ python run.py -m model.name=modelA,modelB dataset.name=datasetA,datasetB


mlflow trackingによる実験log管理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
| mlflowとは、MLOps管理ツールのこと。
| そのうちのmlflow trackingが実験logの管理に対応している。
| pytorch lightningには、mlflowを使用するためのクラスが用意されており、
| 以下の記述でmlflow loggerをインスタンス化して、trainer classに渡すことで、
| 学習途中のmetricの記録を行なってくれる。


.. code-block:: python
   :caption: run.py
   :emphasize-lines: 5,9

   from pytorch_lightning.loggers import MLFlowLogger
   # 中略
      @hydra.main(config_path=config_path, config_name=config_name)
      def run(cfg: DictConfig) -> None:
            mlf_logger = MLFlowLogger(experiment_name='hogehge',run_name='fugafuga')
            # 中略
            trainer = pl.Trainer(
               max_epochs=cfg.globals.max_epochs,
               logger=mlf_logger,
               log_every_n_steps=1,
               callbacks=[callback]
            )



| スクリプトを実行すると、mlruns ディレクトリが生成され、そこに実験結果が記録される。
| 注意点として、hydraを使う際は、カレントディレクトリが実行単位毎に変わってしまう。
| 任意のURIに実験結果を保存したい場合は、環境変数としてmlrunsを作成するURIを指定する必要がある。

.. code-block:: bash 
   :caption: bash

   $ export MLFLOW_TRACKING_URI=file:<カレントディレクトリへのpath> 

| またはスクリプト内で指定してもOK :doc:`waon <utils>`
| EC2等でmlflow tracking serverを立ててURIで指定すれば、webブラウザからアクセスして実験結果を確認することもできる。



Indices and tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   utils
   src.models
   src.data_loaders
   src.loss_funcs


注意事項&お願い
-------------------------
- pytorchちょっとわかる人が他人の猿真似で書いたコードです。至らぬところがたくさんあります🙇‍♂️
- 汎用性のあるテンプレートの作成を目指しています。「もっとこう書いた方がいいんじゃない？」という箇所があったら気軽にissueに投げてください！
- 説明不十分なところが多々あるので、気になる点があれば気軽に質問ください。
- 途中で力尽きてます、、全部にdocstring書けてません。


ToDo
-------
- KFoldもpl.lightning moduleで動的に切り替えられるようにする（したい）。

参考にしたサイト
---------------------
| `pytorchlighitningによるテンプレート管理を行おうと思ったキッカケ`_
| `sphinxでpython docstringからドキュメントを自動生成する`_
| `sphinxはじめの一歩`_
| `チームメイトのためにdocstringを書こう!!`_
| `pytorch lightning 基礎`_
| `mlflow+pytorh-lightning`_
| `sphinxによるdocstring生成`_
| `sphinx warningの解決`_
| `docstring google style 入門`_

.. _pytorchlighitningによるテンプレート管理を行おうと思ったキッカケ: https://tech.jxpress.net/entry/2021/11/17/112214
.. _sphinxでpython docstringからドキュメントを自動生成する: https://helve-blog.com/posts/python/sphinx-documentation/
.. _sphinxはじめの一歩: https://www.slideshare.net/k5yamate/sphinx-29486451
.. _チームメイトのためにdocstringを書こう!!: https://www.slideshare.net/cocodrips/docstring-pyconjp2019
.. _pytorch lightning 基礎: https://venoda.hatenablog.com/entry/2021/06/06/131004
.. _mlflow+pytorh-lightning: https://blog.chowagiken.co.jp/entry/2020/02/25/MLflow%E3%81%AE%E5%B0%8E%E5%85%A5%EF%BC%88%EF%BC%92%EF%BC%89%E5%AE%9F%E9%A8%93%E3%81%AE%E6%95%B4%E7%90%86%E3%81%A8%E6%AF%94%E8%BC%83%E3%82%92PyTorch%2Bpytorch-lightning%E3%81%A7%E3%82%84%E3%81%A3
.. _sphinxによるdocstring生成: https://hesma2.hatenablog.com/entry/2021/04/03/180206#%E3%83%88%E3%83%83%E3%83%97%E3%83%9A%E3%83%BC%E3%82%B8%E3%81%AE%E8%A8%AD%E5%AE%9A%E3%82%92%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81-indexrst-%E3%82%92%E7%B7%A8%E9%9B%86
.. _sphinx warningの解決: https://qiita.com/flcn-x/items/91604508b61c91b6163f
.. _docstring google style 入門: https://qiita.com/11ohina017/items/118b3b42b612e527dc1d