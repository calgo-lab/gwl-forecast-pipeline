import dataclasses
import logging
import multiprocessing
import os
import shutil
import time

from . import config as config
from .batch_generator import create_batch_generator
from .build import build_model, load_model
from ..types import (ModelConfig, ConvLSTMModelConfig, CNNModelConfig, DataContainer, )

logger = logging.getLogger(__name__)


def fit_new_model(train_data: DataContainer, model_conf: ModelConfig,
                  val_data: DataContainer = None):
    with multiprocessing.Manager() as manager:
        model_conf_dict = manager.dict(dataclasses.asdict(model_conf))
        result_dict = manager.dict()
        log_dict = manager.dict(dict(tensorboard=config.TENSORBOARD_PATH, csv_logger=config.CSV_LOGGER_PATH))
        data_dict = manager.dict()
        data_dict['train'] = manager.dict(dataclasses.asdict(train_data))
        if val_data:
            data_dict['validation'] = manager.dict(dataclasses.asdict(val_data))

        p = multiprocessing.Process(
            target=_fit_new_model_thread,
            args=(data_dict, model_conf_dict, result_dict),
            kwargs=dict(log=log_dict),
        )
        p.start()
        p.join()
        return dict(**result_dict)


def fit_model(model, train_data, model_conf: ModelConfig,
              val_data: DataContainer = None, tensorboard=None, csv_logger=None,
              tune_callback=False):
    import tensorflow as tf
    tf.keras.backend.clear_session()

    if isinstance(model, ModelConfig):
        model = load_model(
            os.path.join(config.MODEL_PATH, f'{model_conf.name}.h5'),
            model_conf=model_conf)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=model_conf.early_stop_patience,
            mode='min', restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODEL_PATH, f'{model_conf.name}.h5'),
        )
    ]
    if tune_callback:
        from ray.tune.integration.keras import TuneReportCallback
        callbacks.append(TuneReportCallback())
    if tensorboard:
        tb_path = os.path.join(tensorboard, model_conf.name)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        os.mkdir(tb_path)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=10, embeddings_freq=10))
    if csv_logger:
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(csv_logger, f'{model_conf.name}.csv')))
    t0 = time.time()
    logger.debug('start fitting model')
    history = model.fit(train_data, epochs=model_conf.epochs, validation_data=val_data,
        verbose=1, callbacks=callbacks)
    logger.debug(f'finished fitting model; time elapsed: {(time.time() - t0):.1f}s')
    return history


def _fit_new_model_thread(
        data: dict,
        model_conf: dict,
        result: dict,
        log: dict = None,
):
    try:
        if model_conf['type_'] == 'cnn':
            model_config = CNNModelConfig(**model_conf)
        elif model_conf['type_'] == 'conv_lstm':
            model_config = ConvLSTMModelConfig(**model_conf)
        else:
            raise ValueError("unknown model type")
        model = build_model(model_config)

        train_data = create_batch_generator(DataContainer(**(data['train'])), model_config)
        val_data = None
        if "validation" in data.keys():
            val_data = create_batch_generator(DataContainer(**data['validation']), model_config)

        history = fit_model(
            model=model,
            train_data=train_data,
            model_conf=model_config,
            val_data=val_data,
            tensorboard=log['tensorboard'] if log else None,
            csv_logger=log['csv_logger'] if log else None,
        )

        model.save(os.path.join(config.MODEL_PATH, f'{model_config.name}.h5'),
                   save_format="h5")
        result['history'] = history.history
    except Exception as e:
        logger.exception("error during training")
        raise e
