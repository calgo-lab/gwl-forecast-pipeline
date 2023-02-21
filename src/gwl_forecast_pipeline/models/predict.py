import dataclasses
import multiprocessing
import os

import pandas as pd

from .build import load_model
from .batch_generator import create_batch_generator
from ..constants import FREQ
from ..types import (CNNModelConfig, ConvLSTMModelConfig, DataContainer, ModelConfig, )
from .. import config as config



def predict(model_conf: ModelConfig, data: DataContainer):
    with multiprocessing.Manager() as manager:
        model_conf_dict = manager.dict(dataclasses.asdict(model_conf))
        result_dict = manager.dict()
        data_dict = manager.dict(**dataclasses.asdict(data))

        p = multiprocessing.Process(target=_predict_thread, args=(
            data_dict, model_conf_dict, result_dict))
        p.start()
        p.join()
        predictions_df = result_dict['predictions'].copy()
    return predictions_df


def _predict_thread(
    data: dict,
    model_conf: dict,
    result: dict,
):
    import tensorflow as tf
    tf.keras.backend.clear_session()

    if model_conf['type_'] == 'cnn':
        model_config = CNNModelConfig(**model_conf)
    elif model_conf['type_'] == 'conv_lstm':
        model_config = ConvLSTMModelConfig(**model_conf)
    else:
        raise ValueError("unknown model type")

    model = load_model(
        os.path.join(config.MODEL_PATH, f'{model_config.name}.h5'),
        model_conf=model_config
    )
    data_container = DataContainer(**data)
    data_gen = create_batch_generator(data_container, model_config,
                                      shuffle=False, train=False)
    predictions = model.predict(data_gen)

    if model_config.group_loss:
        predictions = predictions[:, :-1]

    index_df = data_gen.x_temp_idx[['proj_id', 'time']].copy()
    y = data_container.target[
        data_gen.lead_indices[data_gen.x_temp_idx.index]
    ][:, :, 0]
    y = y.reshape(y.shape[:-1])
    y_df = pd.DataFrame(y)
    y_df.columns = [list(range(1, model_config.lead + 1)), ['y'] * model_config.lead]
    y_hat_df = pd.DataFrame(predictions)
    y_hat_df.columns = [list(range(1, model_config.lead+1)), ['y_hat']*model_config.lead]
    result_df = pd.concat([index_df, y_df, y_hat_df], axis=1)
    result_df.set_index(['proj_id', 'time'], inplace=True)
    result_df.columns = pd.MultiIndex.from_tuples(result_df.columns, names=['horizon', 'value'])
    result_df = result_df.stack(0)
    result_df.reset_index(inplace=True)
    result_df['time'] = result_df['time'] + (result_df['horizon'] * FREQ)
    result_df.set_index(['proj_id', 'time', 'horizon'], inplace=True)
    result['predictions'] = result_df
