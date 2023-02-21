import os
import time
import logging
from typing import Dict

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from ..constants import *
from ..types import ModelConfig, DataContainer
from ..features.preprocessing.stack_transformer import StaticHStackTransformer
from .. import config as config


logger = logging.getLogger(__name__)


def weights_to_array(weights: Dict):
    return np.array(
        [weights[i] if i in weights.keys() else 0
         for i in range(min(weights.keys()), max(weights.keys()) + 1)]
    )


def get_lag_lead_indices(df, lag, lead):
    _df = df.copy()
    _lag_indices_list = [_df['temp_idx'].values.reshape(-1, 1).astype(float)]
    for _lag in range(1, lag):
        lag_idx = (_df['temp_idx'] - _lag).values.astype(float)
        lag_idx[((_df['proj_id'] != _df['proj_id'].shift(_lag)) | (
                    (_df['time'] - _lag * FREQ) != _df['time'].shift(
                _lag))).values] = np.nan
        _lag_indices_list.append(lag_idx.reshape(-1, 1))
    _lag_indices = np.concatenate(_lag_indices_list[::-1], axis=1)
    _lead_indices_list = []
    for _lead in range(1, lead+1):
        lead_idx = (_df['temp_idx'] + _lead).values.astype(float)
        lead_idx[((_df['proj_id'] != _df['proj_id'].shift(-_lead)) | (
                (_df['time'] + _lead * FREQ) != _df['time'].shift(
            -_lead))).values] = np.nan
        _lead_indices_list.append(lead_idx.reshape(-1, 1))
    _lead_indices = np.concatenate(_lead_indices_list, axis=1)
    mask = ~(np.any(np.isnan(_lag_indices), axis=1) | np.any(np.isnan(_lead_indices), axis=1))
    _df = _df[mask].reset_index(drop=True)
    return _df, _lag_indices[mask].astype(int), _lead_indices[mask].astype(int)


def get_feature_indices(enabled, available):
    return [i for i, feature in enumerate(available) if feature in enabled]


def create_batch_generator(data_container: DataContainer, conf: ModelConfig,
                           shuffle=True, train=True):
    import tensorflow as tf

    class DataGenerator(tf.keras.utils.Sequence):

        def __init__(self, data_container: DataContainer, conf: ModelConfig,
                     shuffle=True, train=True):
            logger.debug('initialize data generator')
            t0 = time.time()
            self.conf = conf
            self.lag = conf.lag
            self.lead = conf.lead
            self.batch_size = conf.batch_size
            self.shuffle = shuffle
            self.train = train

            if config.DATA_IN_MEMORY:
                data_container.load()
            else:
                data_container.collect()

            self.weight_array = None
            self.weight_feature_idx = None
            if self.conf.sample_weights and conf.weighted_feature:
                self.weight_feature_idx = StaticHStackTransformer.default_categoric_features.index(conf.weighted_feature)
                self.weight_array = weights_to_array(self.conf.sample_weights)


            self._x_static_idx: pd.DataFrame = data_container.static_index
            self._x_static_idx.index.rename('static_index', inplace=True)
            self._x_static_idx.reset_index(inplace=True)

            self._x_static_numeric: np.ndarray = data_container.numeric_static_raster
            self._x_static_categorical: np.ndarray = data_container.categorical_static_raster

            self.x_temp_idx: pd.DataFrame = data_container.temporal_index
            self.x_temp_idx.index.rename('temp_idx', inplace=True)
            self.x_temp_idx.reset_index(inplace=True)
            self.x_temp_idx = self.x_temp_idx.merge(
                self._x_static_idx, how='left', on='proj_id'
            )
            self.x_temp_idx, self.lag_indices, self.lead_indices = get_lag_lead_indices(self.x_temp_idx, self.lag, self.lead)
            if conf.group_loss:
                self.x_temp_idx['well_id'] = OrdinalEncoder().fit_transform(self.x_temp_idx['proj_id'].values.reshape(-1, 1))

            self._x_gwl_raster: np.ndarray = data_container.gwl_raster
            self._x_feature_raster: np.ndarray = data_container.temporal_feature_raster
            if self.train:
                self._y: np.ndarray = data_container.target
            self._indices = self.x_temp_idx.index.values.copy()
            self.on_epoch_end()
            logger.debug(f'done initialization of data generator; time elapsed: {(time.time() - t0):.2f}s')

        def __len__(self):
            # divide-ceil
            return -(-len(self.x_temp_idx) // self.batch_size)

        def __getitem__(self, idx):
            start = idx * self.batch_size
            stop = (idx + 1) * self.batch_size
            indices = self._indices[start:stop]

            batch_gwl_lag = self._x_gwl_raster[
                self.lag_indices[indices],
            ]
            batch_features_lag = self._x_feature_raster[
                self.lag_indices[indices],
            ]
            batch_features_lead = self._x_feature_raster[
                self.lead_indices[indices],
            ]
            batch_x_num_static = self._x_static_numeric[
                self.x_temp_idx.loc[indices, 'static_index']
            ]
            batch_x_cat_static = self._x_static_categorical[
                self.x_temp_idx.loc[indices, 'static_index']
            ]
            batch_x_cat_static = np.split(batch_x_cat_static, batch_x_cat_static.shape[1], axis=1)

            batch_x = batch_x_cat_static + [batch_x_num_static, batch_features_lag, batch_gwl_lag, batch_features_lead]
            if self.conf.group_loss:
                batch_x.append(
                    self.x_temp_idx.loc[indices, 'well_id'].values.reshape(-1, 1)
                )
            if self.train:
                batch_y = self._y[self.lead_indices[indices]].reshape(-1, self.lead)
                if self.conf.group_loss:
                    # ToDo: well id is not present
                    batch_y = np.concatenate([
                        batch_y,
                        self.x_temp_idx.loc[indices, 'well_id'].values.reshape(-1, 1)
                    ], axis=1)
                if self.weight_array and self.weight_feature_idx:
                    feature = batch_x_cat_static[self.weight_feature_idx]
                    mode = np.argmax([np.bincount(x, minlength=np.amax(feature) + 1) for x in
                                      feature.reshape(-1, self.conf.raster_size**2)], axis=1)
                    batch_y = np.concatenate([
                        batch_y,
                        self.weight_array[mode].reshape(-1, 1)
                    ], axis=1)
                return batch_x, batch_y
            else:
                return batch_x, None

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self._indices)

    return DataGenerator(data_container, conf, shuffle=shuffle, train=train)
