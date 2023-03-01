import logging
import os
from dataclasses import dataclass

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ... import config as config
from .pipeline import (
    StaticRasterPreprocessingPipeline,
    TemporalRasterPreprocessingPipeline,
    TargetPreprocessingPipeline,
)
from ...types import (ModelConfig, DataContainer)

logger = logging.getLogger(__name__)


@dataclass
class Preprocessor:
    model_conf: ModelConfig

    def __post_init__(self):
        self.static_preprocessor = StaticRasterPreprocessingPipeline(self.model_conf)
        self.temporal_preprocessor = TemporalRasterPreprocessingPipeline(self.model_conf)
        self.target_preprocessor = TargetPreprocessingPipeline(self.model_conf)

    @staticmethod
    def from_cache(model_conf: ModelConfig):
        preprocessor = Preprocessor(model_conf)
        preprocessor.static_preprocessor.restore_scalers(config.MODEL_PATH, name=model_conf.name)
        preprocessor.temporal_preprocessor.restore_scalers(config.MODEL_PATH, name=model_conf.name)
        preprocessor.target_preprocessor.restore_scalers(config.MODEL_PATH, name=model_conf.name)
        return preprocessor

    def store(self):
        self.static_preprocessor.dump_scalers(config.MODEL_PATH, name=self.model_conf.name)
        self.temporal_preprocessor.dump_scaler(config.MODEL_PATH, name=self.model_conf.name)
        self.target_preprocessor.dump_scaler(config.MODEL_PATH, name=self.model_conf.name)

    def fit(self, raw_data):
        meta_data, static_data_generator, temporal_data_generator = raw_data
        self.static_preprocessor.fit(next(static_data_generator))
        for temp_data in temporal_data_generator:
            self.temporal_preprocessor.fit(temp_data, partial=True)
            self.target_preprocessor.fit(temp_data, partial=True)

    def preprocess(self, raw_data, fit=True, use_fs_buffer=True, use_cached=True):
        meta_data, static_data_generator, temporal_data_generator = raw_data
        data_container = DataContainer.from_path(config.PREPROCESSOR_CACHE_PATH,
                                                 meta=meta_data)
        errors = data_container.collect(dry_run=True)

        if set(errors).intersection(set(data_container.static_fields)) or not use_cached:
            self._preprocess_static(
                static_data_generator, fit, data_container, use_fs_buffer,
            )
        else:
            logger.debug('use cached preprocessed static data')
        if set(errors).intersection(set(data_container.temporal_fields)) or not use_cached:
            self._preprocess_temp(
                temporal_data_generator, data_container, fit, use_fs_buffer,
            )
        else:
            logger.debug('use cached preprocessed temporal data')
        if not use_fs_buffer:
            data_container.collect()
            data_container.load()
        return data_container

    def inverse_transform_gwl(self, index, values: np.ndarray):
        return self.target_preprocessor.inverse_transform(index, values)

    def _preprocess_static(self, static_data_generator, fit, data_container, use_fs_buffer):
        static_data = next(static_data_generator)
        if fit:
            static_idx, static_numeric_features, static_categorical_features = self.static_preprocessor.fit_transform(static_data)
        else:
            static_idx, static_numeric_features, static_categorical_features = self.static_preprocessor.transform(static_data)

        if use_fs_buffer:
            static_idx.to_csv(data_container.static_index, index=False)

            data_container.numeric_static_raster.create()
            data_container.numeric_static_raster.write(static_numeric_features)

            data_container.categorical_static_raster.create()
            data_container.categorical_static_raster.write(static_categorical_features)
        else:
            data_container.static_index = static_idx
            data_container.numeric_static_raster = static_numeric_features
            data_container.categorical_static_raster = static_categorical_features

    def _preprocess_temp(self, temporal_data_generator, data_container: DataContainer, fit, use_fs_buffer,):
        if use_fs_buffer:
            pd.DataFrame(columns=['proj_id', 'time']).to_csv(data_container.temporal_index, index=False)
            data_container.gwl_raster.create()
            data_container.temporal_feature_raster.create()
            data_container.target.create()
        else:
            feature_raster_list = []
            gwl_raster_list = []
            target_list = []
            indices = []

        try:
            offset = 0
            for temp_data in temporal_data_generator:
                if fit:
                    transformed = self.temporal_preprocessor.fit_transform(temp_data, fit_partial=True)
                    targets = self.target_preprocessor.fit_transform(temp_data, fit_partial=True)
                else:
                    transformed = self.temporal_preprocessor.transform(temp_data)
                    targets = self.target_preprocessor.transform(temp_data)

                if not transformed[0].empty:
                    temp_idx, gwl_stack, feature_stack = transformed

                    if use_fs_buffer:
                        temp_idx.to_csv(data_container.temporal_index, index=False, header=False, mode="a")
                        data_container.gwl_raster.write(gwl_stack, offset=offset)
                        data_container.temporal_feature_raster.write(feature_stack, offset=offset)
                        data_container.target.write(targets, offset=offset)
                    else:
                        indices.append(temp_idx)
                        target_list.append(targets)
                        gwl_raster_list.append(gwl_stack)
                        feature_raster_list.append(feature_stack)
                offset += len(transformed[0])

            if not use_fs_buffer:
                data_container.temporal_index = pd.concat(indices)
                data_container.gwl_raster = np.concatenate(gwl_raster_list)
                data_container.temporal_feature_raster = np.concatenate(feature_raster_list)
                data_container.target = np.concatenate(target_list)
        except Exception as e:
            logger.exception("exception in temporal preprocessing")
            os.remove(data_container.temporal_index)
            os.remove(data_container.gwl_raster.file)
            os.remove(data_container.temporal_feature_raster.file)
            os.remove(data_container.target.file)
            raise e
