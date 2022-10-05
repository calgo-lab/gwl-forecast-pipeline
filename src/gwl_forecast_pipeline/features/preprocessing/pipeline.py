import logging
import os
import time

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .feature_transformer import StaticFeatureTransformer, TemporalFeatureTransformer
from .normalizer import StaticNormalizer, TemporalNormalizer
from .stack_transformer import (
    StaticHStackTransformer,
    StaticVStackTransformer,
    TemporalVStackTransformer,
    TemporalHStackTransformer,
)
from ...types import ModelConfig

logger = logging.getLogger(__name__)


class StaticPreprocessingPipeline(BaseEstimator):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.normalizer = StaticNormalizer(conf)
        steps = [
            ('vstack_transformer',  StaticVStackTransformer(conf)),
            ('feature_transformer', StaticFeatureTransformer(conf)),
            ('normalizer',          self.normalizer),
            ('hstack_transformer',  StaticHStackTransformer(conf)),
        ]
        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None, use_cache=True, partial=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start fitting static preprocessing pipeline')
        self.pipeline.fit(X, normalizer__partial=partial)
        logger.debug(f'finished fitting static preprocessing pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return self

    def transform(self, X, use_cache=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start transforming with static preprocessing pipeline')
        transformed = self.pipeline.transform(X)
        logger.debug(f'finished transforming with static pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return transformed

    def fit_transform(self, X, y=None, fit_partial=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start fitting and transforming static with preprocessing pipeline')
        result = self.pipeline.fit_transform(X, normalizer__partial=fit_partial)
        logger.debug(f'finished fitting and transforming with static preprocessing pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return result

    def dump_scalers(self, path, name: str = None):
        _name = name + '_' if name else ""
        joblib.dump(self.normalizer.scaler_static_2d,
                    os.path.join(path, f'{_name}static_scaler_2d.gz'))

    def restore_scalers(self, path, name: str = None):
        _name = name + '_' if name else ""
        self.normalizer.scaler_static_2d = joblib.load(
            os.path.join(path, f'{_name}static_scaler_2d.gz'))


class TemporalPreprocessingPipeline(BaseEstimator):

    def __init__(self, conf: ModelConfig, exclude_benchmark=False):
        self.conf = conf
        self.normalizer = TemporalNormalizer(conf)
        self.feature_transformer = TemporalFeatureTransformer(conf, exclude_benchmark=exclude_benchmark)
        steps = [
            ('vstack_transformer',  TemporalVStackTransformer(conf)),
            ('feature_transformer', self.feature_transformer),
            ('normalizer',          self.normalizer),
            ('hstack_transformer',  TemporalHStackTransformer(conf)),
        ]
        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None, use_cache=True, partial=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start fitting preprocessing temporal pipeline')
        self.pipeline.fit(X, normalizer__partial=partial)
        logger.debug(f'finished fitting preprocessing temporal pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return self

    def transform(self, X, use_cache=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start transforming with temporal preprocessing pipeline')
        transformed = self.pipeline.transform(X)
        logger.debug(f'finished transforming with temporal pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return transformed

    def fit_transform(self, X, y=None, fit_partial=False, **kwargs):
        t0 = time.time()
        logger.debug(f'start fitting and transforming with temporal preprocessing pipeline')
        result = self.pipeline.fit_transform(X, normalizer__partial=fit_partial)
        logger.debug(f'finished fitting and transforming with temporal preprocessing pipeline; time elapsed: {(time.time() - t0):.1f}s')
        return result

    def inverse_transform(self, index, values):
        inverse = values
        if self.conf.target_normalized:
            inverse = self.normalizer.inverse_transform_1d(index, values)
        inverse = self.feature_transformer.inverse_transform(inverse)
        return inverse

    def dump_scaler(self, path, name: str = None):
        _name = name + '_' if name else ""
        joblib.dump(self.normalizer.group_scaler_1d, os.path.join(path, f'{_name}temporal_group_scaler_1d.gz'))
        joblib.dump(self.normalizer.group_scaler_2d, os.path.join(path, f'{_name}temporal_group_scaler_2d.gz'))
        joblib.dump(self.normalizer.glob_scaler_1d, os.path.join(path, f'{_name}temporal_glob_scaler_1d.gz'))
        joblib.dump(self.normalizer.glob_scaler_2d, os.path.join(path, f'{_name}temporal_glob_scaler_2d.gz'))

    def restore_scalers(self, path, name: str = None):
        _name = name + '_' if name else ""
        self.normalizer.group_scaler_1d = joblib.load(os.path.join(path, f'{_name}temporal_group_scaler_1d.gz'))
        self.normalizer.group_scaler_2d = joblib.load(os.path.join(path, f'{_name}temporal_group_scaler_2d.gz'))
        self.normalizer.glob_scaler_1d = joblib.load(os.path.join(path, f'{_name}temporal_glob_scaler_1d.gz'))
        self.normalizer.glob_scaler_2d = joblib.load(os.path.join(path, f'{_name}temporal_glob_scaler_2d.gz'))
