import numpy as np
from .mixins import TransformerMixin, VStackTransformerMixin

from ...types import ModelConfig
from ...constants import (
    CATEGORICAL_STATIC_RASTER_FEATURES,
    GWL_SCALAR_FEATURES,
    GWL_RASTER_FEATURES,
    NUMERIC_STATIC_RASTER_FEATURES,
    TEMPORAL_RASTER_FEATURES,
)


class TemporalVStackTransformer(VStackTransformerMixin):
    raster_columns = ['gwl_asl_raster', 'gwl_raster_one_hot', 'temperature', 'humidity',
                      'precipitation', 'lai']
    scalar_columns = ['gwl_asl', 'gwl_delta']

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx = X.index
        raster_cols = X[self.raster_columns].columns.to_list()
        scalar_cols = X[self.scalar_columns].columns.to_list()
        raster_vals = self.df_to_np_stack(X[self.raster_columns], self.raster_size)
        scalar_vals = X[self.scalar_columns].values
        return idx, scalar_cols, raster_cols, scalar_vals, raster_vals


class TemporalHStackTransformer(TransformerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, scalar_cols, raster_cols, scalar_vals, raster_vals = X
        temporal_index = idx.to_frame().reset_index(drop=True)

        gwl_stack = raster_vals[
            :,
            self._get_cols_index(raster_cols, GWL_RASTER_FEATURES)
        ].astype(np.float32)

        feature_stack = raster_vals[
            :,
            self._get_cols_index(raster_cols, TEMPORAL_RASTER_FEATURES)
        ].astype(np.float32)

        targets = scalar_vals[
            :,
            self._get_cols_index(scalar_cols, GWL_SCALAR_FEATURES)
        ].astype(np.float32)

        return temporal_index, gwl_stack, feature_stack, targets


class StaticVStackTransformer(VStackTransformerMixin):
    columns = ['relief', 'aspect', 'slope', 'seepage', 'gw_recharge', 'lulc',
               'huek_lga', 'huek_lgc', 'huek_lha', 'huek_lkf']

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.columns]
        idx = X.index
        cols = X.columns.to_list()
        vals = self.df_to_np_stack(X, self.raster_size)
        return idx, cols, vals


class StaticHStackTransformer(TransformerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, cols, vals = X
        static_index = idx.to_frame().reset_index(drop=True)

        numeric_raster_features = vals[
            :,
            self._get_cols_index(cols, NUMERIC_STATIC_RASTER_FEATURES)
        ].astype(np.float32)

        categorical_raster_features = vals[
            :,
            self._get_cols_index(cols, CATEGORICAL_STATIC_RASTER_FEATURES)
        ].astype(np.int32)

        return static_index, numeric_raster_features, categorical_raster_features
