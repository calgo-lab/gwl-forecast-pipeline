import numpy as np
from .mixins import TransformerMixin, VStackTransformerMixin

from ...types import ModelConfig
from ...constants import (FEATURES, GWL_RASTER_FEATURES, )


class TemporalVStackTransformer(VStackTransformerMixin):
    features_default_selection = [
        FEATURES.HUMIDITY,
        FEATURES.TEMPERATURE,
        FEATURES.PRECIPITATION,
        FEATURES.LEAF_AREA_INDEX,
        FEATURES.GROUNDWATER_LEVEL,
        FEATURES.GROUNDWATER_LEVEL_ONE_HOT,
    ]

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx = X.index
        cols = self.features_default_selection.copy()
        vals = self.df_to_np_stack(X[self.features_default_selection], self.raster_size)
        return idx, cols, vals


class TemporalHStackTransformer(TransformerMixin):
    default_exo_features = [
        FEATURES.HUMIDITY,
        FEATURES.TEMPERATURE,
        FEATURES.PRECIPITATION,
        FEATURES.LEAF_AREA_INDEX,
    ]
    default_gwl_features = GWL_RASTER_FEATURES

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, cols, vals = X
        temporal_index = idx.to_frame().reset_index(drop=True)

        gwl_feature_stack = vals[:, self._get_cols_index(cols, self.default_gwl_features)].astype(np.float32)

        exo_feature_stack = vals[:, self._get_cols_index(cols, self.default_exo_features)].astype(np.float32)

        return temporal_index, gwl_feature_stack, exo_feature_stack


class StaticVStackTransformer(VStackTransformerMixin):
    features_default_selection = [
        FEATURES.LAND_COVER,
        FEATURES.ROCK_TYPE,
        FEATURES.CHEMICAL_ROCK_TYPE,
        FEATURES.CAVITY_TYPE,
        FEATURES.PERMEABILITY,
        FEATURES.GROUNDWATER_RECHARGE,
        FEATURES.PERCOLATION,
        FEATURES.ELEVATION,
        FEATURES.SLOPE,
        FEATURES.ASPECT,
    ]

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.features_default_selection]
        cols = self.features_default_selection.copy()
        idx = X.index
        vals = self.df_to_np_stack(X, self.raster_size)
        return idx, cols, vals


class StaticHStackTransformer(TransformerMixin):
    default_categoric_features = [
        FEATURES.LAND_COVER,
        FEATURES.ROCK_TYPE,
        FEATURES.CHEMICAL_ROCK_TYPE,
        FEATURES.CAVITY_TYPE,
        FEATURES.PERMEABILITY,
    ]
    default_numeric_features = [
        FEATURES.GROUNDWATER_RECHARGE,
        FEATURES.PERCOLATION,
        FEATURES.ELEVATION,
        FEATURES.SLOPE,
        'aspect_sin',
        'aspect_cos',
    ]

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, cols, vals = X
        static_index = idx.to_frame().reset_index(drop=True)

        numeric_features_stack = vals[:, self._get_cols_index(cols, self.default_numeric_features)].astype(np.float32)

        categoric_features_stack = vals[:, self._get_cols_index(cols, self.default_categoric_features)].astype(np.int32)

        return static_index, numeric_features_stack, categoric_features_stack
