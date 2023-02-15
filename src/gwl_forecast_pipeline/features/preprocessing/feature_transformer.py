import numpy as np

from .mixins import FeatureTransformerMixin
from ...types import ModelConfig
from ...constants import GWL_ASL_OFFSET, FEATURES, GROUNDWATER_LEVEL_NORM_PER_WELL

EPSILON = 1.0001


class StaticFeatureTransformer(FeatureTransformerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, cols, vals = X

        # decode aspect
        aspect_sin, aspect_cos = self._decode_aspect(vals[:, cols.index(FEATURES.ASPECT)])
        aspect_sin = aspect_sin.reshape(-1, 1, *aspect_sin.shape[-2:])
        aspect_cos = aspect_cos.reshape(-1, 1, *aspect_cos.shape[-2:])
        vals = np.delete(vals, cols.index(FEATURES.ASPECT), axis=1)
        vals = np.concatenate([vals, aspect_sin, aspect_cos], axis=1)
        cols.remove(FEATURES.ASPECT)
        cols.extend(['aspect_sin', 'aspect_cos'])

        # log transformation
        vals[:, cols.index(FEATURES.ELEVATION)] = np.log(
            vals[:, cols.index(FEATURES.ELEVATION)] + GWL_ASL_OFFSET
        )

        # decode slope
        vals[:, cols.index(FEATURES.SLOPE)] = self._decode_slope(vals[:, cols.index(FEATURES.SLOPE)])

        # fill missing static data
        vals[:, cols.index(FEATURES.PERCOLATION)] = self.fill_raster(vals[:, cols.index(FEATURES.PERCOLATION)])
        vals[:, cols.index(FEATURES.GROUNDWATER_RECHARGE)] = self.fill_raster(vals[:, cols.index(FEATURES.GROUNDWATER_RECHARGE)])
        vals[:, cols.index(FEATURES.ELEVATION)] = self.fill_raster(vals[:, cols.index(FEATURES.ELEVATION)])
        return idx, cols, vals

    @staticmethod
    def _decode_aspect(aspect_col):
        array_rad = np.deg2rad(aspect_col * 360. / 255.)
        array_sin = np.sin(array_rad.copy())
        array_cos = np.cos(array_rad.copy())
        return array_sin, array_cos

    @staticmethod
    def _decode_slope(slope_col):
        return np.arccos(slope_col / 250.) * 180. / np.pi


class TemporalFeatureTransformer(FeatureTransformerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, cols, vals = X

        # fill missing gwl stack data
        vals[:, cols.index(FEATURES.GROUNDWATER_LEVEL)] = self.fill_raster(
            vals[:, cols.index(FEATURES.GROUNDWATER_LEVEL)], max_iter=0)

        # log transformation
        log_features = []
        if FEATURES.PRECIPITATION in cols:
            log_features.append(FEATURES.PRECIPITATION)
        if FEATURES.LEAF_AREA_INDEX in cols:
            log_features.append(FEATURES.LEAF_AREA_INDEX)
        if log_features:
            vals[:, self._get_cols_index(cols, log_features)] = np.log(
                vals[:, self._get_cols_index(cols, [FEATURES.PRECIPITATION, FEATURES.LEAF_AREA_INDEX])] + EPSILON
            )
        if FEATURES.GROUNDWATER_LEVEL in cols:
            gwl_vals = vals[:, [cols.index(FEATURES.GROUNDWATER_LEVEL)]].copy()
            vals = np.concatenate([vals, gwl_vals], axis=1)
            cols.append(GROUNDWATER_LEVEL_NORM_PER_WELL)
            vals[:, cols.index(FEATURES.GROUNDWATER_LEVEL)] = np.log(vals[:, cols.index(FEATURES.GROUNDWATER_LEVEL)] + GWL_ASL_OFFSET)
        return idx, cols, vals


class TargetTransformer(FeatureTransformerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx = X.index
        vals = X[[self.conf.target_variable]].values
        if not self.conf.scale_per_group:
            vals = np.log(vals + GWL_ASL_OFFSET)
        return idx, vals

    def inverse_transform(self, values):
        if not self.conf.scale_per_group:
            return np.exp(values) - GWL_ASL_OFFSET
        else:
            return values
