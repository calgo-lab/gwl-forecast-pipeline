import numpy as np

from .mixins import FeatureTransformerMixin
from ...types import ModelConfig
from ...constants import GWL_ASL_OFFSET

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
        aspect_sin, aspect_cos = self._decode_aspect(vals[:, cols.index('aspect')])
        aspect_sin = aspect_sin.reshape(-1, 1, *aspect_sin.shape[-2:])
        aspect_cos = aspect_cos.reshape(-1, 1, *aspect_cos.shape[-2:])
        vals = np.delete(vals, cols.index('aspect'), axis=1)
        cols.remove('aspect')
        cols += ['aspect_sin', 'aspect_cos']
        vals = np.concatenate([vals, aspect_sin, aspect_cos], axis=1)

        # log transformation
        vals[:, cols.index('relief')] = np.log(
            vals[:, cols.index('relief')] + GWL_ASL_OFFSET
        )

        # decode slope
        vals[:, cols.index('slope')] = self._decode_slope(vals[:, cols.index('slope')])

        # fill missing static data
        vals[:, cols.index('seepage')] = self.fill_raster(vals[:, cols.index('seepage')])
        vals[:, cols.index('gw_recharge')] = self.fill_raster(vals[:, cols.index('gw_recharge')])
        vals[:, cols.index('relief')] = self.fill_raster(vals[:, cols.index('relief')])
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

    def __init__(self, conf: ModelConfig, exclude_benchmark=False):
        self.conf = conf
        self.exclude_benchmark = exclude_benchmark

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        idx, scalar_cols, raster_cols, scalar_vals, raster_vals = X

        # fill missing gwl stack data
        raster_vals[:, raster_cols.index('gwl_asl_raster')] = self.fill_raster(
            raster_vals[:, raster_cols.index('gwl_asl_raster')], max_iter=0)

        if not self.exclude_benchmark:
            # set gwl raster center value to target
            raster_vals[
                :,
                raster_cols.index('gwl_asl_raster'),
                self.conf.raster_size//2,
                self.conf.raster_size//2
            ] = scalar_vals[:, scalar_cols.index('gwl_asl')].copy()

        # log transformation
        raster_vals[:, self._get_cols_index(raster_cols, ['precipitation', 'lai'])] = np.log(
            raster_vals[:, self._get_cols_index(raster_cols, ['precipitation', 'lai'])] + EPSILON
        )
        raster_vals[:, raster_cols.index('gwl_asl_raster')] = np.log(
            raster_vals[:, raster_cols.index('gwl_asl_raster')] + GWL_ASL_OFFSET
        )
        scalar_vals[:, scalar_cols.index('gwl_asl')] = np.log(
            scalar_vals[:, scalar_cols.index('gwl_asl')] + GWL_ASL_OFFSET
        )

        # clip gwl delta
        scalar_vals[:, scalar_cols.index('gwl_delta')] = np.clip(
            scalar_vals[:, scalar_cols.index('gwl_delta')], -1., 1.
        )

        return idx, scalar_cols, raster_cols, scalar_vals, raster_vals

    @staticmethod
    def inverse_transform(values):
        return np.exp(values) - GWL_ASL_OFFSET
