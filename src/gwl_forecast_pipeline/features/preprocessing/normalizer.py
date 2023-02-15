import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .mixins import NormalizerMixin
from ...constants import (TARGET_COL, FEATURES, GROUNDWATER_LEVEL_NORM_PER_WELL, )
from ...types import ModelConfig


class StaticNormalizer(NormalizerMixin):
    default_norm_features = [
        FEATURES.GROUNDWATER_RECHARGE,
        FEATURES.PERCOLATION,
        FEATURES.ELEVATION,
        FEATURES.SLOPE,
    ]

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.scaler = StandardScaler()

    def fit(self, X, y=None, partial=False):
        idx, cols, vals = X
        numeric_features = vals[:, self._get_cols_index(cols, self.default_norm_features)]
        self._fit_2d(numeric_features, self.default_norm_features,
                     scaler=self.scaler, partial=partial)
        return self

    def transform(self, X):
        idx, cols, vals = X
        vals[:, self._get_cols_index(cols, self.default_norm_features)] = \
            self._transform_2d(
                vals[:, self._get_cols_index(cols, self.default_norm_features)],
                self.default_norm_features, scaler=self.scaler
            )
        return idx, cols, vals


class TemporalRasterNormalizer(NormalizerMixin):
    default_norm_features = [
        FEATURES.HUMIDITY,
        FEATURES.TEMPERATURE,
        FEATURES.PRECIPITATION,
        FEATURES.LEAF_AREA_INDEX,
        FEATURES.GROUNDWATER_LEVEL,
    ]

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size
        self.group_scaler = {}
        self.glob_scaler = StandardScaler()

    def fit(self, X, y=None, partial=False):
        idx, cols, vals = X

        # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
        group_ids, group_indices_slices = self._get_group_indices(idx)
        for i, _slice in enumerate(group_indices_slices):
            self.group_scaler[group_ids[i]] = scaler = StandardScaler()
            self._fit_2d(
                vals[_slice, [cols.index(GROUNDWATER_LEVEL_NORM_PER_WELL)]],
                [GROUNDWATER_LEVEL_NORM_PER_WELL],
                scaler=scaler, partial=partial
            )

        self._fit_2d(
            vals[:, self._get_cols_index(cols, self.default_norm_features)],
            self.default_norm_features,
            scaler=self.glob_scaler, partial=partial
        )
        return self

    def transform(self, X):
        idx, cols, vals = X

        # scale per group
        gwl_scaled_locally = np.zeros((len(idx), 1, self.raster_size, self.raster_size))

        # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
        group_ids, group_indices_slices = self._get_group_indices(idx)
        for i, _slice in enumerate(group_indices_slices):
            try:
                gwl_scaled_locally[_slice] = self._transform_2d(
                    vals[_slice, [cols.index(GROUNDWATER_LEVEL_NORM_PER_WELL)]],
                    [GROUNDWATER_LEVEL_NORM_PER_WELL], scaler=self.group_scaler[group_ids[i]]
                )
            except KeyError:
                # if well was not in training set, then fit a new scaler
                self.group_scaler[group_ids[i]] = scaler = StandardScaler()
                self._fit_2d(
                    vals[_slice, [cols.index(GROUNDWATER_LEVEL_NORM_PER_WELL)]],
                    [GROUNDWATER_LEVEL_NORM_PER_WELL], scaler=scaler,
                )
                gwl_scaled_locally[_slice] = self._transform_2d(
                    vals[_slice, [cols.index(GROUNDWATER_LEVEL_NORM_PER_WELL)]],
                    [GROUNDWATER_LEVEL_NORM_PER_WELL], scaler=scaler,
                )

        # scale global
        vals[:, self._get_cols_index(cols, self.default_norm_features)] = self._transform_2d(
            vals[:, self._get_cols_index(cols, self.default_norm_features)],
            self.default_norm_features,
            scaler=self.glob_scaler,
        )

        return idx, cols, vals


class TargetNormalizer(NormalizerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.group_scaler = {}
        self.glob_scaler = StandardScaler()

    def fit(self, X, y=None, partial=False):
        idx, vals = X

        if self.conf.target_normalized:
            if self.conf.scale_per_group:
                # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
                group_ids, group_indices_slices = self._get_group_indices(idx)
                for i, _slice in enumerate(group_indices_slices):
                    self.group_scaler[group_ids[i]] = scaler = StandardScaler()
                    if partial:
                        scaler.partial_fit(pd.DataFrame(vals[_slice], columns=[self.conf.target_variable]))
                    else:
                        scaler.fit(pd.DataFrame(vals[_slice], columns=[self.conf.target_variable]))
            else:
                if partial:
                    self.glob_scaler.partial_fit(pd.DataFrame(vals, columns=[self.conf.target_variable]))
                else:
                    self.glob_scaler.fit(pd.DataFrame(vals, columns=[self.conf.target_variable]))

        return self

    def transform(self, X):
        idx, vals = X

        if self.conf.target_normalized:
            if self.conf.scale_per_group:
                # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
                group_ids, group_indices_slices = self._get_group_indices(idx)
                for i, _slice in enumerate(group_indices_slices):
                    try:
                        if self.conf.target_normalized and self.conf.scale_per_group:
                            vals[_slice] = self.group_scaler[group_ids[i]].transform(
                                pd.DataFrame(vals[_slice].copy(), columns=[self.conf.target_variable])
                            )
                    except KeyError:
                        # if well was not in training set, then fit a new scaler
                            self.group_scaler[group_ids[i]] = scaler = StandardScaler()
                            vals[_slice] = scaler.fit_transform(
                                pd.DataFrame(vals[_slice].copy(), columns=[self.conf.target_variable])
                            )
            else:
                # scale global
                vals = self.glob_scaler.transform(
                    pd.DataFrame(vals, columns=[TARGET_COL])
                )

        return vals

    def inverse_transform(self, index, values):
        if self.conf.target_normalized:
            if self.conf.scale_per_group:
                values = values.copy()
                for proj_id in index.unique():
                    locations = index.get_loc(proj_id)
                    vals = values[locations].copy()
                    if not isinstance(vals, np.ndarray):
                        vals = np.array([vals])
                    df = pd.DataFrame({TARGET_COL: vals})
                    inverse = self.group_scaler[proj_id].inverse_transform(df)
                    values[locations] = inverse
                return values
            else:
                df = pd.DataFrame({self.conf.target_variable: values})
                inverse = self.glob_scaler.inverse_transform(df)
                return inverse
        else:
            return values
