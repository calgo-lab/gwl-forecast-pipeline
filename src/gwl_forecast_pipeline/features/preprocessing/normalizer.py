import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .mixins import NormalizerMixin
from ...constants import (
    NUMERIC_STATIC_RASTER_FEATURES,
    TEMPORAL_RASTER_FEATURES,
)
from ...types import ModelConfig


class StaticNormalizer(NormalizerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.scaler_static_2d = StandardScaler()

    def fit(self, X, y=None, partial=False):
        idx, cols, vals = X
        numeric_features = vals[:, self._get_cols_index(cols, NUMERIC_STATIC_RASTER_FEATURES[2:])]
        self._fit_2d(numeric_features, NUMERIC_STATIC_RASTER_FEATURES[2:],
                     scaler=self.scaler_static_2d, partial=partial)
        return self

    def transform(self, X):
        idx, cols, vals = X
        vals[:, self._get_cols_index(cols, NUMERIC_STATIC_RASTER_FEATURES[2:])] = \
            self._transform_2d(
                vals[:, self._get_cols_index(cols, NUMERIC_STATIC_RASTER_FEATURES[2:])],
                NUMERIC_STATIC_RASTER_FEATURES[2:], scaler=self.scaler_static_2d
            )
        return idx, cols, vals


class TemporalNormalizer(NormalizerMixin):

    def __init__(self, conf: ModelConfig):
        self.conf = conf
        self.raster_size = self.conf.raster_size
        self.group_scaler_1d = {}
        self.group_scaler_2d = {}
        self.glob_scaler_1d = StandardScaler()
        self.glob_scaler_2d = StandardScaler()

    def fit(self, X, y=None, partial=False):
        idx, scalar_cols, raster_cols, scalar_vals, raster_vals = X

        # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
        group_ids, group_indices_slices = self._get_group_indices(idx)
        for i, _slice in enumerate(group_indices_slices):
            self.group_scaler_1d[group_ids[i]] = scaler_1d = StandardScaler()
            if partial:
                scaler_1d.partial_fit(pd.DataFrame(scalar_vals[_slice], columns=scalar_cols))
            else:
                scaler_1d.fit(pd.DataFrame(scalar_vals[_slice], columns=scalar_cols))
            self.group_scaler_2d[group_ids[i]] = scaler_2d = StandardScaler()
            self._fit_2d(
                raster_vals[_slice, [raster_cols.index('gwl_asl_raster')]],
                ['gwl_asl_raster'],
                scaler=scaler_2d, partial=partial
            )

        if partial:
            self.glob_scaler_1d.partial_fit(pd.DataFrame(scalar_vals, columns=scalar_cols))
        else:
            self.glob_scaler_1d.fit(pd.DataFrame(scalar_vals, columns=scalar_cols))

        self._fit_2d(
            raster_vals[:, self._get_cols_index(raster_cols, ['gwl_asl_raster'] + TEMPORAL_RASTER_FEATURES)],
            ['gwl_asl_raster'] + TEMPORAL_RASTER_FEATURES,
            scaler=self.glob_scaler_2d, partial=partial
        )
        return self

    def transform(self, X):
        idx, scalar_cols, raster_cols, scalar_vals, raster_vals = X
        # scale per group
        gwl_norm_per_group = np.zeros((len(idx), 2))
        gwl_asl_raster_scaled_per_group = np.zeros((len(idx), 1, self.raster_size, self.raster_size))

        # custom faster alternative to pandas.groupby using the fact that all samples per well are consecutive
        group_ids, group_indices_slices = self._get_group_indices(idx)
        for i, _slice in enumerate(group_indices_slices):
            try:
                gwl_norm_per_group[_slice] = self.group_scaler_1d[group_ids[i]].transform(
                    pd.DataFrame(scalar_vals[_slice].copy(), columns=scalar_cols)
                )
                gwl_asl_raster_scaled_per_group[_slice] = self._transform_2d(
                    raster_vals[_slice, [raster_cols.index('gwl_asl_raster')]],
                    ['gwl_asl_raster'], scaler=self.group_scaler_2d[group_ids[i]]
                )
            except KeyError:
                # if well was not in training set, then fit a new scaler
                self.group_scaler_1d[group_ids[i]] = scaler_1d = StandardScaler()
                gwl_norm_per_group[_slice] = scaler_1d.fit_transform(
                    pd.DataFrame(scalar_vals[_slice].copy(), columns=scalar_cols)
                )
                self.group_scaler_2d[group_ids[i]] = scaler_2d = StandardScaler()
                self._fit_2d(
                    raster_vals[_slice, [raster_cols.index('gwl_asl_raster')]],
                    ['gwl_asl_raster'], scaler=scaler_2d,
                )
                gwl_asl_raster_scaled_per_group[_slice] = self._transform_2d(
                    raster_vals[_slice, [raster_cols.index('gwl_asl_raster')]],
                    ['gwl_asl_raster'], scaler=scaler_2d,
                )

        # scale global
        gwl_norm = self.glob_scaler_1d.transform(
            pd.DataFrame(scalar_vals, columns=scalar_cols)
        )
        raster_vals[
            :,
            self._get_cols_index(raster_cols, ['gwl_asl_raster'] + TEMPORAL_RASTER_FEATURES)
        ] = self._transform_2d(
            raster_vals[
                :,
                self._get_cols_index(raster_cols, ['gwl_asl_raster'] + TEMPORAL_RASTER_FEATURES)
            ],
            ['gwl_asl_raster'] + TEMPORAL_RASTER_FEATURES,
            scaler=self.glob_scaler_2d,
        )

        scalar_cols += ['gwl_asl_norm', 'gwl_delta_norm', 'gwl_asl_norm_per_group',
                        'gwl_delta_norm_per_group']
        scalar_vals = np.concatenate([scalar_vals, gwl_norm, gwl_norm_per_group], axis=1)
        raster_cols.append('gwl_asl_raster_scaled_per_group')
        raster_vals = np.concatenate([raster_vals, gwl_asl_raster_scaled_per_group],
                                     axis=1)

        return idx, scalar_cols, raster_cols, scalar_vals, raster_vals

    def inverse_transform_1d(self, index, values):
        if self.conf.scale_per_group:
            values = values.copy()
            for proj_id in index.unique():
                locations = index.get_loc(proj_id)
                vals = values[locations].copy()
                if not isinstance(vals, np.ndarray):
                    vals = np.array([vals])
                df = pd.DataFrame({self.conf.target_variable: vals})
                for col in {'gwl_asl', 'gwl_delta'}.difference({self.conf.target_variable, }):
                    df[col] = np.nan
                inverse = self.group_scaler_1d[proj_id].inverse_transform(df)[
                    :,
                    df.columns.get_loc(self.conf.target_variable)
                ]
                values[locations] = inverse
            return values
        else:
            df = pd.DataFrame({self.conf.target_variable: values})
            for col in {'gwl_asl', 'gwl_delta'}.difference({self.conf.target_variable, }):
                df[col] = np.nan
            inverse = self.glob_scaler_1d.inverse_transform(df)[
                :,
                df.columns.get_loc(self.conf.target_variable)
            ]
            return inverse

    @staticmethod
    def _get_group_indices(idx):
        proj_ids = idx.get_level_values('proj_id').to_series()
        bool_group_indices = (proj_ids != proj_ids.shift())
        group_ids = proj_ids[bool_group_indices].values
        group_indices = np.arange(len(idx))[bool_group_indices]
        group_indices_slices = [
            slice(
                group_indices[i],
                group_indices[i + 1] if i < len(group_indices) - 1 else None
            ) for i in range(len(group_indices))
        ]
        return group_ids, group_indices_slices
