import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin as T


class TransformerMixin(BaseEstimator, T):

    @staticmethod
    def _get_cols_index(col_idx, features):
        return list(map(lambda col: col_idx.index(col), features))

    @staticmethod
    def _to_df(idx, cols, vals):
        """
        for debugging purpose
        """
        return pd.DataFrame(list(map(list, list(vals))), index=idx, columns=cols)


class FeatureTransformerMixin(TransformerMixin):

    @staticmethod
    def rolling_mean3d(a: np.ndarray, shape):
        result = a.copy()
        offset = (shape[1] - 1) // 2
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + (
        a.shape[2] - shape[2] + 1,) + shape
        strides = a.strides + a.strides
        strided = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)
        mean = np.nanmean(strided, axis=(3, 4, 5))
        result[:, offset:-offset, offset:-offset] = mean
        return result

    @staticmethod
    def shift_fill_raster(array, backwards=False):
        direction = -1 if backwards else 1
        mask = np.all(np.all(np.isnan(array), axis=1), axis=1)
        idx = np.where(~mask, np.arange(1, len(mask)+1), 0)
        np.maximum.accumulate(idx[::direction], axis=0, out=idx)
        update_mask = mask & idx.astype(bool)
        array[update_mask] = array[idx-1][update_mask]
        return array

    @classmethod
    def fill_raster(cls, array, max_iter=3):
        # pad array and iterate rolling mean 2d
        array = np.pad(array, [(0, 0), (1, 1), (1, 1)], mode='edge')
        original_values_mask = ~np.isnan(array)
        i = 0
        while np.any(np.isnan(array[:, 1:-1, 1:-1])):
            if i >= max_iter:
                break
            mean = cls.rolling_mean3d(array, (1, 3, 3))
            array[~original_values_mask] = mean[~original_values_mask]
            i += 1
        array = array[:, 1:-1, 1:-1]

        # use global 2d mean
        nan_mask = np.isnan(array)
        if np.any(nan_mask):

            mean = np.repeat(np.nanmean(array, axis=(1, 2)),
                             np.prod(array.shape[1:])).reshape(array.shape)
            array[nan_mask] = mean[nan_mask]

        # remaining nans cover a complete raster, therefore shift along time axis
        if np.any(np.isnan(array)):
            array = cls.shift_fill_raster(array, backwards=True)
        return array


class NormalizerMixin(TransformerMixin):

    @classmethod
    def _fit_2d(cls, vals, names, scaler, partial=False):
        flat = cls._flatten(vals)
        df = pd.DataFrame(flat, columns=names)
        if partial:
            scaler.partial_fit(df)
        else:
            scaler.fit(df)

    @classmethod
    def _transform_2d(cls, vals, names, scaler):
        original_shape = vals.shape
        flat = cls._flatten(vals)
        df = pd.DataFrame(flat, columns=names)
        transformed = scaler.transform(df)
        unflat = cls._unflatten(transformed, original_shape)
        return unflat

    @staticmethod
    def _unflatten(array, target_shape):
        return np.transpose(array.reshape(-1, *target_shape[1:][::-1]), (0, 3, 1, 2))

    @staticmethod
    def _flatten(array):
        return np.transpose(array, (0, 2, 3, 1)).reshape(-1, array.shape[1])

    @staticmethod
    def _get_group_indices(idx):
        proj_ids = idx.get_level_values('proj_id').to_series()
        bool_group_indices = (proj_ids != proj_ids.shift())
        group_ids = proj_ids[bool_group_indices].values
        group_indices = np.arange(len(idx))[bool_group_indices]
        group_indices_slices = [slice(group_indices[i],
            group_indices[i + 1] if i < len(group_indices) - 1 else None) for i in
            range(len(group_indices))]
        return group_ids, group_indices_slices


class VStackTransformerMixin(TransformerMixin):

    @staticmethod
    def df_to_np_stack(df, raster_size):
        return np.stack(np.vstack(df.values).flatten()).reshape(
            (len(df.index), len(df.columns), raster_size, raster_size)
        )
