import logging
import os
import time
import warnings
from typing import Union

import binpacking
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile

from ..constants import *
from ..types import MetaData
from .. import config as config


logger = logging.getLogger(__name__)


class FileSystemRaster:

    def __init__(self, raster: str):
        self.raster = raster

    def read(self, window: Window = None):
        with rasterio.open(self.raster) as ds:
            return ds.read(window=window)


class InMemoryRaster:

    def __init__(self, raster: MemoryFile):
        self.raster = raster

    def read(self, window: Window = None):
        with self.raster.open() as ds:
            return ds.read(window=window)


class DataLoader:

    def __init__(self, exclude_benchmark_gwl_raster=False, prefetch_sources=False):
        self.base_path = os.path.join(config.DATA_PATH, 'processed')
        self.exclude_benchmark_gwl_raster = exclude_benchmark_gwl_raster
        self._static_df = None
        self._gwl_df = None
        self._init_raster_data()
        if prefetch_sources:
            self.prefetch_sources()

    def _init_raster_data(self):
        self._raster_data = {}
        for feature in FEATURES:
            feature_name = feature.value
            feature_path = PROCESSED_DATA_PATHS[feature_name]
            if feature_name == FEATURES.WELL_META:
                continue
            if feature_name in [FEATURES.GROUNDWATER_LEVEL,
                                FEATURES.GROUNDWATER_LEVEL_ONE_HOT]:
                if self.exclude_benchmark_gwl_raster:
                    feature_name += BENCHMARK_SUFFIX
            self._raster_data[feature_name] = FileSystemRaster(os.path.join(self.base_path, feature_path, f'{feature_name}.tif'))

    def _prefetch_static_df(self):
        if self._static_df is None:
            self._static_df = pd.read_csv(
                os.path.join(self.base_path, 'well_meta', 'well_meta.csv'))
            self._static_df.set_index('proj_id', inplace=True)

    def _prefetch_gwl_df(self):
        if self._gwl_df is None:
            self._gwl_df = pd.read_feather(
                os.path.join(self.base_path, 'groundwater_level', 'gwl_asl.feather'))
            self._gwl_df.set_index(['proj_id', 'time'], inplace=True)
            self._gwl_df = self._gwl_df.sort_index()
            self._gwl_df = self._gwl_df[['gwl_asl']].astype(np.float32)
            self._gwl_df.loc[self._gwl_df.index[:1], :]

    def prefetch_sources(self):
        t0 = time.time()
        logger.debug('start prefetching data sources')
        self._prefetch_static_df()
        self._prefetch_gwl_df()
        for feature_name, raster in self._raster_data.items():
            if isinstance(raster, FileSystemRaster):
                with rasterio.open(raster.raster) as ds:
                    mem_file = MemoryFile()
                    with mem_file.open(**ds.meta) as mf:
                        mf.write(ds.read())
                self._raster_data[feature_name] = InMemoryRaster(mem_file)
        logger.debug(f'finished prefetching data sources, time elapsed: {(time.time() - t0):.2f}s')

    def load_data(self, well_ids, start, end, raster_size, max_chunk_size=3500):
        n_wells, static_data_generator = self._load_static_data(well_ids, raster_size)
        num_total_samples, temporal_data_generator = self._load_sequential_data(
            well_ids, start, end, raster_size, max_chunk_size
        )
        meta_data = MetaData(
            well_ids=well_ids,
            start=start,
            end=end,
            raster_size=raster_size,
            n_wells=n_wells,
            n_samples=num_total_samples,
        )
        return meta_data, static_data_generator, temporal_data_generator

    @property
    def static_df(self):
        if self._static_df is None:
            self._prefetch_static_df()
        return self._static_df

    @property
    def gwl_df(self):
        if self._gwl_df is None:
            self._prefetch_gwl_df()
        return self._gwl_df

    @staticmethod
    def _get_window(x, y, raster_size):
        return Window(x - raster_size // 2, y - raster_size // 2, raster_size, raster_size)

    def _load_static_data(self, well_ids, raster_size):
        n_wells = int(sum(self.static_df.index.isin(well_ids)))
        return n_wells, self._load_static_df(well_ids, raster_size)

    def _load_static_df(self, well_ids, raster_size):
        t0 = time.time()
        logger.debug("start loading static data")
        static_data = []
        for well_id in np.unique(well_ids):
            try:
                well_meta = self.static_df.loc[well_id]
            except KeyError:
                warnings.warn(f'could not find static data for well {well_id}')
                continue
            x, y = well_meta['x'], well_meta['y']
            window = self._get_window(x, y, raster_size)
            static_data.append({
                'proj_id': well_id,
                FEATURES.ELEVATION: self._raster_data[FEATURES.ELEVATION].read(window=window)[0],
                FEATURES.ASPECT: self._raster_data[FEATURES.ASPECT].read(window=window)[0],
                FEATURES.SLOPE: self._raster_data[FEATURES.SLOPE].read(window=window)[0],
                FEATURES.PERCOLATION: self._raster_data[FEATURES.PERCOLATION].read(window=window)[0],
                FEATURES.GROUNDWATER_RECHARGE: self._raster_data[FEATURES.GROUNDWATER_RECHARGE].read(window=window)[0],
                FEATURES.LAND_COVER: self._raster_data[FEATURES.LAND_COVER].read(window=window)[0],
                FEATURES.PERMEABILITY: self._raster_data[FEATURES.PERMEABILITY].read(window=window)[0],
                FEATURES.CHEMICAL_ROCK_TYPE: self._raster_data[FEATURES.CHEMICAL_ROCK_TYPE].read(window=window)[0],
                FEATURES.ROCK_TYPE: self._raster_data[FEATURES.ROCK_TYPE].read(window=window)[0],
                FEATURES.CAVITY_TYPE: self._raster_data[FEATURES.CAVITY_TYPE].read(window=window)[0],
            })
        static_df = pd.DataFrame(static_data).set_index('proj_id')
        logger.debug(f"finished loading static data, time elapsed: {(time.time() - t0):.2f}s")
        yield static_df

    def _load_seq_for_chunk(self, chunk, start, end, raster_size):
        index = pd.date_range(HYRAS_START, HYRAS_END, freq=FREQ, name="time")
        _dfs = []
        lais = []

        for proj_id, group in chunk.groupby(axis=0, level=0):
            x, y = self.static_df.loc[proj_id, 'x'], self.static_df.loc[proj_id, 'y']
            window = self._get_window(x, y, raster_size)
            _df = pd.DataFrame(index=pd.date_range(start, end, freq=FREQ, name="time"))
            _df['proj_id'] = proj_id

            for name in [FEATURES.HUMIDITY, FEATURES.TEMPERATURE, FEATURES.PRECIPITATION,
                         FEATURES.GROUNDWATER_LEVEL, FEATURES.GROUNDWATER_LEVEL_ONE_HOT]:
                raster_data = self._raster_data[name].read(window=window)
                _df[name] = pd.Series(list(raster_data),
                                      index=index[:len(raster_data)])[start:end]
            _dfs.append(_df.reset_index())
            lai = self._raster_data[FEATURES.LEAF_AREA_INDEX].read(window=window)
            lai = pd.Series(list(lai), index=pd.RangeIndex(1, 13, name='month'),
                            name=FEATURES.LEAF_AREA_INDEX).to_frame().reset_index()
            lai['proj_id'] = proj_id
            lais.append(lai)
        df = pd.concat(_dfs)
        lai_df = pd.concat(lais)

        chunk['month'] = chunk.index.get_level_values('time').copy().month
        chunk = (
            chunk
            .merge(df, on=['proj_id', 'time'], how='left')
            .merge(lai_df, on=['proj_id', 'month'], how='left')
        )
        return chunk.drop(columns=['month']).set_index(['proj_id', 'time'])

    def _load_sequential_data(self, well_ids, start, end, raster_size, max_chunk_size, use_gwl_raster_df=False):
        seq_df = self.gwl_df.loc[(slice(None), slice(start, end)), :]
        seq_df = seq_df[seq_df.index.get_level_values('proj_id').isin(well_ids)].copy().dropna()
        seq_df.rename(columns={FEATURES.GROUNDWATER_LEVEL: TARGET_COL}, inplace=True)
        return len(seq_df), self._load_chunk(seq_df, start, end, raster_size, max_chunk_size, use_gwl_raster_df)

    def _load_chunk(self, seq_df, start, end, raster_size, max_chunk_size, use_gwl_raster_df=False):
        samples_per_group = seq_df.groupby(axis=0, level=0).count().iloc[:, 0]
        if max_chunk_size < samples_per_group.max():
            raise ValueError("minimum chunk size has to be big enough to hold all samples of one well,"
                             f"max samples per well is: {samples_per_group.max()}")
        bins = binpacking.to_constant_volume(samples_per_group.to_dict(), max_chunk_size)
        logger.debug(f"load {len(seq_df)} samples in {len(bins)} chunks")

        for idx, bin_ in enumerate(bins):
            t0 = time.time()
            logger.debug(f"start loading {idx+1}. chunk of sequential data")
            chunk = seq_df[seq_df.index.get_level_values('proj_id').isin(list(bin_.keys()))]
            chunk = self._load_seq_for_chunk(chunk, start, end, raster_size)
            logger.debug(f"finished loading {idx+1}. chunk of sequential data, time elapsed: {(time.time() - t0):.2f}s")
            yield chunk
