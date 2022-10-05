import logging
import os
import time
import warnings

import binpacking
import numpy as np
import pandas as pd
import rasterio

from ..constants import FREQ
from ..types import RuntimeConfig, MetaData

logger = logging.getLogger(__name__)

HYRAS_START = pd.Timestamp(1951, 1, 7)
HYRAS_END = pd.Timestamp(2016, 1, 3)


class DataLoader:
    rasters_src = {
        # temporal
        'humidity': os.path.join('hyras', 'humidity', 'raster_new', 'humidity.tif'),
        'temperature': os.path.join('hyras', 'air_temperature_mean', 'raster_new', 'air_temperature_mean.tif'),
        'precipitation': os.path.join('hyras', 'precipitation', 'raster_new', 'precipitation.tif'),
        'gwl_asl': os.path.join('groundwater_levels', 'raster', 'gwl_asl_interpolated.tif'),
        'gwl_one_hot': os.path.join('groundwater_levels', 'raster', 'gwl_one_hot.tif'),
        'lai': os.path.join('leaf_area_index', 'lai.tif'),
        # static
        'relief': os.path.join('eudem', 'base_cropped.tif'),
        'aspect': os.path.join('eudem', 'aspect_cropped.tif'),
        'slope': os.path.join('eudem', 'slope_cropped.tif'),
        'lulc': os.path.join('landcover', 'lulc_cat.tif'),
        'huek_lga': os.path.join('huek250', 'huek_LGA.tif'),
        'huek_lgc': os.path.join('huek250', 'huek_LGC.tif'),
        'huek_lha': os.path.join('huek250', 'huek_LHA.tif'),
        'huek_lkf': os.path.join('huek250', 'huek_LKF.tif'),
        'seepage': os.path.join('seepage', 'seepagerate_masked.tif'),
        'groundwater_recharge': os.path.join('groundwater_recharge', 'gwrecharge_masked.tif'),
    }

    def __init__(self, runtime_conf: RuntimeConfig, exclude_benchmark_gwl_raster=False):
        self.base_path = runtime_conf.data_dir
        self.cache_path = runtime_conf.data_loader_cache_dir
        self.exclude_benchmark_gwl_raster = exclude_benchmark_gwl_raster
        if self.exclude_benchmark_gwl_raster:
            self.rasters_src['gwl_raster'] = os.path.join('groundwater_levels', 'raster', 'gwl_asl_interpolated_benchmark.tif')
            self.rasters_src['gwl_one_hot'] = os.path.join('groundwater_levels', 'raster', 'gwl_one_hot_benchmark.tif')
        self._static_df = None
        self._gwl_df = None
        self.max_in_memory_raster = runtime_conf.max_in_memory_raster_size_byte
        self.runtime_conf = runtime_conf

    def prefetch_sources(self):
        t0 = time.time()
        logger.debug('start prefetching data sources')
        self._static_df = pd.read_csv(os.path.join(self.base_path, 'well_meta', 'well_meta_inner.csv'))
        self._static_df.set_index('proj_id', inplace=True)
        self._gwl_df = pd.read_feather(os.path.join(self.base_path, 'groundwater_levels', 'groundwater_levels_clean_interpolated_2006-.feather'))
        self._gwl_df.set_index(['proj_id', 'datum'], inplace=True)
        self._gwl_df = self._gwl_df.sort_index()
        self._gwl_df = self._gwl_df[['gwl_asl', 'gwl_delta']].astype(np.float32)
        self._gwl_df.loc[self._gwl_df.index[:1], :]
        # Todo extra memmap for benchmark raster
        for name, path in self.rasters_src.items():
            with rasterio.open(os.path.join(self.base_path, path)) as ds:
                if os.path.getsize(os.path.join(self.base_path, path)) > self.max_in_memory_raster:
                    memmap_path = os.path.join(self.cache_path, f'{name}.npy')
                    if not os.path.exists(memmap_path):
                        fp = np.memmap(memmap_path, dtype=np.float32, shape=(ds.count, *ds.shape), mode="w+")
                        fp[:] = ds.read().astype(np.float32)
                        fp.flush()
                    setattr(self, f'_{name}_array', np.memmap(memmap_path, dtype=np.float32, shape=(ds.count, *ds.shape), mode="r"))
                else:
                    setattr(self, f'_{name}_array', ds.read().astype(np.float32))
                setattr(self, f'_{name}_meta', {
                    'bounds': ds.bounds,
                    'resolution': ds.res,
                    'transform': ds.transform,
                })
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

    # def load_random_sample(self, start, end, raster_size, sample_size, random_state=42):
    #     sample = self._gwl_df[
    #         self._gwl_df.index.get_level_values('proj_id').isin(
    #             self.static_df.index.get_level_values('proj_id').unique()
    #         )
    #     ].loc[(slice(None), slice(start, end)), :].sample(sample_size, random_state=random_state)
    #     sample = sample.dropna()
    #     sample_proj_ids = sample.index.get_level_values('proj_id').unique().tolist()
    #     static_df = self._load_static_data(sample_proj_ids, raster_size)
    #     seq_df = self._load_seq_for_chunk(sample, start, end, raster_size)
    #     return static_df, seq_df, len(sample)

    @property
    def static_df(self):
        if self._static_df is None:
            self.prefetch_sources()
        return self._static_df

    @property
    def gwl_df(self):
        if self._gwl_df is None:
            self.prefetch_sources()
        return self._gwl_df

    @staticmethod
    def _get_mask_from_memory_array(x, y, size_x_km, size_y_km, array, meta):
        resolution_x, resolution_y = max(abs(meta['resolution'][0]), 1000.), max(abs(meta['resolution'][1]), 1000.)
        steps_x = int(size_x_km // (resolution_x // 1000))
        steps_y = int(size_y_km // (resolution_y // 1000))
        offset_x = steps_x // 2
        offset_y = steps_y // 2
        center_x = int(np.floor((x - meta['bounds'].left) / resolution_x))
        center_y = int(np.floor((meta['bounds'].top - y) / resolution_y))
        mask = array[
            :,
            center_y-offset_y:center_y+offset_y+1,
            center_x-offset_x:center_x+offset_x+1,
        ]
        return np.array(mask, dtype=mask.dtype)

    def _load_static_data(self, well_ids, raster_size):
        n_wells = int(sum(self.static_df.index.isin(well_ids)))
        return n_wells, self._load_static_df(well_ids, raster_size)

    def _load_static_df(self, well_ids, raster_size):
        t0 = time.time()
        logger.debug("start loading static data")
        static_data = []
        for well_id in well_ids:
            try:
                well_meta = self.static_df.loc[well_id]
            except KeyError:
                warnings.warn(f'could not find static data for well {well_id}')
                continue
            x, y = well_meta['x_coord'], well_meta['y_coord']

            relief = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._relief_array, self._relief_meta)[0]
            aspect = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._aspect_array, self._aspect_meta)[0]
            slope = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._slope_array, self._slope_meta)[0]
            lulc = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._lulc_array, self._lulc_meta)[0]
            huek_lga = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._huek_lga_array, self._huek_lga_meta)[0]
            huek_lgc = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._huek_lgc_array, self._huek_lgc_meta)[0]
            huek_lha = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._huek_lha_array, self._huek_lha_meta)[0]
            huek_lkf = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._huek_lkf_array, self._huek_lkf_meta)[0]
            seepage = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._seepage_array, self._seepage_meta)[0]
            gw_recharge = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._groundwater_recharge_array, self._groundwater_recharge_meta)[0]
            static_data.append(dict(
                proj_id=well_id,
                relief=relief,
                aspect=aspect,
                slope=slope,
                seepage=seepage,
                gw_recharge=gw_recharge,
                lulc=lulc,
                huek_lkf=huek_lkf,
                huek_lgc=huek_lgc,
                huek_lga=huek_lga,
                huek_lha=huek_lha,
            ))
        static_df = pd.DataFrame(static_data).set_index('proj_id')
        logger.debug(f"finished loading static data, time elapsed: {(time.time() - t0):.2f}s")
        yield static_df

    def _load_seq_for_chunk(self, chunk, start, end, raster_size):
        index = pd.date_range(HYRAS_START, HYRAS_END, freq=FREQ, name="datum")
        _dfs = []
        lais = []

        for proj_id, group in chunk.groupby(axis=0, level=0):
            x, y = self.static_df.loc[proj_id, 'x_coord'], self.static_df.loc[proj_id, 'y_coord']
            _df = pd.DataFrame(index=pd.date_range(start, end, freq=FREQ, name="datum"))
            _df['proj_id'] = proj_id

            humidity = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._humidity_array, self._humidity_meta)
            if raster_size > 1:
                humidity = np.kron(humidity, np.ones((5, 5)))
            _df['humidity'] = pd.Series(list(humidity), index=index)[start:end]

            temperature = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._temperature_array, self._temperature_meta)
            if raster_size > 1:
                temperature = np.kron(temperature, np.ones((5, 5)))
            _df['temperature'] = pd.Series(list(temperature), index=index)[start:end]

            precipitation = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._precipitation_array, self._precipitation_meta)
            _df['precipitation'] = pd.Series(list(precipitation), index=index)[start:end]

            gwl_asl = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._gwl_asl_array, self._gwl_asl_meta)
            _df['gwl_asl_raster'] = pd.Series(list(gwl_asl), index=index)[start:end]

            gwl_one_hot = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._gwl_one_hot_array, self._gwl_one_hot_meta)
            _df['gwl_raster_one_hot'] = pd.Series(list(gwl_one_hot), index=index)[start:end]
            _dfs.append(_df.reset_index())

            lai = self._get_mask_from_memory_array(x, y, raster_size, raster_size, self._lai_array, self._lai_meta)
            lai = pd.Series(list(lai), index=pd.RangeIndex(1, 13, name='month'), name='lai').to_frame().reset_index()
            lai['proj_id'] = proj_id
            lais.append(lai)
        df = pd.concat(_dfs)
        lai_df = pd.concat(lais)

        chunk['month'] = chunk.index.get_level_values('datum').copy().month
        chunk = (
            chunk
            .merge(df, on=['proj_id', 'datum'], how='left')
            .merge(lai_df, on=['proj_id', 'month'], how='left')
        )
        return chunk.drop(columns=['month']).set_index(['proj_id', 'datum'])

    def _load_sequential_data(self, well_ids, start, end, raster_size, max_chunk_size, use_gwl_raster_df=False):
        seq_df = self.gwl_df.loc[(slice(None), slice(start, end)), :]
        seq_df = seq_df[seq_df.index.get_level_values('proj_id').isin(well_ids)].copy().dropna()
        return len(seq_df), self._load_chunk(seq_df, start, end, raster_size, max_chunk_size, use_gwl_raster_df)

    def _load_chunk(self, seq_df, start, end, raster_size, max_chunk_size, use_gwl_raster_df=False):
        samples_per_group = seq_df.groupby(axis=0, level=0).count()['gwl_asl']
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
