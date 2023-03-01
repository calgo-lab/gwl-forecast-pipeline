import logging
import os
import shutil
import sys
from subprocess import Popen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from geocube.api.core import make_geocube
from pyproj import Transformer, CRS
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import Resampling
from rasterio.windows import Window

from ..constants import *
from ..features.preprocessing.mixins import FeatureTransformerMixin
from .. import config as config

logger = logging.getLogger(__name__)


def prepare_gwn1000_data(raw_raster_path, target_base_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.GROUNDWATER_RECHARGE])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info('Prepare groundwater recharge data...')
    file = os.path.join(path, f'{FEATURES.GROUNDWATER_RECHARGE}.tif')
    shutil.copyfile(raw_raster_path, file)
    logger.info("...done.")
    return file


def prepare_eudem_data(raw_raster_base_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.ELEVATION])
    if not os.path.exists(path):
        os.mkdir(path)
    for name, files in [
        (FEATURES.ELEVATION, ['EUD_CP-DEMS_4500025000-AA.tif', 'EUD_CP-DEMS_4500035000-AA.tif']),
        (FEATURES.ASPECT, ['EUD_CP-ASPC_4500025000-AA.tif', 'EUD_CP-ASPC_4500035000-AA.tif']),
        (FEATURES.SLOPE, ['EUD_CP-SLOP_4500025000-AA/EUD_CP-SLOP_4500025000-AA.tif',
                          'EUD_CP-SLOP_4500035000-AA/EUD_CP-SLOP_4500035000-AA.tif']),
    ]:
        slug = f'{DATA_SETS.EUDEM}_{name.lower()}'
        logger.info(f"Prepare relief data ({name})...")
        target_path = os.path.join(path, f'{name}.tif')
        _files = [os.path.join(raw_raster_base_path, slug, f) for f in files]
        _prepare_eudem_data(_files, target_path, raster_reference_path)
        logger.info(f"...done.")


def _prepare_eudem_data(raw_raster_files, target_path, raster_reference_path):
    tmp_path = os.path.join(os.path.dirname(target_path), 'tmp.tif')
    merge(raw_raster_files, dst_path=tmp_path, dst_kwds={'BIGTIFF': 'YES'})
    src_ds = rioxarray.open_rasterio(tmp_path)
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    dst_ds = src_ds.rio.reproject_match(reference_ds, Resampling.bilinear)
    dst_ds.rio.to_raster(target_path)
    os.remove(tmp_path)


def prepare_gwl_data(raw_base_path, target_base_path, raster_reference_path):
    data_path = config.DATA_PATH
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.GROUNDWATER_LEVEL])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info(f"Prepare groundwater level data...")
    df = pd.read_feather(os.path.join(raw_base_path, 'gwl', 'groundwater_levels.feather'))
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    hyras_ds = xr.open_mfdataset(os.path.join(raw_base_path, 'hyras', 'hyras_hurs', '*.nc'), engine='netcdf4')
    min_date, max_date = np.min(hyras_ds.time.to_numpy()), np.max(hyras_ds.time.to_numpy())
    well_meta_df = pd.read_csv(os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.WELL_META], f'{FEATURES.WELL_META}.csv'))
    benchmark_df = pd.read_csv(os.path.join(data_path, 'external', 'benchmark_results', 'benchmark_wells_wunsch2022.csv'))
    df = df.groupby('proj_id').resample(FREQ, on='datum').mean(numeric_only=True).reset_index()
    df = df.loc[(df['datum'] >= min_date) & (df['datum'] <= max_date), :]

    # remove drop periods
    drop_periods = pd.read_csv(
        os.path.join(raw_base_path, 'drop_gwl_periods', 'drop_periods.csv'),
        parse_dates=['start', 'end']
    )
    df = df.merge(drop_periods, left_on='proj_id', right_on='well_id', how='left')
    df['start'] = df['start'].mask(df['start'].isnull(), pd.Timestamp(2100, 1, 1))
    df['end'] = df['end'].mask(df['end'].isnull(), pd.Timestamp(1900, 1, 1))
    df = df[~df['datum'].between(df['start'], df['end'])]

    # remove constant periods
    const_records = []
    for well_id, group in df.groupby('proj_id'):
        levels = group.copy()[['datum', 'nn_messwert']]
        levels['cumsum'] = (levels['nn_messwert'].diff(1) != 0).astype('int').cumsum()
        _const_periods = []
        for cumsum, group in levels.groupby('cumsum'):
            if len(group) > 4:
                group['proj_id'] = well_id
                group['period_length'] = len(group)
                const_records.append(group[['proj_id', 'datum', 'period_length']].copy())

    const_records_df = pd.concat(const_records).set_index(['proj_id', 'datum'])
    df.set_index(['proj_id', 'datum'], inplace=True)
    df = df.loc[
        df.index.difference(const_records_df[const_records_df['period_length'] > 6].index),
        :
    ].copy().reset_index()

    # temporal interpolation
    df.drop_duplicates(subset=['proj_id', 'datum'], inplace=True)
    groups = []
    for well_id, group in df.groupby('proj_id'):
        _df = group.set_index('datum')['nn_messwert'].resample(FREQ).interpolate(method='linear', limit=6).reset_index().copy()
        _df['proj_id'] = well_id
        groups.append(_df)
    df = pd.concat(groups).reset_index(drop=True)

    # merge meta
    df = df.merge(well_meta_df[['proj_id', 'x_coord', 'y_coord']], on='proj_id')
    df.rename(columns={'datum': 'time', 'x_coord': 'x', 'y_coord': 'y',
                       'nn_messwert': 'gwl_asl'}, inplace=True)
    df['gwl_asl'] = df['gwl_asl'].astype(np.float32)

    # rasterize
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']))
    gdf.set_crs(epsg=3034, inplace=True)
    ds = make_geocube(vector_data=gdf, measurements=['gwl_asl'], group_by='time',
                      like=reference_ds)
    ds['gwl_asl'].rio.to_raster(os.path.join(path, 'gwl_asl_sparse.tif'))
    with rasterio.open(os.path.join(path, 'gwl_asl_sparse.tif')) as src:
        meta = src.meta

    # one hot encode measured values
    data = ds['gwl_asl'].to_numpy()
    data_one_hot = data.copy()
    data_one_hot[np.isnan(data)] = 0
    data_one_hot[~np.isnan(data)] = 1

    with rasterio.open(os.path.join(path, f'{FEATURES.GROUNDWATER_LEVEL_ONE_HOT}.tif'), 'w',
                       **meta) as dst:
        for idx, layer in enumerate(data_one_hot, start=1):
            dst.write_band(idx, layer)
    del data_one_hot

    # spatial interpolation
    interpolated_data = data.copy()
    # interpolate in chunks to reduce RAM usage
    start = 0
    for _ in range(int(np.ceil(interpolated_data.shape[0] / 10))):
        end = start + 10
        interpolated_data[start:end] = FeatureTransformerMixin.fill_raster(interpolated_data[start:end],
                                                                           max_iter=7, shift_fill=False)
        start = end

    with rasterio.open(os.path.join(path, f'{FEATURES.GROUNDWATER_LEVEL}.tif'), 'w',
                       **meta) as dst:
        for idx, layer in enumerate(interpolated_data, start=1):
            dst.write_band(idx, layer)
    del interpolated_data
    del ds
    del data
    del gdf

    df_bench = df[~df['proj_id'].isin(benchmark_df['well_id'].unique())]
    gdf_bench = gpd.GeoDataFrame(df_bench, geometry=gpd.points_from_xy(df_bench['x'], df_bench['y']))
    gdf_bench.set_crs(epsg=3034, inplace=True)
    ds_bench = make_geocube(vector_data=gdf_bench, measurements=['gwl_asl'], group_by='time',
                            like=reference_ds)
    ds_bench['gwl_asl'].rio.to_raster(os.path.join(path, f'{FEATURES.GROUNDWATER_LEVEL}_sparse{BENCHMARK_SUFFIX}.tif'))
    data_bench = ds_bench['gwl_asl'].to_numpy()

    data_bench_one_hot = data_bench.copy()
    data_bench_one_hot[np.isnan(data_bench)] = 0
    data_bench_one_hot[~np.isnan(data_bench)] = 1

    with rasterio.open(os.path.join(path, f'{FEATURES.GROUNDWATER_LEVEL_ONE_HOT}{BENCHMARK_SUFFIX}.tif'), 'w',
                       **meta) as dst:
        for idx, layer in enumerate(data_bench_one_hot, start=1):
            dst.write_band(idx, layer)

    interpolated_data_bench = data_bench.copy()
    # interpolate in chunks to reduce RAM usage
    start = 0
    for _ in range(int(np.ceil(interpolated_data_bench.shape[0] / 10))):
        end = start + 10
        interpolated_data_bench[start:end] = FeatureTransformerMixin.fill_raster(
            interpolated_data_bench[start:end], max_iter=7, shift_fill=False)
        start = end

    with rasterio.open(os.path.join(path, f'{FEATURES.GROUNDWATER_LEVEL}{BENCHMARK_SUFFIX}.tif'), 'w',
                       **meta) as dst:
        for idx, layer in enumerate(interpolated_data_bench, start=1):
            dst.write_band(idx, layer)

    # remove wells close to border
    border_wells = []
    with rasterio.open(
            os.path.join(data_path, 'processed',
                         PROCESSED_DATA_PATHS[FEATURES.HUMIDITY],
                         f'{FEATURES.HUMIDITY}.tif')
    ) as src:
        for proj_id in df['proj_id'].unique():
            x, y = well_meta_df.loc[well_meta_df['proj_id'] == proj_id, ['x', 'y']].iloc[0]
            window = Window(x - MAX_RASTER_SIZE // 2, y - MAX_RASTER_SIZE // 2, MAX_RASTER_SIZE, MAX_RASTER_SIZE)
            data = src.read(1, window=window)
            if np.any(np.isnan(data)):
                border_wells.append(proj_id)
    df = df[~df['proj_id'].isin(border_wells)].reset_index(drop=True)
    df[['proj_id', 'time', 'gwl_asl']].to_feather(os.path.join(path, 'gwl_asl.feather'))


    logger.info(f"...done.")


def prepare_well_meta_data(raw_data_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.WELL_META])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info(f"Prepare well meta data...")
    well_meta_df = pd.read_csv(os.path.join(raw_data_path, 'well_meta.csv'))
    static_df = pd.read_csv(os.path.join(raw_data_path, 'gwl_germany_features.csv'))

    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326")
    well_meta_df[['lat', 'lon']] = well_meta_df[['x_coord', 'y_coord']].apply(
        lambda row: transformer.transform(row['x_coord'], row['y_coord']), axis=1,
        result_type="expand")
    transformer = Transformer.from_crs("EPSG:32632", "EPSG:3034")
    well_meta_df[['y_coord', 'x_coord']] = well_meta_df[['x_coord', 'y_coord']].apply(
        lambda row: transformer.transform(row['x_coord'], row['y_coord']), axis=1,
        result_type="expand")

    well_meta_df = well_meta_df.merge(static_df, left_on='proj_id',
                                      right_on='station_id')
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    well_meta_df['y'], well_meta_df['x'] = rasterio.transform.rowcol(
        reference_ds.rio.transform(), well_meta_df['x_coord'].values,
        well_meta_df['y_coord'].values)
    well_meta_df.to_csv(os.path.join(path, f'{FEATURES.WELL_META}.csv'), index=False)
    logger.info(f"...done.")


def prepare_swr1000_data(raw_raster_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.PERCOLATION])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info('Prepare percolation data...')
    src_ds = rioxarray.open_rasterio(raw_raster_path)
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    dst_ds = src_ds.rio.reproject_match(reference_ds, Resampling.bilinear)
    dst_ds.rio.to_raster(os.path.join(path, f'{FEATURES.PERCOLATION}.tif'))
    logger.info('...done')


def prepare_clc_data(raw_raster_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.LAND_COVER])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info('Prepare land cover data...')
    src_ds = rioxarray.open_rasterio(raw_raster_path)
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    dst_ds = src_ds.rio.reproject_match(reference_ds, Resampling.bilinear)
    dst_ds.rio.to_raster(os.path.join(path, f'{FEATURES.LAND_COVER}.tif'))
    logger.info('...done.')


def prepare_lai_data(raw_raster_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.LEAF_AREA_INDEX])
    if not os.path.exists(path):
        os.mkdir(path)
    tmp_path = os.path.join(path, 'tmp.tif')
    files = [os.path.join(raw_raster_path, f'lai_{i:0>2}') for i in range(1, 13)]
    logger.info('Prepare leaf area index data...')
    for file in files:
        process = Popen(['gdal_translate', f'{file}', '-of', 'GTiff', '-q',
                         f'{os.path.join(path, os.path.basename(file))}.tif'])
        process.communicate()
    files = [os.path.join(path, f'lai_{i:0>2}.tif') for i in range(1, 13)]
    with rasterio.open(files[0]) as src0:
        meta = src0.meta
        meta.update(count=len(files))
        meta.update(crs=CRS.from_epsg(3035))
    with rasterio.open(tmp_path, 'w', **meta) as dst:
        for _id, layer in enumerate(files, start=1):
            with rasterio.open(layer) as src:
                data = src.read(1)
                dst.write_band(_id, data)
    src_ds = rioxarray.open_rasterio(tmp_path)
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    dst_ds = src_ds.rio.reproject_match(reference_ds, Resampling.bilinear)
    dst_ds.rio.to_raster(os.path.join(path, f'{FEATURES.LEAF_AREA_INDEX}.tif'))
    os.remove(tmp_path)
    for file in files:
        os.remove(file)
    logger.info('...done.')


def prepare_huek_data(raw_raster_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.ROCK_TYPE])
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info('Prepare hydrogeologic static data...')
    target_path = os.path.join(path, f'{FEATURES.PERMEABILITY}.tif')
    process = Popen(
        ['gdal_rasterize', '-l', 'huek250__25832_v103_poly', '-a', 'kf', '-tr',
         '1000.0', '1000.0', '-a_nodata', '0', '-ot', 'Int32', '-q',
         '-of', 'GTiff', f'{os.path.join(raw_raster_path, "huek250__25832_v103_poly.shp")}',
         target_path])
    process.communicate()
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    rioxarray.open_rasterio(target_path)\
        .rio.reproject_match(reference_ds, Resampling.nearest)\
        .rio.to_raster(target_path)

    for field, vals in [
        (FEATURES.ROCK_TYPE, ROCK_TYPE_CHANNELS),
        (FEATURES.CAVITY_TYPE, CAVITY_TYPE_CHANNELS),
        (FEATURES.CHEMICAL_ROCK_TYPE, CHEMICAL_ROCK_TYPE_CHANNELS),
    ]:
        target_path = os.path.join(os.path.join(path, f'{field.lower()}.tif'))
        process = Popen(
            ['gdal_rasterize', '-l', 'huek250__25832_v103_poly', '-burn', '0', '-init', '0', '-tr',
             '1000.0', '1000.0', '-a_nodata', '0', '-ot', 'Int32', '-of', 'GTiff', '-q',
             f'{os.path.join(raw_raster_path, "huek250__25832_v103_poly.shp")}', target_path])
        process.communicate()
        for num, val in enumerate(vals):
            process = Popen(
                ['gdal_rasterize', '-l', 'huek250__25832_v103_poly', '-burn', f'{num}',
                 '-where', f'{field.upper()}=\'{val}\'', '-q',
                 f'{os.path.join(raw_raster_path, "huek250__25832_v103_poly.shp")}', target_path])
            process.communicate()
        rioxarray.open_rasterio(target_path)\
            .rio.reproject_match(reference_ds, Resampling.nearest)\
            .rio.to_raster(target_path)
    logger.info('...done.')


def prepare_hyras_data(raw_data_path, target_base_path, raster_reference_path):
    path = os.path.join(target_base_path, PROCESSED_DATA_PATHS[FEATURES.HUMIDITY])
    tmp_path = os.path.join(path, 'tmp.tif')
    if not os.path.exists(path):
        os.mkdir(path)
    reference_ds = rioxarray.open_rasterio(raster_reference_path)
    for name, slug in [
        (FEATURES.HUMIDITY, 'hurs'),
        (FEATURES.TEMPERATURE, 'tas'),
        (FEATURES.PRECIPITATION, 'pr')
    ]:
        logger.info(f'Prepare meteorologic data ({name})...')
        ds = xr.open_mfdataset(os.path.join(raw_data_path, f'hyras_{slug}', '*.nc'), engine='netcdf4')
        ds = ds[slug]
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
        ds = ds.reset_coords(names=['lat', 'lon'], drop=True)
        ds = ds.resample(time='1W').mean()
        ds = ds.where(ds.time >= HYRAS_START, drop=True)
        ds = ds.rio.write_crs("epsg:3034")
        del ds.attrs['grid_mapping']
        _ds = ds.isel(time=0)
        _ds = _ds.rio.reproject_match(reference_ds, Resampling.cubic)
        _ds.rio.to_raster(tmp_path)
        with rasterio.open(tmp_path) as tmp:
            meta = tmp.meta
            meta.update(count=len(ds.time))
        with rasterio.open(os.path.join(path, f'{name}.tif'), 'w', **meta) as dst:
            for count, (_, group) in enumerate(ds.groupby('time'), start=1):
                data = group.rio.reproject_match(reference_ds, Resampling.cubic)
                dst.write_band(count, data.to_numpy())
        os.remove(tmp_path)
        logger.info('...done.')


def create_data_path():
    data_path = config.DATA_PATH
    if not data_path:
        msg = 'DATA_PATH environment variable is not set. Please configure DATA_PATH ' \
              'variable in .env'
        logger.error(msg)
        raise ValueError(msg)
    processed_data_path = os.path.join(data_path, 'processed')
    if not os.path.exists(os.path.join(data_path, 'raw')):
        raise ValueError('Cannot prepare data. Raw data is missing. Please run '
                         'download_data.py first')
    else:
        if not os.path.exists(processed_data_path):
            try:
                os.mkdir(processed_data_path)
            except Exception as e:
                logger.exception(e)
        return processed_data_path


def main():
    base_path = config.DATA_PATH
    raw_data_path = os.path.join(base_path, 'raw')
    data_path = create_data_path()

    reference_raster_path = prepare_gwn1000_data(
        os.path.join(raw_data_path, DATA_SETS.GWN1000, 'GWN1000_v1', 'tiff',
                     'GWN1000__3034_v1_raster1.tif'), data_path)
    prepare_swr1000_data(os.path.join(raw_data_path, DATA_SETS.SWR1000, 'swr1000_250.tif'),
                         data_path, reference_raster_path)
    prepare_clc_data(os.path.join(raw_data_path, DATA_SETS.CLC,
                                  'E000N60_PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif'),
                     data_path, reference_raster_path)
    prepare_eudem_data(os.path.join(raw_data_path, DATA_SETS.EUDEM), data_path,
                       reference_raster_path)
    prepare_lai_data(os.path.join(raw_data_path, DATA_SETS.LAI), data_path, reference_raster_path)
    prepare_hyras_data(os.path.join(raw_data_path, DATA_SETS.HYRAS), data_path,
                       reference_raster_path)
    prepare_huek_data(os.path.join(raw_data_path, DATA_SETS.HUEK250, 'huek250_v103', 'shp'),
                      data_path, reference_raster_path)
    prepare_well_meta_data(os.path.join(raw_data_path, DATA_SETS.WELL_META), data_path,
                           reference_raster_path)
    prepare_gwl_data(raw_data_path, data_path, reference_raster_path)
