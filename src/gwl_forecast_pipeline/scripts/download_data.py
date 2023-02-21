import logging
import os
import zipfile

import patoolib
import requests
from pydantic import ConfigError
from tqdm import tqdm

from .. import config as config
from ..constants import *

logger = logging.getLogger(__name__)

def _get_file_over_http(url, path):
    response = requests.get(url, stream=True, timeout=(5, 10))
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    try:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
    except requests.exceptions.ReadTimeout as e:
        logger.exception(e)
    finally:
        progress_bar.close()
        file.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        if os.path.exists(path):
            os.remove(path)
        raise ValueError(f"could not retrieve file {url}")


def _get_data(base_path, slug, name, files=None, zip=False):
    path = os.path.join(base_path, slug)
    if not os.path.exists(path):
        os.mkdir(path)
    url = getattr(config, f'{slug.upper()}_URL')
    if not url:
        msg = f'{name} data could not be fetched because ' \
              f'{slug.upper()}_URL environment variable is not set. Please ' \
              f'configure {slug.upper()}_URL in .env'
        raise ConfigError(msg)
    download_path = os.path.join(path, os.path.basename(url))
    files = files or [download_path]
    if all([os.path.exists(os.path.join(path, f)) for f in files]):
        logger.info(f'Found existing {name} data: {path}. Skip download')
        return 'skip'
    else:
        logger.info(f'Download {name} data...')
        try:
            _get_file_over_http(url, download_path)
        except ValueError as e:
            logger.error(e)
        else:
            logger.info(f'...done.')
            if zip:
                logger.info(f'Extract files...')
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(path)
                os.remove(download_path)
                logger.info(f'...done.')
            logger.info(f'Saved {name} data to {path}')


def get_gwl_data(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.GWL, name="groundwater levels")
    except ConfigError as e:
        logger.error(e)


def get_gwl_drop_periods(raw_data_path):
    try:
        _get_data(raw_data_path, slug="drop_gwl_periods",
                  name="Groundwater level data to drop")
    except ConfigError as e:
        logger.error(e)

def get_well_meta(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.WELL_META, name='Meta Data on groundwater wells',
                  files=['well_meta.csv', 'gwl_germany_features.csv'], zip=True)
    except ConfigError as e:
        logger.error(e)


def get_benchmark_data(external_data_path):
    try:
        _get_data(external_data_path, slug=DATA_SETS.BENCHMARK_RESULTS, name='Benchmark study results',
                  files=['benchmark_wells_wunsch2022.csv'])
    except ConfigError as e:
        logger.error(e)


def get_hyras_data(raw_data_path):
    hyras_path = os.path.join(raw_data_path, 'hyras')
    if not os.path.exists(hyras_path):
        os.mkdir(hyras_path)
    for name, slug in [
        (FEATURES.HUMIDITY, 'hurs'),
        (FEATURES.TEMPERATURE, 'tas'),
        (FEATURES.PRECIPITATION, 'pr'),
    ]:
        try:
            _get_data(hyras_path, slug=f'{DATA_SETS.HYRAS}_{slug}', name=name)
        except ConfigError as e:
            logger.error(e)


def get_eudem_data(raw_data_path):
    eudem_path = os.path.join(raw_data_path, DATA_SETS.EUDEM)
    if not os.path.exists(eudem_path):
        os.mkdir(eudem_path)
    for slug, files in [
        (FEATURES.ELEVATION, ['EUD_CP-DEMS_4500025000-AA.tif', 'EUD_CP-DEMS_4500035000-AA.tif']),
        (FEATURES.ASPECT, ['EUD_CP-ASPC_4500025000-AA.tif', 'EUD_CP-ASPC_4500035000-AA.tif']),
        (FEATURES.SLOPE, ['EUD_CP-SLOP_4500025000-AA.tif', 'EUD_CP-SLOP_4500035000-AA.tif']),
    ]:
        try:
            result = _get_data(eudem_path, slug=f'{DATA_SETS.EUDEM}_{slug}', name=f'EU-DEM {slug.lower()}',
                               files=files, zip=True)
        except ConfigError as e:
            logger.error(e)
        else:
            if result == 'skip':
                continue
            for file in files:
                path = os.path.join(eudem_path, f'{DATA_SETS.EUDEM}_{slug.lower()}')
                if os.path.exists(os.path.join(path, file[:-3]+'rar')):
                    zip_file = os.path.join(path, file[:-3]+'rar')
                else:
                    zip_file = os.path.join(path, file[:-3] + 'zip')
                patoolib.extract_archive(zip_file, outdir=path)
                os.remove(os.path.join(path, zip_file))


def get_swr1000(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.SWR1000, name='SWR1000 (percolation)',
              files=['swr1000_250.tif'], zip=True)
    except ConfigError as e:
        logger.error(e)


def get_gwn1000(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.GWN1000, name="GWN1000 (groundwater recharge)",
              files=['GWN1000_v1/tiff/GWN1000__3034_v1_raster1.tif'], zip=True)
    except ConfigError as e:
        logger.error(e)


def get_huek250(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.HUEK250, name="HUEK (Hydrogeologic Overview Map)",
              files=['huek250_v103/shp/huek250__25832_v103_poly.shp'], zip=True)
    except ConfigError as e:
        logger.error(e)


def get_lai_data(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.LAI, name="LAI (leaf area index)",
              files=['lai_01/w001001.adf', 'lai_02/w001001.adf', 'lai_03/w001001.adf',
                     'lai_04/w001001.adf', 'lai_05/w001001.adf', 'lai_06/w001001.adf',
                     'lai_07/w001001.adf', 'lai_08/w001001.adf', 'lai_09/w001001.adf',
                     'lai_10/w001001.adf', 'lai_11/w001001.adf', 'lai_12/w001001.adf'],
              zip=True)
    except ConfigError as e:
        logger.error(e)

def get_clc_data(raw_data_path):
    try:
        _get_data(raw_data_path, slug=DATA_SETS.CLC, name='CLC (Corine Land Cover)')
    except ConfigError as e:
        logger.error(e)


def create_data_path():
    raw_data_path = os.path.join(config.DATA_PATH, 'raw')
    external_data_path = os.path.join(config.DATA_PATH, 'external')
    if not config.DATA_PATH:
        msg = 'DATA_PATH environment variable is not set. Please configure DATA_PATH ' \
              'variable in .env'
        logger.error(msg)
        raise ValueError(msg)
    else:
        if not os.path.exists(config.DATA_PATH):
            try:
                os.mkdir(config.DATA_PATH)
                os.mkdir(raw_data_path)
                os.mkdir(external_data_path)
            except Exception as e:
                logger.exception(e)
        else:
            logger.warning('data path already exists. existing data may be overwritten')
        return raw_data_path, external_data_path


def main():
    raw_data_path, external_data_path = create_data_path()
    get_hyras_data(raw_data_path)
    get_eudem_data(raw_data_path)
    get_gwl_data(raw_data_path)
    get_gwl_drop_periods(raw_data_path)
    get_well_meta(raw_data_path)
    get_benchmark_data(external_data_path)
    get_swr1000(raw_data_path)
    get_huek250(raw_data_path)
    get_gwn1000(raw_data_path)
    get_clc_data(raw_data_path)
    get_lai_data(raw_data_path)
