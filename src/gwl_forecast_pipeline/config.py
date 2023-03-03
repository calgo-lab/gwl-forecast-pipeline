import logging.config
import os
import sys

from decouple import AutoConfig
from yaml import load, Loader

# path to settings.ini
SETTINGS_PATH = os.getenv('SETTINGS_PATH')
config = AutoConfig(search_path=SETTINGS_PATH)

# URL for EU-DEM elevation raw data (https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1)
EUDEM_ELEVATION_URL = config('EUDEM_ELEVATION_URL', default='')

# URL for EU-DEM slope raw data (https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1-0-and-derived-products/slope)
EUDEM_SLOPE_URL = config('EUDEM_SLOPE_URL', default='')

# URL for EU-DEM aspect raw data (https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1-0-and-derived-products/aspect)
EUDEM_ASPECT_URL = config('EUDEM_ASPECT_URL', default='')

# URL for groundwater level raw time series data, provided by BGR (https://www.bgr.bund.de)
GWL_URL = config('GWL_URL', default='')

# URL for data on detected implausible periods of groundwater level time series' to be removed
DROP_GWL_PERIODS_URL = config('DROP_GWL_PERIODS_URL', default='')

# URL for benchmark metrics from related study by Wunsch et al. (https://doi.org/10.1038/s41467-022-28770-2)
BENCHMARK_RESULTS_URL = config('BENCHMARK_RESULTS_URL', default='')

# URL for static features and meta data on groundwater wells, provided by BGR (https://www.bgr.bund.de)
WELL_META_URL = config('WELL_META_URL', default='')

# URL for HYRAS humidity raw data (https://www.dwd.de/DE/leistungen/hyras/hyras.html)
HYRAS_HURS_URL = config('HYRAS_HURS_URL', default='https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/humidity/hurs_hyras_5_1951_2020_v5-0_de.nc')

# URL for HYRAS mean air temperature raw data (https://www.dwd.de/DE/leistungen/hyras/hyras.html)
HYRAS_TAS_URL = config('HYRAS_TAS_URL', default='https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/tas_hyras_5_1951_2020_v5-0_de.nc')

# URL for HYRAS precipitation raw data (https://www.dwd.de/DE/leistungen/hyras/hyras.html)
HYRAS_PR_URL = config('HYRAS_PR_URL', default='https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/pr_hyras_1_1931_2020_v5-0_de.nc')

# URL for Mean Annual Rate of Percolation data (https://www.bgr.bund.de/DE/Themen/Boden/Bilder/Bod_Themenkarten_HAD_4-5_g.html)
SWR_1000_URL = config('SWR_1000_URL', default='https://download.bgr.de/bgr/boden/SWR1000/geotiff/swr1000_250.zip')

# URL for Mean Annual Groundwater Recharge Rate data (https://www.bgr.bund.de/DE/Themen/Wasser/Projekte/abgeschlossen/Beratung/Had/had_projektbeschr.html)
GWN_1000_URL = config('GWN_1000_URL', default='https://download.bgr.de/BGR/Grundwasser/GWN1000/tiff/GWN1000.zip')

# URL for hydrogeologic overview map data (https://www.bgr.bund.de/DE/Themen/Wasser/Projekte/laufend/Beratung/Huek200/huek200_projektbeschr.html)
HUEK_250_URL = config('HUEK_250_URL', default='https://download.bgr.de/bgr/Grundwasser/huek250/shp/huek250.zip')

# URL for Copernicus Leaf Area Index raw data (https://land.copernicus.eu/global/products/lai)
LAI_URL = config('LAI_URL', default='https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/MAPPE/MAPPE_Europe/LATEST/Vegetation/D_18_LAI/D_18_LAI.zip')

# URL for Copernicus Land Cover raw data (https://land.copernicus.eu/global/products/lc)
CLC_URL = config('CLC_URL', default='https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2018/E000N60/E000N60_PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif')

# Path for data storage
DATA_PATH = config('DATA_PATH', default='')

# path for model storage
MODEL_PATH = config('MODEL_PATH', default='')

# path to store preprocessed data
PREPROCESSOR_CACHE_PATH = config('PREPROCESSOR_CACHE_PATH', default='')

# path to store predictions
PREDICTION_RESULT_PATH = config('PREDICTION_RESULT_PATH', default='')

# path to store results of hyperparameter optimization
HYPEROPT_RESULT_PATH = config('HYPEROPT_RESULT_PATH', default='')

# path to store scores
SCORE_RESULT_PATH = config('SCORE_RESULT_PATH', default='')

# path for csv logging of model training
CSV_LOGGER_PATH = config('CSV_LOGGER_PATH', default=None)

# path for tensorboard logs of model training
TENSORBOARD_PATH = config('TENSORBOARD_PATH', default=None)

# whether to use a GPU for model training
GPU = config('GPU', default=False, cast=bool)

# whether to keep preprocessed data in RAM on model training
DATA_IN_MEMORY = config('DATA_IN_MEMORY', default=False, cast=bool)

# path to logging configuration file
LOGGING_CONF = config('LOGGING_CONF', default=None)

if LOGGING_CONF:
    with open(LOGGING_CONF) as f:
        config = load(f.read(), Loader=Loader)
    logging.config.dictConfig(config)
else:
    root = logging.getLogger('gwl_forecast_pipeline')
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
