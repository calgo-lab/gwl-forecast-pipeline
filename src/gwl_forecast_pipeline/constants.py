from strenum import StrEnum

import pandas as pd

FREQ = pd.offsets.Week(1, weekday=6)


DEFAULT_PREPROCESSOR_CACHE_FILES = {
    'static_index': 'static_idx.csv',
    'numeric_static_raster': 'static_numeric_features.npy',
    'categorical_static_raster': 'categorical_numeric_features.npy',
    'temporal_index': 'temporal_index.csv',
    'gwl_raster': 'gwl_raster.npy',
    'temporal_feature_raster': 'temporal_feature_raster.npy',
    'target': 'target.npy',
}
GWL_ASL_OFFSET = 15.

class FEATURES(StrEnum):
    HUMIDITY = 'humidity'
    TEMPERATURE = 'temperature'
    PRECIPITATION = 'precipitation'
    GROUNDWATER_LEVEL = 'gwl_asl'
    GROUNDWATER_LEVEL_ONE_HOT = 'gwl_one_hot'
    LEAF_AREA_INDEX = 'leaf_area_index'
    ELEVATION = 'elevation'
    ASPECT = 'aspect'
    SLOPE = 'slope'
    LAND_COVER = 'land_cover'
    ROCK_TYPE = 'ga'
    CHEMICAL_ROCK_TYPE = 'gc'
    CAVITY_TYPE = 'ha'
    PERMEABILITY = 'kf'
    PERCOLATION = 'percolation'
    GROUNDWATER_RECHARGE = 'gw_recharge'
    WELL_META = 'well_meta'

PROCESSED_DATA_PATHS = {
    FEATURES.GROUNDWATER_RECHARGE: 'hydrogeologic',
    FEATURES.HUMIDITY: 'meteorologic',
    FEATURES.TEMPERATURE: 'meteorologic',
    FEATURES.PRECIPITATION: 'meteorologic',
    FEATURES.GROUNDWATER_LEVEL: 'groundwater_level',
    FEATURES.GROUNDWATER_LEVEL_ONE_HOT: 'groundwater_level',
    FEATURES.LEAF_AREA_INDEX: 'vegetational',
    FEATURES.ELEVATION: 'relief',
    FEATURES.ASPECT: 'relief',
    FEATURES.SLOPE: 'relief',
    FEATURES.LAND_COVER: 'land_cover',
    FEATURES.ROCK_TYPE: 'hydrogeologic',
    FEATURES.CHEMICAL_ROCK_TYPE: 'hydrogeologic',
    FEATURES.CAVITY_TYPE: 'hydrogeologic',
    FEATURES.PERMEABILITY: 'hydrogeologic',
    FEATURES.PERCOLATION: 'hydrogeologic',
    FEATURES.WELL_META: 'well_meta',
}


class DATA_SETS(StrEnum):
    HUEK250 = 'huek_250'
    SWR1000 = 'swr_1000'
    GWN1000 = 'gwn_1000'
    EUDEM = 'eudem'
    CLC = 'clc'
    LAI = 'lai'
    HYRAS = 'hyras'
    GWL = 'gwl'
    WELL_META = 'well_meta'
    BENCHMARK_RESULTS = 'benchmark_results'


BENCHMARK_SUFFIX = '_wo_benchmark'

CATEGORICAL_STATIC_RASTER_FEATURES = [
    FEATURES.LAND_COVER,
    FEATURES.ROCK_TYPE,
    FEATURES.CHEMICAL_ROCK_TYPE,
    FEATURES.CAVITY_TYPE,
    FEATURES.PERMEABILITY,
]

ROCK_TYPE_CHANNELS = ['k.A.', 'S', 'Me', 'Ma', 'G']
CHEMICAL_ROCK_TYPE_CHANNELS = ['k.A.', 's', 's/o', 'm', 'k', 'o', 'g', 'g/h', 'h', 'nb', 'a', 'Gew']
CAVITY_TYPE_CHANNELS = ['k.A.', 'P', 'K/P', 'K', 'K/KA', 'nb', 'G']

CATEGORICAL_STATIC_RASTER_FEATURES_CARDINALITIES = {
    FEATURES.LAND_COVER: 171,
    FEATURES.ROCK_TYPE: len(ROCK_TYPE_CHANNELS),
    FEATURES.CAVITY_TYPE: len(CAVITY_TYPE_CHANNELS),
    FEATURES.CHEMICAL_ROCK_TYPE: len(CHEMICAL_ROCK_TYPE_CHANNELS),
    FEATURES.PERMEABILITY: 13,
}

TARGET_COL = 'gwl_asl_target'
GROUNDWATER_LEVEL_NORM_PER_WELL = 'gwl_local_norm'
HYRAS_START = pd.Timestamp(1951, 1, 7)
HYRAS_END = pd.Timestamp(2021, 1, 3)

GWL_RASTER_FEATURES = [
    FEATURES.GROUNDWATER_LEVEL,
    FEATURES.GROUNDWATER_LEVEL_ONE_HOT,
    GROUNDWATER_LEVEL_NORM_PER_WELL,
]