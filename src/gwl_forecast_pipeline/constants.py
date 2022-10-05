import pandas as pd

FREQ = pd.offsets.Week(1, weekday=6)
CATEGORICAL_STATIC_RASTER_FEATURES = [
    'huek_lga',
    'huek_lgc',
    'huek_lha',
    'huek_lkf',
    'lulc',
]
CATEGORICAL_STATIC_RASTER_FEATURES_CARDINALITIES = {
    'lulc': 38,
    'huek_lga': 6,
    'huek_lha': 5,
    'huek_lgc': 9,
    'huek_lkf': 14,
}
NUMERIC_STATIC_RASTER_FEATURES = [
    'aspect_cos',
    'aspect_sin',
    'gw_recharge',
    'seepage',
    'slope',
    'relief',
]
TEMPORAL_RASTER_FEATURES = [
    'humidity',
    'lai',
    'precipitation',
    'temperature',
]
GWL_SCALAR_FEATURES = [
    'gwl_asl',
    'gwl_delta',
    'gwl_asl_norm',
    'gwl_delta_norm',
    'gwl_asl_norm_per_group',
    'gwl_delta_norm_per_group',
]
GWL_RASTER_FEATURES = [
    'gwl_asl_raster',
    'gwl_raster_one_hot',
    'gwl_asl_raster_scaled_per_group',
]
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
