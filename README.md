# Groundwater Level Forecast Pipeline

Data- and Machine-Learning-Pipeline for the prediction of Groundwater Levels in Germany.

## About

The goal of the given study is to develop a _global_ groundwater level time series forecasting model
for Germany with the help of Convolutional Neural Networks (CNN) in a sequence to sequence
(seq2seq) setup. The model takes observed groundwater levels sequences as well as exogenous
attributes, such as weather data and geomorphic features as inputs in the form of temporal 2D
map data sequences (3D input) to predict a future sequence of groundwater levels at a given location.
Model inputs are gridded groundwater level observations as well as geological map data on relief,
soil and hydrogeologic attributes. Further inputs are gridded
weather parameters (humidity, temperature and precipitation), vegetational data, and land cover
data. The given set of input parameters covers a wide range of possible factors of influence and
offers the chance for the model to learn seasonal impacts, natural effects like inflows and outflows
as well as human impact on the groundwater, such as withdrawals in urban, industrial or agricultural areas. The approach using 3D input data enables the model to extract spatio-temporal
relationships, while the global approach helps to learn the underlying relationships by a great
variety of hydrogeologic conditions.

The given python-package holds the functionality of the entire Machine Learning Pipeline, offering
a convenient interface for:
* Downloading Raw Data
* Preparing Raw Data (Data Harmonization)
* Preprocessing Data (Feature Engineering and Normalization)
* Training Models
* Hyperparameter Optimization
* Run Forecasts
* Model Evaluation


## Requirements

### Python

To install the package a python version of `3.8` (or higher) is required.
 
### Linux

While many parts of the pipeline are OS-agnostic, the _Data Preparation Stage_ relies on subprocesses
that are specific to Linux.
The Rasterization of geospatial data is done with the help of the [`gdal`-library](https://gdal.org/). 
In order to run the _Data Preparation Stage_ the libraries `gdal-bin` and `libgdal-dev` are
required to be installed on your system.

### GPU / CUDA

For GPU accelerated model performance, [`CUDA`](https://developer.nvidia.com/cuda-downloads) and [`cuDNN`](https://developer.nvidia.com/cudnn) are required to be installed on your system.
Using a [docker container](https://hub.docker.com/r/nvidia/cuda/) may be a viable solution.

### RAM

Most of the stages are optimized for low RAM usage via stream processing and file system caches, 
such as Data Loading, Data Preprocessing and Model Training. RAM-usage is configurable via
several parameters (documented below). For flawless execution we recommend at minimum 64 GB of RAM. 

### Disk Space

For raw and processed data about 110 GB of free disk space are needed. When file system caches
are activated (to save RAM on data preprocessing and training) the disk usage will increase with
selected number of wells, selected training period and selected raster size. 


## Installation

install the package via `pip`
```shell
pip install git+https://github.com/calgo-lab/gwl-forecast-pipeline
```

## Configuration / Setup

### Data Sets and Data Access

Most of the used data sets are publicly available for direct download. An exception to this
are the [EU-DEM](https://land.copernicus.eu/imagery-in-situ/eu-dem) data sets that can be downloaded after a free registration. 
The data sets on groundwater levels and groundwater well meta data are not published. In order to gain access, contact the authors of the study.


### Settings

1. create a `settings.ini` file declaring the following config variables:

```ini
[settings]
EUDEM_ELEVATION_URL=...
EUDEM_SLOPE_URL=...
EUDEM_ASPECT_URL=...
GWL_URL=...
DROP_GWL_PERIODS_URL=...
BENCHMARK_RESULTS_URL=...
WELL_META_URL=...
DATA_PATH=...
MODEL_PATH=...
PREPROCESSOR_CACHE_PATH=...
PREDICTION_RESULT_PATH=...
HYPEROPT_RESULT_PATH=...
SCORE_RESULT_PATH=...
```

| VARIABLE                  | SEMANTICS                                                                                           | Required | Default         |
|---------------------------|-----------------------------------------------------------------------------------------------------|----------|-----------------|
| `EUDEM_ELEVATION_URL`     | URL for EU-DEM elevation raw data (see section on data download below)                              | yes      |                 |
| `EUDEM_SLOPE_URL`         | URL for EU-DEM slope raw data (see section on data download below)                                  | yes      |                 |
| `EUDEM_ASPECT_URL`        | URL for EU-DEM aspect raw data (see section on data download below)                                 | yes      |                 |
| `GWL_URL`                 | URL for groundwater level raw time series data, provided by BGR                                     | yes      |                 |
| `DROP_GWL_PERIODS_URL`    | URL for data on detected implausible periods of groundwater level time series' to be removed        | yes      |                 |
| `WELL_META_URL`           | URL for static features and meta data on groundwater wells, provided by BGR                         | yes      |                 |
| `BENCHMARK_RESULTS_URL`   | URL for benchmark wells and scores by [Wunsch et al.](https://doi.org/10.1038/s41467-022-28770-2)   | yes      |                 |
| `HYRAS_HURS_URL`          | URL for HYRAS humidity raw data                                                                     | no       | see `config.py` |
| `HYRAS_TAS_URL`           | URL for HYRAS mean air temperature raw data                                                         | no       | see `config.py` |
| `HYRAS_PR_URL`            | URL for HYRAS precipitation raw data                                                                | no       | see `config.py` |
| `SWR_1000_URL`            | URL for Mean Annual Rate of Percolation data                                                        | no       | see `config.py` |
| `GWN_1000_URL`            | URL for Mean Annual Groundwater Recharge Rate data                                                  | no       | see `config.py` |
| `HUEK_250_URL`            | URL for hydrogeologic overview map data                                                             | no       | see `config.py` |
| `LAI_URL`                 | URL for Copernicus Leaf Area Index raw data                                                         | no       | see `config.py` |
| `CLC_URL`                 | URL for Copernicus Land Cover raw data                                                              | no       | see `config.py` |
| `DATA_PATH`               | path for data storage                                                                               | yes      |                 |
| `MODEL_PATH`              | path for model storage                                                                              | yes      |                 |
| `PREPROCESSOR_CACHE_PATH` | path to store preprocessed data                                                                     | yes      |                 |
| `PREDICTION_RESULT_PATH`  | path to store predictions                                                                           | yes      |                 |
| `HYPEROPT_RESULT_PATH`    | path to store results of hyperparameter optimization                                                | yes      |                 |
| `SCORE_RESULT_PATH`       | path to store scores                                                                                | yes      |                 |
| `CSV_LOGGER_PATH`         | path for csv logging of model training                                                              | no       | '' (None)       |
| `TENSORBOARD_PATH`        | path for tensorboard logs of model training                                                         | no       | '' (None)       |
| `GPU`                     | whether to use a GPU for model training (0 or 1)                                                    | no       | 0 (False)       |
| `DATA_IN_MEMORY`          | whether to keep complete set of preprocessed data in RAM for faster model training (0 or 1)         | no       | 0 (False)       |
| `LOGGING_CONF`            | path to logging configuration file                                                                  | no       | '' (None)       |

2. Set an environment variable pointing to the directory holding the `settings.ini`-file.
```shell
export SETTINGS_PATH=path/to/settings.ini
```

#### Logging

Many package functions log their activities on DEBUG level. Custom Logging can be defined via a [logging config file](https://docs.python.org/3/howto/logging.html#configuring-logging).
The package's parent logger's name is `'gwl_forecast_pipeline'`. Register your logging configuration file in the `LOGGING_CONF` variable in `settings.ini`.
If no logging config file is provided, then all functions log to `stdout` per default.

### Download the raw data

Run the `gwl_download_data`-command in your shell, in order to download and extract the raw data. Make sure to have configured
all download URLs in the `settings.ini` and the `SETTINGS_PATH` environment variable beforehand. Read the section below on how to obtain the EU-DEM data. You will need about **30GB of free disk space** for the raw data. 
The files that are expected to be downloaded and their respective data sets and file sizes are listed in the table below. 
The downloaded files will be stored in `DATA_PATH`.

```shell
gwl_download_data
```

| Dataset            | File                                                   | Size (MB) |
|--------------------|--------------------------------------------------------|-----------|
| HYRAS (HURS)       | hurs_hyras_5_1951_2020_v5-0_de.nc                      | 920       |
| HYRAS (TAS)        | tas_hyras_5_1951_2020_v5-0_de.nc                       | 349       |
| HYRAS (PR)         | pr_hyras_1_1931_2020_v5-0_de.nc                        | 11195     |
| EU-DEM (ELEVATION) | eu_dem_v11_E40N20.TIF                                  | 4906      |
| EU-DEM (ELEVATION) | eu_dem_v11_E40N30.TIF                                  | 4038      |
| EU-DEM (SLOPE)     | EUD_CP-SLOP_4500025000-AA.tif                          | 441       |
| EU-DEM (SLOPE)     | EUD_CP-SLOP_4500035000-AA.tif                          | 116       |
| EU-DEM (ASPECT)    | EUD_CP-ASPC_4500025000-AA.tif                          | 1405      |
| EU-DEM (ASPECT)    | EUD_CP-ASPC_4500035000-AA.tif                          | 1194      |
| HUEK250            | huek250__25832_v103_poly.dbf                           | 58        |
| HUEK250            | huek250__25832_v103_poly.shp                           | 170       |
| GWN1000            | GWN1000__3034_v1_raster1.tif                           | 0.5       |
| SWR1000            | swr1000_250.tif                                        | 13        |
| CLC                | U2018_CLC2012_V2020_20u1.tif                           | 206       |
| LAI                | 12 files (lai_01/w001001.adf, lai_02/w001001.adf, ...) | ~1080     |
| GWL                | groundwater_levels.feather                             | 1394      |
| WELL META          | gwl_germany_features.csv                               | 11        |


#### EU-DEM

In order to access the data from EU-DEM data set, follow these steps:

1. Register on https://land.copernicus.eu
2. Go to https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1-0-and-derived-products
3. Choose the following products and select the following map tiles for download (Download Tab):

    #### EU-DEM Elevation (v1.0) tiles:
   * EU-DEM 45000-35000: `EUD_CP-DEMS_4500035000-AA` 
   * EU-DEM 45000-25000: `EUD_CP-DEMS_4500025000-AA`
    
    #### EU-DEM Slope (v1.0) tiles: 
    * slope 4500035000: `EUD_CP-SLOP_4500035000-AA`
    * slope 4500025000: `EUD_CP-SLOP_4500025000-AA`

    #### EU-DEM Aspect (v1.0) tiles:
    * Aspect 45000-35000: `EUD_CP-ASPC_4500035000-AA`
    * Aspect 45000-25000: `EUD_CP-ASPC_4500025000-AA`
4. For each product you are provided with a download link combining the 2 tiles in one compressed archive. Place the download links into the 
respective variables in the `settings.ini`-file. 

### Prepare the raw data

The data preparation stage harmonizes the raw data to a common format (GeoTiff),
common temporal resolution (1 week), common Coordinate Reference System (EPSG:3034), common spatial resolution (1km x 1km) and common 
spatial and temporal extent. 

To obtain the harmonized data, run the `gwl_prepare_data`-command. This process may take several hours.
The resulting data is stored in `DATA_PATH`.

```shell
gwl_prepare_data
```

## Usage

### Load data

The `raw_data` is loaded in a lazy manner, the result of the `DataLoader.load_data()`-function
is a triple of (meta data, generator for static raster features, generator for temporal raster features)
Use `max_chunk_size=` in order to define the number of samples per iteration in the generator and 
therefore to control the RAM consumption of the data loading process.

```python
import pandas as pd
from gwl_forecast_pipeline import DataLoader

WELL_IDS = ['BB_25470023', 'BB_25470024']
START = pd.Timestamp(2000, 1, 1) 
END = pd.Timestamp(2014, 1, 1)
RASTER_SIZE = 5  # km

data_loader = DataLoader()
raw_data = data_loader.load_data(WELL_IDS, START, END, RASTER_SIZE, max_chunk_size=3500)
```

### Pre-process data

The data preprocessing involves the feature engineering and normalization. The
pre-processed data are `numpy`-arrays. The processed data is divided
into separated arrays: categorical static raster features (int32), numeric static raster features (float32),
groundwater level raster features (float32), exogenous temporal raster features (float32), and the target variable (float32).
The return value of the `preprocess`-function is of a custom type, `DataContainer` which
bundles all arrays and abstracts from the storage. The arrays are either stored in-memory 
or in the file system under `PREPROCESSOR_CACHE_PATH`. This is controlled by the parameter `use_fs-buffer=`.
Once, pre-processed, the data and the fitted preprocessor can be reused. The preprocessor is considered 
a part of the model and therefore is stored under `MODEL_PATH` by the name of the model. The preprocessor requires
a `ModelConfig`-object, which holds information on the raster size and data normalization.


```python
from gwl_forecast_pipeline import (
    Preprocessor,
    CNNModelConfig,
)

model_conf = CNNModelConfig(
   name='my_model',
   raster_size=RASTER_SIZE,
   target_normalized=True,
   scale_per_group=True,
   # ... there are way more params which are not of interest here
)

try:
   preprocessor = Preprocessor.from_cache(model_conf)
except:
   preprocessor = Preprocessor(model_conf)
   
train_data = preprocessor.preprocess(raw_train_data, fit=True, use_fs_buffer=True)
val_data = preprocessor.preprocess(raw_val_data, fit=False, use_fs_buffer=True)
test_data = preprocessor.preprocess(raw_test_data, fit=False, use_fs_buffer=True)

# store fitted preprocessor instance for reuse
preprocessor.store()
```

### Train a new model


Two types of models are available via `CNNModelConfig` or `ConvLSTMModelConfig`. The trained model is stored
in the file system under `MODEL_PATH`. There also exists a `fit_model`-function to train an existing model. 
```python
from gwl_forecast_pipeline import (
    fit_new_model,
    CNNModelConfig,
    ConvLSTMModelConfig,
    FEATURES,
)

model_conf = CNNModelConfig(
    name='my_model',
    lag=4, # number of lag observations in weeks
    lead=1, # length of predicted sequence in weeks
    loss='mse', # name of the loss function, choices are standard TensorFlow losses and custom loss-function: "mean_group_mse"
    epochs=10, 
    batch_size=512, 
    learning_rate=.0001,
    batch_norm=True,
    dropout=.5, # drop out rate in encoder, decoder and dense layers
    n_dense_layers=1, # number of final dense layers after the encoder-decoder
    n_encoder_layers=2, # number of network layers in the encoder
    n_decoder_layers=2, # number of network layers in the decoder
    n_nodes=32, # number of nodes in the first encoder layer, number of nodes in subsequent layers is derived from this number
    dropout_embedding=.2, # dropout rate after the embedding layers
    dropout_static_features=.33, # dropout rate for static features
    dropout_temporal_features=.25, # dropout rate for temporal features
    pre_decoder_dropout=.25, # dropout rate between encoder and decoder
    early_stop_patience=10, # number of epochs for early stopping patience
    weighted_feature=FEATURES.ROCK_TYPE,  # name of a static categorical feature to weigh samples
    sample_weights={0: 1., 1: 1., 2: 1.5, 3: 2., 4: 1.}, # {channel: weight}, channels can be found in constants-module or raw data
    # if weighted_feature and sample_weights are provided, a weighted MSE will be applied as loss-function
)

history = fit_new_model(train_data, model_conf, val_data=val_data)
```

### Make predictions and obtain scores

`predict`-function runs the model inference and returns the predicted values, as well as
the true values indexed by well_id, timestamp and forecast horizon. 
`score`-function evaluates the predictions by NSE, nRMSE and rMBE. 

```python
from gwl_forecast_pipeline import (
    predict,
    score,
)

predictions = predict(model_conf, test_data)
scores = score(predictions)
```

### Hyperparameter Optimization

t.b.d.