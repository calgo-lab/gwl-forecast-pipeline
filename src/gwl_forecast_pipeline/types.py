import abc
import hashlib
import os
from dataclasses import dataclass, field
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
from ray import tune

from .constants import (
    GWL_RASTER_FEATURES,
    DEFAULT_PREPROCESSOR_CACHE_FILES,
    TARGET_COL,
)
from . import config as config


class Updateable:
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


@pydantic_dataclass(config=dict(validate_assignment=True))
class ModelConfig(Updateable, metaclass=abc.ABCMeta):
    name: str
    lag: int = 4
    lead: int = 1
    raster_size: int = 5
    target_variable: str = TARGET_COL
    target_normalized: bool = True
    scale_per_group: bool = True
    loss: str = 'mse'
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = .0001
    batch_norm: bool = True
    dropout: float = .5
    n_dense_layers: int = 1
    n_encoder_layers: int = 1
    n_decoder_layers: int = 2
    n_nodes: int = 32
    dropout_embedding: float = .2
    dropout_static_features: float = .33
    dropout_temporal_features: float = .25
    pre_decoder_dropout: float = .25
    early_stop_patience: int = 20
    weighted_feature: str = None
    sample_weights: dict = None
    type_: str = None

    @property
    def group_loss(self) -> bool:
        return self.loss == "mean_group_mse"

    @property
    def data_format(self) -> str:
        if config.GPU:
            return 'channels_first'
        else:
            return 'channels_last'


@pydantic_dataclass(config=dict(validate_assignment=True))
class ConvLSTMModelConfig(ModelConfig):
    epochs: int = 100
    recurrent_dropout: float = .1
    type_: str = 'conv_lstm'


@pydantic_dataclass(config=dict(validate_assignment=True))
class CNNModelConfig(ModelConfig):
    epochs: int = 200
    type_: str = 'cnn'


@dataclass
class ModelHpSpace:
    lag = field(default_factory=lambda: tune.quniform(2, 26, 1))
    learning_rate = field(default_factory=lambda: tune.qloguniform(.0001, .005, .0001))
    n_nodes = field(default_factory=lambda: tune.qloguniform(32, 128, 8))
    n_encoder_layers = field(default_factory=lambda: tune.quniform(1, 3, 1))
    n_decoder_layers = field(default_factory=lambda: tune.quniform(2, 3, 1))
    n_dense_layers = field(default_factory=lambda: tune.quniform(1, 3, 1))
    dropout = field(default_factory=lambda: tune.quniform(0., .5, .1))
    dropout_embedding = field(default_factory=lambda: tune.quniform(0., .5, .1))
    dropout_static_features = field(default_factory=lambda: tune.quniform(0., .5, .1))
    dropout_temporal_features = field(default_factory=lambda: tune.quniform(0., .5, .1))
    pre_decoder_dropout = field(default_factory=lambda: tune.quniform(0., .5, .1))


@dataclass
class ConvLSTMModelHpSpace(ModelHpSpace):
    recurrent_dropout = field(default_factory=lambda: tune.quniform(0., .5, .1))


@dataclass
class MemmapWrapper:
    file: Union[Path, str]
    shape: Tuple[int, ...]
    dtype: type = np.float32

    def get(self, mode="r"):
        memmap = np.memmap(self.file, dtype=self.dtype, shape=self.shape, mode=mode)
        return memmap

    def create(self):
        np.memmap(self.file, dtype=self.dtype, shape=self.shape, mode="w+")

    def write(self, data: np.ndarray, offset=0):
        memmap = np.memmap(self.file, dtype=self.dtype, shape=self.shape, mode="r+")
        memmap[offset:offset+data.shape[0]] = data
        memmap.flush()


@dataclass
class MetaData:
    well_ids: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    raster_size: int
    n_wells: int
    n_samples: int

    def get_hash(self):
        hash_input = f'{"".join(sorted(self.well_ids))}' \
                     f'{self.start}' \
                     f'{self.end}' \
                     f'{self.raster_size}'
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()


@dataclass
class DataContainer:
    static_index: Union[pd.DataFrame, Path, str]
    numeric_static_raster: Union[np.ndarray, np.memmap, dict, MemmapWrapper]
    categorical_static_raster: Union[np.ndarray, np.memmap, dict, MemmapWrapper]
    temporal_index: Union[pd.DataFrame, Path, str]
    gwl_raster: Union[np.ndarray, np.memmap, dict, MemmapWrapper]
    temporal_feature_raster: Union[np.ndarray, np.memmap, dict, MemmapWrapper]
    target: Optional[Union[np.ndarray, np.memmap, dict, MemmapWrapper]] = None

    @property
    def static_fields(self):
        return ['static_index', 'numeric_static_raster', 'categorical_static_raster']

    @property
    def temporal_fields(self):
        return ['temporal_index', 'gwl_raster', 'temporal_feature_raster', 'target']

    def collect(self, dry_run=False):
        error_fields = []
        for _field in self.__dataclass_fields__.keys():
            value = getattr(self, _field)
            if _field == 'static_index':
                if isinstance(value, (Path, str)):
                    try:
                        if dry_run:
                            f = open(self.static_index, mode="r")
                            f.close()
                        else:
                            self.static_index = pd.read_csv(self.static_index)
                    except FileNotFoundError:
                        error_fields.append(_field)
            elif _field == 'temporal_index':
                if isinstance(value, (Path, str)):
                    try:
                        if dry_run:
                            f = open(self.temporal_index, mode="r")
                            f.close()
                        else:
                            self.temporal_index = pd.read_csv(self.temporal_index,
                                                              parse_dates=['time'])
                    except FileNotFoundError:
                        error_fields.append(_field)
            else:
                if isinstance(value, (dict, DictProxy)):
                    setattr(self, _field, MemmapWrapper(**value))
                    value = getattr(self, _field)
                if isinstance(value, MemmapWrapper):
                    try:
                        if dry_run:
                            f = open(value.file, mode="r")
                            f.close()
                        else:
                            setattr(self, _field, value.get())
                    except FileNotFoundError:
                        error_fields.append(_field)
        return error_fields

    def load(self):
        self.collect()
        for _field in ['numeric_static_raster', 'categorical_static_raster', 'gwl_raster',
                       'temporal_feature_raster', 'target']:
            setattr(self, _field, np.array(getattr(self, _field)))

    @staticmethod
    def from_path(path, meta: MetaData):
        config_hash = meta.get_hash()
        raster_size_tuple = (meta.raster_size, meta.raster_size)
        files = DEFAULT_PREPROCESSOR_CACHE_FILES
        return DataContainer(
            static_index=os.path.join(path, f'{config_hash}_{files["static_index"]}'),
            numeric_static_raster=MemmapWrapper(
                os.path.join(path,  f'{config_hash}_{files["numeric_static_raster"]}'),
                shape=(meta.n_wells, 6, *raster_size_tuple),
            ),
            categorical_static_raster=MemmapWrapper(
                os.path.join(path, f'{config_hash}_{files["categorical_static_raster"]}'),
                shape=(meta.n_wells, 5, *raster_size_tuple),
                dtype=np.int32
            ),
            temporal_index=os.path.join(path, f'{config_hash}_{files["temporal_index"]}'),
            gwl_raster=MemmapWrapper(
                os.path.join(path, f'{config_hash}_{files["gwl_raster"]}'),
                shape=(meta.n_samples, len(GWL_RASTER_FEATURES), *raster_size_tuple),
            ),
            temporal_feature_raster=MemmapWrapper(
                os.path.join(path, f'{config_hash}_{files["temporal_feature_raster"]}'),
                shape=(meta.n_samples, 4, *raster_size_tuple),
            ),
            target=MemmapWrapper(
                os.path.join(path, f'{config_hash}_{files["target"]}'),
                shape=(meta.n_samples, 1),
            ),
        )
