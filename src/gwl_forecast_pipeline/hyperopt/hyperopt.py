import logging
import os
from dataclasses import asdict
from typing import Dict

from ray import air, tune
from ray.tune import Tuner
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.utils import wait_for_gpu

from .. import config as config_
from ..models import fit_model, build_model
from ..models.batch_generator import create_batch_generator
from ..types import (ModelConfig, DataContainer, ModelHpSpace, CNNModelConfig,
                     ConvLSTMModelConfig)

logger = logging.getLogger(__name__)


def hyperopt(training_data: DataContainer, validation_data: DataContainer,
             model_config: ModelConfig, param_space: ModelHpSpace, max_evals=300, resume=False):
    tuner = None
    if resume:
        try:
            tuner = Tuner.restore(path=os.path.join(config_.HYPEROPT_RESULT_PATH, model_config.name))
        except RuntimeError:
            logger.warning(f"could not find a hyperparameter tuning session for {model_config.name}"
                           f"in {config_.HYPEROPT_RESULT_PATH}. start a new session instead.")
    if not tuner:
        run_conf = air.RunConfig(
            name=model_config.name,
            local_dir=config_.HYPEROPT_RESULT_PATH,
        )
        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=model_config.epochs,
        )
        bohb_search = TuneBOHB(
            metric="val_loss",
            mode="min",
        )
        tune_conf = tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=max_evals,
            max_concurrent_trials=1,
        )

        tuner = tune.Tuner(
            tune.with_parameters(
                tune.with_resources(
                    _objective_fn,
                    {'cpu': 0, 'gpu': 1 if config_.GPU else 0}
                ),
                training_data=asdict(training_data),
                validation_data=asdict(validation_data),
                model_conf=asdict(model_config)
            ),
            tune_config=tune_conf,
            run_config=run_conf,
            param_space=asdict(param_space)
        )
    results = tuner.fit()
    model_config.update(results.get_best_result().config)
    return model_config


def _objective_fn(hyper_params, training_data: Dict, validation_data: Dict,
                  model_conf: Dict = None):
    # if config_.GPU:
    #     wait_for_gpu()
    if model_conf['type_'] == 'cnn':
        model_conf = CNNModelConfig(**model_conf)
    elif model_conf['type_'] == 'conv_lstm':
        model_conf = ConvLSTMModelConfig(**model_conf)
    else:
        raise ValueError("unknown model type")
    model_conf.update(hyper_params)

    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    model = build_model(model_conf)
    train_data = create_batch_generator(DataContainer(**training_data), model_conf)
    val_data = create_batch_generator(DataContainer(**validation_data), model_conf)

    fit_model(model, train_data, model_conf, val_data=val_data, tune_callback=True)
