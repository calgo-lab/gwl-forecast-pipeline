import logging
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
os.environ['HYPEROPT_FMIN_SEED'] = '39657454'
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, JOB_STATE_DONE

from ..models import fit_new_model
from ..types import (ModelConfig, DataContainer, )

logger = logging.getLogger(__name__)

EPSILON = 1e-10


@dataclass
class HyperoptLogger:
    max_evals: int
    logger: logging.Logger
    csv_file: str
    iteration_counter: int = 1

    def log(self, model_config: ModelConfig, loss):
        self.logger.info(f'hyperopt {model_config}; '
                         f'iteration {self.iteration_counter}/{self.max_evals}; '
                         f'loss: {loss:.4f}')

    def log_csv(self, hyper_params, loss):
        df = pd.DataFrame([hyper_params])
        df['loss'] = loss
        df.sort_index(inplace=True, axis=1)
        df.to_csv(self.csv_file, index=False, header=False, mode='a')


def hyperopt(training_data: DataContainer, validation_data: DataContainer,
             model_config: ModelConfig, max_evals=300):
    # Todo: use KerasTuner instead
    hp_space = _hp_space(model_config.type_)
    csv_file = os.path.join(os.path.join(os.environ.get('HYPEROPT_RESULT_PATH'),
                                         f'{model_config.name}.csv'))
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        trials = _restore_trials(df, hp_space)
    else:
        trials = Trials()
        pd.DataFrame(columns=sorted(list(hp_space.keys()) + ['loss'])).to_csv(
            csv_file, index=False, mode='w',
        )

    hyperopt_logger = HyperoptLogger(max_evals, logger, csv_file)
    objective_fn = partial(_objective_fn,
                           training_data=training_data,
                           validation_data=validation_data,
                           model_conf=model_config,
                           hyperopt_logger=hyperopt_logger)
    best = fmin(fn=objective_fn, space=hp_space, algo=tpe.suggest,
                trials=trials, max_evals=max_evals)
    model_config.update(best)
    return model_config


def _restore_trials(df, hp_space):
    trials = Trials()
    for tid, (index, row) in enumerate(df.iterrows()):
        hyperopt_trial = Trials().new_trial_docs(
            tids=[tid],
            specs=[None],
            results=[{'loss': row['loss'], 'status': STATUS_OK}],
            miscs=[{
                'tid': tid, 'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'idxs': {**{key: [tid] for key in hp_space.keys()}},
                'vals': {**{key: [row[key]] for key in hp_space.keys()}},
                'workdir': None
            }]
        )
        hyperopt_trial[0]['state'] = JOB_STATE_DONE
        trials.insert_trial_docs(hyperopt_trial)
        trials.refresh()
    return trials


def _l(x):
    return np.log(x + EPSILON)


def _hp_space(model_type):
    hp_space = {
        'lag': hp.quniform('lag', 2, 5, 1),  # Todo: try longer lags
        'learning_rate': hp.qloguniform('learning_rate', _l(.0001), _l(.005), .0001),
        'n_nodes': hp.qloguniform('n_nodes', _l(32), _l(128), 8),
        'n_encoder_layers': hp.quniform('n_encoder_layers', 1, 3, 1),
        'n_decoder_layers': hp.quniform('n_decoder_layers', 2, 3, 1),
        'n_dense_layers': hp.quniform('n_dense_layers', 1, 3, 1),
        'dropout': hp.quniform('dropout', 0., .5, .1),
        'dropout_embedding': hp.quniform('dropout_embedding', 0., .5, .1),
        'dropout_static_features': hp.quniform('dropout_static_features', 0., .5, .1),
        'dropout_temporal_features': hp.quniform('dropout_temporal_features', 0., .5, .1),
        'pre_decoder_dropout': hp.quniform('pre_decoder_dropout', 0., .5, .1),
        'early_stop_patience': hp.quniform('early_stop_patience',  5, 25, 5),
    }
    if model_type == 'conv_lstm':
        extra_space = {
            'recurrent_dropout': hp.qloguniform('recurrent_dropout', _l(0.), _l(.5), .1),
        }
        hp_space = dict(**hp_space, **extra_space)
    return hp_space


def _objective_fn(hyper_params, training_data: DataContainer=None, validation_data: DataContainer=None,
                  model_conf: ModelConfig = None, hyperopt_logger: HyperoptLogger = None):
    model_conf.update(hyper_params)
    try:
        result = fit_new_model(training_data, model_conf, val_data=validation_data)
        loss = result['history']['val_loss'][-1]
    except:
        loss = np.nan
    hyperopt_logger.log_csv(hyper_params, loss)
    hyperopt_logger.log(model_conf, loss)
    hyperopt_logger.iteration_counter += 1
    return loss
