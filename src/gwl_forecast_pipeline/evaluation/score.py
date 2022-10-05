import pandas as pd

from .metrics import nse, nrmse, rmbe


def score(predictions):
    metrics = [(nrmse, 'nRMSE'), (nse, 'NSE'), (rmbe, 'rMBE')]
    _metrics = []
    for (well_id, horizon), group in predictions.groupby(
            [predictions.index.get_level_values('proj_id'),
             predictions.index.get_level_values('horizon')]):
        for metric in metrics:
            _df = pd.DataFrame([
                {
                    'metric': metric[1],
                    'value': metric[0](group['y_hat'], group['y']).round(3),
                    'well_id': well_id,
                    'horizon': horizon,
                }
            ])
            _metrics.append(_df)

    metrics_df = pd.concat(_metrics)
    metrics_df = metrics_df.set_index(['well_id', 'horizon', 'metric']).unstack()
    return metrics_df.droplevel(axis=1, level=0)
