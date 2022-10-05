import logging.config
import os
from yaml import load, Loader


def setup_logging(filename, base_path=None):
    if base_path:
        os.chdir(base_path)
    with open("src/gwl_forecast_pipeline/logging/logging_conf.yaml") as f:
        config = load(f.read(), Loader=Loader)
        config['handlers']['file']['filename'] = filename
    logging.config.dictConfig(config)
