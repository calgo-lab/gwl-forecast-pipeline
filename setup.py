from setuptools import find_packages, setup

setup(
    name='gwl_forecast_pipeline',
    packages=[
        'gwl_forecast_pipeline',
        'gwl_forecast_pipeline.data',
        'gwl_forecast_pipeline.features',
        'gwl_forecast_pipeline.features.preprocessing',
        'gwl_forecast_pipeline.logging',
        'gwl_forecast_pipeline.evaluation',
        'gwl_forecast_pipeline.models',
        'gwl_forecast_pipeline.hyperopt',
    ],
    package_dir={
        'gwl_forecast_pipeline': 'src/gwl_forecast_pipeline',
        'gwl_forecast_pipeline.data': 'src/gwl_forecast_pipeline/data',
        'gwl_forecast_pipeline.features': 'src/gwl_forecast_pipeline/features',
        'gwl_forecast_pipeline.features.preprocessing': 'src/gwl_forecast_pipeline/features/preprocessing',
        'gwl_forecast_pipeline.logging': 'src/gwl_forecast_pipeline/logging',
        'gwl_forecast_pipeline.evaluation': 'src/gwl_forecast_pipeline/evaluation',
        'gwl_forecast_pipeline.models': 'src/gwl_forecast_pipeline/models',
        'gwl_forecast_pipeline.hyperopt': 'src/gwl_forecast_pipeline/hyperopt',
    },
    version='0.1.0',
    description='model pipeline for groundwater level prediction in Germany',
    author='Alexander Schulz',
    license='',
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib",
        "scipy",
        "setuptools",
        "pyarrow",
        "pyyaml",
        "pandas",
        "rasterio",
        "tensorflow-gpu",
        "hyperopt",
        "binpacking",
    ]
)
