import logging

from ..constants import (
    CATEGORICAL_STATIC_RASTER_FEATURES,
    CATEGORICAL_STATIC_RASTER_FEATURES_CARDINALITIES,
    GWL_RASTER_FEATURES,
)
from .losses import mean_group_mse, weighted_mse
from ..types import (
    ModelConfig,
    ConvLSTMModelConfig,
    CNNModelConfig,
)

logger = logging.getLogger(__name__)


def load_model(path, model_conf: ModelConfig):
    from tensorflow.keras.models import load_model as _load_model
    if model_conf.loss == 'mean_group_mse':
        return _load_model(path, custom_objects={'mean_group_mse': mean_group_mse})
    else:
        return _load_model(path)


def build_model(conf: ModelConfig):
    import tensorflow.keras.layers as layers
    import tensorflow.keras.losses as losses
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    raster_size_tuple = (conf.raster_size, conf.raster_size)

    # inputs
    lag_features: Model = _build_lag_tensor(conf)
    lead_input = layers.Input(
        shape=(conf.lead, 4, *raster_size_tuple),
        name="lead_input"
    )
    inputs = lag_features.inputs + [lead_input]
    if conf.group_loss:
        well_id_in = layers.Input(shape=(1,), name="well_id_in")
        inputs.append(well_id_in)

    # encoder decoder
    encoder_decoder: Model = _build_encoder_decoder(conf, lag_features, lead_input)
    last = encoder_decoder.output

    # final dense layers
    if last.shape[-2] >= 9:
        last = layers.TimeDistributed(
            layers.MaxPooling2D(pool_size=3, data_format=conf.data_format),
            name="post_decoder_max_pooling",
        )(last)
    last = layers.TimeDistributed(
        layers.Flatten(), name="post_decoder_flatten"
    )(last)
    nodes = conf.n_nodes * 2**(conf.n_encoder_layers-1) // (2**(conf.n_decoder_layers))
    for i in range(conf.n_dense_layers):
        last = layers.TimeDistributed(
            layers.Dense(nodes, activation='relu',
                         kernel_initializer="he_normal"),
            name=f"post_decoder_dense_{i}",
        )(last)
        # last = layers.TimeDistributed(
        #     layers.Dropout(conf.dropout)
        # )(last)
        nodes //= 2
    output = layers.TimeDistributed(
        layers.Dense(1, activation='linear'), name="output_dense"
    )(last)
    output = layers.Reshape(output.shape[1:-1], name="reshape_output")(output)

    # compile model
    loss = conf.loss
    if conf.group_loss:
        output = layers.Concatenate(axis=1)([output, well_id_in])
        loss = mean_group_mse
    if conf.sample_weights and conf.weighted_feature:
        loss = weighted_mse

    model = Model(inputs=inputs, outputs=[output])
    model.compile(loss=loss, optimizer=Adam(learning_rate=conf.learning_rate),
                  metrics=['mse', 'mae'])
    model.summary(print_fn=logger.debug)
    return model


def _build_static_embeddings(conf: ModelConfig):
    import tensorflow.keras.layers as layers
    from tensorflow.keras.models import Model
    raster_size_tuple = (conf.raster_size, conf.raster_size)
    inputs = []
    outputs = []
    for feature in CATEGORICAL_STATIC_RASTER_FEATURES:
        input_ = layers.Input(shape=(1, *raster_size_tuple), name=f"{feature}_in")
        inputs.append(input_)
        flat = layers.Flatten(name=f"flatten_{feature}")(input_)
        cardinality = CATEGORICAL_STATIC_RASTER_FEATURES_CARDINALITIES[feature]
        embedding_dim = max(2, cardinality//4)
        flat_embedding = layers.Embedding(
            cardinality, embedding_dim, input_length=conf.raster_size**2,
            name=f"embedding_{feature}",
        )(flat)
        permuted_flat_embedding = layers.Permute(
            (2, 1), name=f"permute_{feature}",
        )(flat_embedding)
        embedding = layers.Reshape(
            (embedding_dim, *raster_size_tuple), name=f"reshape_{feature}"
        )(permuted_flat_embedding)
        dropout = layers.Dropout(conf.dropout_embedding, name=f"dropout_{feature}")(embedding)
        outputs.append(dropout)

    concat = layers.Concatenate(axis=1)(outputs)
    model = Model(inputs=inputs, outputs=concat)
    return model


def _build_lag_tensor(conf: ModelConfig):
    import tensorflow.keras.layers as layers
    from tensorflow.keras.models import Model

    raster_size_tuple = (conf.raster_size, conf.raster_size)

    # get all static inputs
    embedded_static: Model = _build_static_embeddings(conf)
    static_num_in = layers.Input(
        shape=(6, *raster_size_tuple),
        name="static_numeric_in",
    )

    #  concat all static features
    static_features = layers.Concatenate(
        axis=1, name="concat_static",
    )([embedded_static.output, static_num_in])
    static_dropout = layers.SpatialDropout2D(
        conf.dropout_static_features,
        data_format="channels_first",
    )(static_features)

    # temporal lag features
    temp_feature_lag_in = layers.Input(
        shape=(conf.lag, 4, *raster_size_tuple),
        name="temp_feature_lag_in",
    )
    temp_feature_lag_dropout = layers.TimeDistributed(
        layers.SpatialDropout2D(
            conf.dropout_temporal_features,
            data_format="channels_first",
        ),
        name="temp_feature_lag_dropout",
    )(temp_feature_lag_in)
    gwl_lag_in = layers.Input(
        shape=(conf.lag, len(GWL_RASTER_FEATURES) - (not conf.scale_per_group), *raster_size_tuple),
        name="gwl_lag_in",
    )

    # repeat static data over lag
    flat_static = layers.Flatten(name="flat_static")(static_dropout)
    repeated_flat_static = layers.RepeatVector(conf.lag, name="repeat_flat_static")(flat_static)
    static_lag_features = layers.Reshape(
        (conf.lag, static_features.shape[1], *raster_size_tuple),
        name="reshape_repeated_static"
    )(repeated_flat_static)

    # concat repeated static, temp. lag features and gwl lag
    lag_in = layers.Concatenate(axis=2, name="lag_features")(
        [static_lag_features, temp_feature_lag_dropout, gwl_lag_in]
    )

    model = Model(
        inputs=embedded_static.inputs + [static_num_in, temp_feature_lag_in, gwl_lag_in],
        outputs=lag_in,
    )
    return model


def _build_cnn_encoder_decoder(conf: CNNModelConfig, lag_features, lead_in):
    import tensorflow.keras.layers as layers
    from tensorflow.keras.models import Model

    lead_in: layers.Input = lead_in
    temp_feature_lead_dropout = layers.TimeDistributed(
        layers.SpatialDropout2D(conf.dropout_temporal_features,
                                data_format="channels_first"),
        name="temp_feature_lead_dropout", )(lead_in)
    lag_features: Model = lag_features
    raster_size_tuple = (conf.raster_size, conf.raster_size)

    if conf.data_format == "channels_first":
        concat_axis = 1
        permuted_lag_in = layers.Permute((2, 1, 3, 4))(lag_features.output)
        permuted_lead_in = layers.Permute((2, 1, 3, 4))(temp_feature_lead_dropout)
    else:
        permuted_lag_in = layers.Permute((1, 3, 4, 2))(lag_features.output)
        permuted_lead_in = layers.Permute((1, 3, 4, 2))(temp_feature_lead_dropout)
        concat_axis = 4

    # encoder
    nodes = conf.n_nodes
    last = permuted_lag_in
    for i in range(conf.n_encoder_layers - bool(conf.lag-1)):
        last = layers.Conv3D(
            nodes, kernel_size=(1, 3, 3), kernel_initializer="he_normal", padding="same",
            data_format=conf.data_format
        )(last)
        if conf.batch_norm:
            last = layers.TimeDistributed(layers.BatchNormalization())(last)
        last = layers.ReLU()(last)
        last = layers.TimeDistributed(layers.Dropout(conf.dropout))(last)
        if i < conf.n_encoder_layers - bool(conf.lag-1) - 1:
            # Todo: change logic of nodes per layer
            nodes *= 2

    for _ in range(conf.lag-1):
        last = layers.ZeroPadding3D(padding=(0, 1, 1), data_format=conf.data_format)(last)
        last = layers.Conv3D(nodes, kernel_size=(2, 3, 3),
                             kernel_initializer="he_normal",
                             padding="valid", data_format=conf.data_format)(last)
        if conf.batch_norm:
            last = layers.BatchNormalization()(last)
        last = layers.ReLU()(last)
        last = layers.Dropout(conf.dropout)(last)

    # repeat last encoder output over lead and concat with lead input
    last = layers.Flatten()(last)
    last = layers.RepeatVector(conf.lead)(last)
    if conf.data_format == "channels_first":
        last = layers.Reshape(
            (conf.lead, nodes, *raster_size_tuple)
        )(last)
        last = layers.Permute((2, 1, 3, 4))(last)
    else:
        last = layers.Reshape(
            (conf.lead, *raster_size_tuple, nodes)
        )(last)
    last = layers.Concatenate(axis=concat_axis)([last, permuted_lead_in])
    last = layers.TimeDistributed(
        layers.SpatialDropout2D(conf.pre_decoder_dropout,
                                data_format=conf.data_format)
    )(last)

    # decoder
    lead_layers = False
    for i in range(conf.lead - 1):
        lead_layers = True
        last = layers.Conv3D(nodes, kernel_size=(2, 3, 3),
                             padding="same", kernel_initializer="he_normal",
                             data_format=conf.data_format)(last)
        if conf.batch_norm:
            last = layers.BatchNormalization()(last)
        last = layers.ReLU()(last)
        last = layers.Dropout(conf.dropout)(last)
    nodes //= 2  # Todo: change logic of nodes per layer

    if lead_layers:
        last = layers.MaxPooling3D(
            pool_size=(1, conf.raster_size//5, conf.raster_size//5),
            data_format=conf.data_format,
        )(last)

    for i in range(conf.n_decoder_layers - lead_layers):
        if i == conf.n_decoder_layers - lead_layers - 1:
            padding = 'valid'
        else:
            padding = 'same'
        last = layers.Conv3D(nodes, kernel_size=(1, 3, 3), padding=padding,
                             kernel_initializer="he_normal",
                             data_format=conf.data_format)(last)
        if conf.batch_norm:
            last = layers.TimeDistributed(layers.BatchNormalization())(last)
        last = layers.ReLU()(last)
        last = layers.TimeDistributed(layers.Dropout(conf.dropout))(last)
        if i == 0 and not lead_layers:
            last = layers.MaxPooling3D(
                pool_size=(1, conf.raster_size // 5, conf.raster_size // 5),
                data_format=conf.data_format,
            )(last)
        nodes //= 2

    if conf.data_format == "channels_first":
        last = layers.Permute((2, 1, 3, 4))(last)
    model = Model(inputs=lag_features.inputs + [lead_in], outputs=last)
    return model


def _build_conv_lstm_encoder_decoder(conf: ConvLSTMModelConfig, lag_features, lead_in):
    import tensorflow.keras.layers as layers
    from tensorflow.keras.models import Model

    lead_in: layers.Input = lead_in
    temp_feature_lead_dropout = layers.TimeDistributed(
        layers.SpatialDropout2D(conf.dropout_temporal_features,
            data_format="channels_first"),
        name="temp_feature_lead_dropout",
    )(lead_in)
    lag_features: Model = lag_features
    raster_size_tuple = (conf.raster_size, conf.raster_size)

    if conf.data_format == "channels_first":
        concat_axis = 2
        permuted_lag_in = lag_features.output
        permuted_lead_in = temp_feature_lead_dropout
    else:
        permuted_lag_in = layers.Permute((1, 3, 4, 2))(lag_features.output)
        permuted_lead_in = layers.Permute((1, 3, 4, 2))(temp_feature_lead_dropout)
        concat_axis = 4

    # encoder
    nodes = conf.n_nodes // 2
    last = permuted_lag_in
    for i in range(1, conf.n_encoder_layers + 1):
        nodes *= 2
        if i == conf.n_encoder_layers:
            encoder_last_h1, encoder_last_h2, encoder_last_c = layers.ConvLSTM2D(
                nodes, kernel_size=(3, 3), padding="same",
                data_format=conf.data_format,
                dropout=conf.dropout, recurrent_dropout=conf.recurrent_dropout,
                return_sequences=False, return_state=True,
            )(last)
        else:
            last = layers.ConvLSTM2D(
                nodes, kernel_size=(3, 3), padding="same",
                data_format=conf.data_format,
                dropout=conf.dropout, recurrent_dropout=conf.recurrent_dropout,
                return_sequences=True, return_state=False,
            )(last)

    encoder_last_h1 = layers.BatchNormalization()(encoder_last_h1)
    encoder_last_c = layers.BatchNormalization()(encoder_last_c)

    # repeat last encoder output over lead and concat with lead input
    flat_encoder_last_h1 = layers.Flatten()(encoder_last_h1)
    repeated_flat_encoder_last_h1 = layers.RepeatVector(conf.lead)(flat_encoder_last_h1)
    if conf.data_format == "channels_first":
        repeated_encoder_last_h1 = layers.Reshape(
            (conf.lead, nodes, *raster_size_tuple)
        )(repeated_flat_encoder_last_h1)
    else:
        repeated_encoder_last_h1 = layers.Reshape(
            (conf.lead, *raster_size_tuple, nodes)
        )(repeated_flat_encoder_last_h1)

    decoder_in = layers.Concatenate(axis=concat_axis)(
        [repeated_encoder_last_h1, permuted_lead_in])
    decoder_in = layers.TimeDistributed(
        layers.SpatialDropout2D(
            conf.pre_decoder_dropout, data_format=conf.data_format)
    )(decoder_in)
    # decoder
    last = decoder_in
    for i in range(1, conf.n_decoder_layers + 1):
        if i == 1:
            last = layers.ConvLSTM2D(
                nodes, kernel_size=(3, 3), activation='relu',
                padding="same", return_sequences=True, data_format=conf.data_format,
                dropout=conf.dropout, recurrent_dropout=conf.recurrent_dropout,
            )(last, initial_state=[encoder_last_h1, encoder_last_c])
            last = layers.TimeDistributed(
                layers.MaxPooling2D(
                    pool_size=(conf.raster_size // 5, conf.raster_size // 5),
                    data_format=conf.data_format,
                )
            )(last)
        else:
            if i == conf.n_decoder_layers:
                padding = 'valid'
            else:
                padding = 'same'
            last = layers.ConvLSTM2D(nodes, kernel_size=(3, 3),
                padding=padding, return_sequences=True,
                data_format=conf.data_format, dropout=conf.dropout,
                recurrent_dropout=conf.recurrent_dropout,
            )(last)
        nodes //= 2

    model = Model(inputs=lag_features.inputs + [lead_in], outputs=last)
    return model


def _build_encoder_decoder(conf: ModelConfig, lag_features, lead_in):
    lead_in = lead_in
    if isinstance(conf, ConvLSTMModelConfig):
        return _build_conv_lstm_encoder_decoder(conf, lag_features, lead_in)
    elif isinstance(conf, CNNModelConfig):
        return _build_cnn_encoder_decoder(conf, lag_features, lead_in)
    else:
        raise ValueError(f"unknown model configuration object: {conf}")
