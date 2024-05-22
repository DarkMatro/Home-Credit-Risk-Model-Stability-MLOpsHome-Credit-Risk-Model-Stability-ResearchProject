import keras
import pandas as pd
import numpy as np

from functools import partial
from keras import layers
from keras import ops


class TabTransformer:
    """
    Implementation of Tab Transformer, attention network for tabular data, in Keras

    Parameters
    ----------
    X: pd.DataFrame
        Example frame to create categorical, numerical columns lists,
        cat_features_vocabulary
    cat_features: list[str]
        names of categorical features
    num_features: list[str]
        names of numerical features
    num_transformer_blocks: int
         Number of transformer blocks. Default = 3
    num_heads: int
         Number of attention heads. Default = 4
    embedding_dims: int
        Embedding dimensions of the categorical features. Default=16
    mlp_hidden_units_factors: list | None,
        MLP hidden layer units, as factors of the number of inputs. Default=[2, 1,]
    dropout_rate: float
        Default=0.2
    use_column_embedding=False,
        Add column embedding to categorical feature embeddings.

    Notes
    -----
    Dataset must not contain NaN missing values. All categorical features should be Ordinal encoded.

    Examples
    --------
    >>> clf = TabTransformer(X_train_, cat_features, num_features, embedding_dims=32, num_transformer_blocks=4,
                     num_heads=8, use_column_embedding=True)
    >>> early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1, patience=2, mode='max', restore_best_weights=True)
    >>> clf.fit(X_train_, y_train_, epochs=20, batch_size=256, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])
    >>> y_pred, y_score = model.predict(X_train)
    """

    def __init__(self, X: pd.DataFrame, cat_features: list[str], num_features: list[str], **kwargs) -> None:
        self.proba_threshold = 0.5
        self.batch_size = 128 if 'batch_size' not in kwargs else kwargs['batch_size']
        self.transf_params = None
        self.opt_params = None
        self._create_params(**kwargs)
        self.metrics = [
            keras.metrics.MeanSquaredError(name='Brier score'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.F1Score(name='f1'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),
        ]
        self.optimizer = keras.optimizers.AdamW(**self.opt_params) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.loss = keras.losses.BinaryCrossentropy(reduction="sum_over_batch_size",
                                                    name="binary_crossentropy") if 'loss' not in kwargs else kwargs[
            'loss']
        self.cat_features = X.select_dtypes(
            exclude=np.number).columns.tolist() if cat_features is None else cat_features
        self.num_features = X.select_dtypes(
            include=np.number).columns.tolist() if num_features is None else num_features
        assert len(self.cat_features) > 0, "No categorical features found"
        assert len(self.cat_features) + len(self.num_features) == X.shape[1],\
            "Can't properly detect cat_features and num_features"
        self.cat_features_with_vocabulary = self._get_cat_features_vocabulary(X)
        self.model = self._create_tabtransformer_classifier(**self.transf_params)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def fit(self, x: pd.DataFrame, y: np.ndarray, validation_data=None, **kwargs) -> keras.callbacks.History:
        """
        Trains the self.model for a fixed number of epochs (dataset iterations).

        Parameters
        ----------
        x: Input data. It should be: pd.DataFrame
        y: Target data. Like the input data, should be np.ndarray
        validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. It should be: A tuple (x_val: pd.DataFrame, y_val: np.ndarray)

        """
        x = [x[col] for col in x.columns]
        y = np.array(y).reshape(-1, 1)
        if validation_data is not None:
            validation_data = ([validation_data[0][col] for col in validation_data[0].columns],
                               np.array(validation_data[1]).reshape(-1, 1))
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        return self.model.fit(x, y, validation_data=validation_data, batch_size=self.batch_size, **kwargs)

    def predict_proba(self, x: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Estimate probability.

        Parameters
        ----------
        x: pd.DataFrame
            Input data.
        Returns
        -------
        y_score: np.ndarray
            Estimated probabilities.
        """
        if 'batch_size' in kwargs:
            batch_size = kwargs.pop('batch_size')
        else:
            batch_size = self.batch_size
        return self.model.predict([x[col] for col in x.columns], batch_size=batch_size, **kwargs)

    def predict(self, x: pd.DataFrame, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels for samples in x and estimate probability

        Parameters
        ----------
        x: pd.DataFrame
            The dataframe for which we want to get the predictions.
        Returns
        -------
        y_pred: np.ndarray
            Vector containing the class labels for each sample.
        y_score: np.ndarray
            Estimated probabilities.
        """
        y_score = self.predict_proba(x, **kwargs)
        return (y_score >= self.proba_threshold).astype(int), y_score

    def _create_params(self, **kwargs) -> None:
        """
        Generate default parameters of Transformer. And replace default values by user's values in kwargs.

        Parameters
        ----------
        **kwargs parameters.

        Returns
        -------
        None
        """
        self.transf_params = {'num_transformer_blocks': 3, 'num_heads': 4, 'embedding_dims': 16,
                              'mlp_hidden_units_factors': [2, 1],
                              'dropout_rate': 0.2, 'use_column_embedding': False}
        self.opt_params = {'learning_rate': 0.001, 'weight_decay': 0.004}
        for k, v in kwargs.items():
            if k in self.transf_params:
                self.transf_params[k] = v
            if k in self.opt_params:
                self.opt_params[k] = v

    def _get_cat_features_vocabulary(self, X: pd.DataFrame) -> dict:
        """
        Create cat_features_vocabulary which using in encode_inputs function.

        Parameters
        ----------
        X: pd.DataFrame
            Example frame to create cat_features_vocabulary
        Returns
        -------
        voc: dict
        """
        voc = {}
        for col in self.cat_features:
            voc[col] = sorted(list(X[col].unique()))
        return voc

    def _create_model_inputs(self) -> dict:
        """
        Create inputs for model.

        Returns
        -------
        inputs: dict
        """
        inputs = {}
        for feature_name in self.num_features:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
        for feature_name in self.cat_features:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="int32"
            )
        return inputs

    def _encode_inputs(self, inputs: dict, embedding_dims: int = 16) -> tuple[list, list]:
        """
        Encode features

        Parameters
        ----------
        inputs: dict
            given from self.create_model_inputs
        embedding_dims: int
            Default=16

        Returns
        -------
        out: tuple[list, list]
        """
        encoded_categorical_feature_list = []
        numerical_feature_list = []
        for feature_name, vocabulary in self.cat_features_with_vocabulary.items():
            embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_dims)
            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(inputs[feature_name])
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        for feature_name in self.num_features:
            numerical_feature = ops.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

        return encoded_categorical_feature_list, numerical_feature_list

    def _create_tabtransformer_classifier(self, mlp_hidden_units_factors=None, num_transformer_blocks: int = 3,
                                          num_heads: int = 4, embedding_dims: int = 16, dropout_rate: float = 0.2,
                                          use_column_embedding=False) -> keras.Model:
        """
        Create TabTransformer

        Parameters
        ----------
        num_transformer_blocks: int
             Number of transformer blocks. Default = 3
        num_heads: int
             Number of attention heads. Default = 4
        embedding_dims: int
            Embedding dimensions of the categorical features. Default=16
        mlp_hidden_units_factors: list | None,
            MLP hidden layer units, as factors of the number of inputs. Default=[2, 1,]
        dropout_rate: float
            Default=0.2
        use_column_embedding=False,
            Add column embedding to categorical feature embeddings.

        Returns
        -------
        model: keras.Model
        """
        if mlp_hidden_units_factors is None:
            mlp_hidden_units_factors = [2, 1, ]

        # Create model inputs.
        inputs = self._create_model_inputs()
        # encode features.
        encoded_categorical_feature_list, numerical_feature_list = self._encode_inputs(inputs, embedding_dims)
        # Stack categorical feature embeddings for the Transformer.
        encoded_categorical_features = ops.stack(encoded_categorical_feature_list, axis=1)
        # Concatenate numerical features.
        numerical_features = layers.concatenate(numerical_feature_list)

        # Add column embedding to categorical feature embeddings.
        if use_column_embedding:
            num_columns = encoded_categorical_features.shape[1]
            column_embedding = layers.Embedding(input_dim=num_columns, output_dim=embedding_dims)
            column_indices = ops.arange(start=0, stop=num_columns, step=1)
            encoded_categorical_features = encoded_categorical_features + column_embedding(column_indices)

        # Create multiple layers of the Transformer block.
        for block_idx in range(num_transformer_blocks):
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dims,
                dropout=dropout_rate,
                name=f"multihead_attention_{block_idx}",
            )(encoded_categorical_features, encoded_categorical_features)
            # Skip connection 1.
            x = layers.Add(name=f"skip_connection1_{block_idx}")([attention_output, encoded_categorical_features])
            # Layer normalization 1.
            x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
            # Feedforward.
            feedforward_output = create_mlp(hidden_units=[embedding_dims], activation=keras.activations.gelu,
                normalization_layer=partial(layers.LayerNormalization, epsilon=1e-6),
                                            # using partial to provide keyword arguments before initialization
                dropout_rate=dropout_rate, name=f"feedforward_{block_idx}",)(x)
            # Skip connection 2.
            x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
            # Layer normalization 2.
            encoded_categorical_features = layers.LayerNormalization(name=f"layer_norm2_{block_idx}", epsilon=1e-6)(x)

        # Flatten the "contextualized" embeddings of the categorical features.
        categorical_features = layers.Flatten()(encoded_categorical_features)
        # Apply layer normalization to the numerical features.
        numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
        # Prepare the input for the final MLP block.
        features = layers.concatenate([categorical_features, numerical_features])
        # Compute MLP hidden_units.
        mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
        # Create final MLP.
        mlp = create_mlp(hidden_units=mlp_hidden_units, activation=keras.activations.selu,
                         normalization_layer=layers.BatchNormalization, dropout_rate=dropout_rate, name="MLP")
        features = mlp(features)

        # Add a sigmoid as a binary classifer.
        outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


def create_mlp(hidden_units: list, activation: keras.activations, normalization_layer: layers,
               dropout_rate: float = 0.2, name: str | None = None) -> keras.Sequential:
    """
    MLP as Feedforward and final MLP

    Parameters
    ----------
    hidden_units: list
    activation: keras.activations
    normalization_layer: layers
    name: str | None
    dropout_rate: float
        Default=0.2

    Returns
    -------
    out: keras.Sequential
    """
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer()),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)
