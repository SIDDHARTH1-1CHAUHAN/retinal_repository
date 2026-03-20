from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

STAGE1_LABELS = ("normal", "dr", "hr")


def get_stage1_label_order() -> list[str]:
    return list(STAGE1_LABELS)


def get_stage1_custom_objects() -> dict[str, object]:
    return {
        "PatchEmbedding": PatchEmbedding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "MacroPrecision": MacroPrecision,
        "MacroRecall": MacroRecall,
        "MacroF1Score": MacroF1Score,
    }


@keras.utils.register_keras_serializable(package="stage1_vit")
class PatchEmbedding(layers.Layer):
    def __init__(
        self,
        image_size: Sequence[int] = (224, 224),
        patch_size: int = 16,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_patches = (self.image_size[0] // self.patch_size) * (
            self.image_size[1] // self.patch_size
        )
        self.projection = layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="patch_projection",
        )
        self.dropout = layers.Dropout(self.dropout_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches + 1, self.hidden_size),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        x = self.projection(inputs)
        x = tf.reshape(x, [batch_size, self.num_patches, self.hidden_size])
        class_token = tf.broadcast_to(
            self.class_token, [batch_size, 1, self.hidden_size]
        )
        x = tf.concat([class_token, x], axis=1)
        x = x + self.position_embeddings
        return self.dropout(x, training=training)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
        }


@keras.utils.register_keras_serializable(package="stage1_vit")
class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads,
            dropout=self.attention_dropout_rate,
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential(
            [
                layers.Dense(self.mlp_dim, activation=tf.nn.gelu),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.hidden_size),
                layers.Dropout(self.dropout_rate),
            ],
            name="mlp",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.norm_1(inputs)
        attention_output = self.attention(x, x, training=training)
        x = inputs + self.dropout_1(attention_output, training=training)
        y = self.norm_2(x)
        return x + self.mlp(y, training=training)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
        }


class MacroMetricBase(keras.metrics.Metric):
    def __init__(self, num_classes: int, name: str, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="true_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(self.num_classes,), initializer="zeros"
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        y_true_indices = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred_indices = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
        confusion = tf.cast(
            tf.math.confusion_matrix(
                y_true_indices,
                y_pred_indices,
                num_classes=self.num_classes,
                weights=weights,
            ),
            self.dtype,
        )
        tp = tf.linalg.diag_part(confusion)
        fp = tf.reduce_sum(confusion, axis=0) - tp
        fn = tf.reduce_sum(confusion, axis=1) - tp
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def reset_state(self) -> None:
        for variable in (self.true_positives, self.false_positives, self.false_negatives):
            variable.assign(tf.zeros_like(variable))

    def get_config(self) -> dict:
        return {**super().get_config(), "num_classes": self.num_classes}


@keras.utils.register_keras_serializable(package="stage1_vit")
class MacroPrecision(MacroMetricBase):
    def __init__(self, num_classes: int, name: str = "precision", **kwargs) -> None:
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self) -> tf.Tensor:
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        return tf.reduce_mean(precision)


@keras.utils.register_keras_serializable(package="stage1_vit")
class MacroRecall(MacroMetricBase):
    def __init__(self, num_classes: int, name: str = "recall", **kwargs) -> None:
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self) -> tf.Tensor:
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        return tf.reduce_mean(recall)


@keras.utils.register_keras_serializable(package="stage1_vit")
class MacroF1Score(MacroMetricBase):
    def __init__(self, num_classes: int, name: str = "f1_score", **kwargs) -> None:
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self) -> tf.Tensor:
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)


def _as_image_size(image_size: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in image_size)
    if len(values) != 2:
        raise ValueError(f"image_size must contain exactly two values, got {values}")
    return values


def build_stage1_vit(
    input_shape: Sequence[int] = (224, 224, 3),
    image_size: Sequence[int] = (224, 224),
    patch_size: int = 16,
    hidden_size: int = 768,
    transformer_layers: int = 12,
    num_heads: int = 12,
    mlp_dim: int = 3072,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    num_classes: int = 3,
    classifier: str = "token",
    model_name: str = "stage1_vit_b16",
) -> keras.Model:
    if classifier not in {"token", "gap"}:
        raise ValueError("classifier must be either 'token' or 'gap'")

    image_size = _as_image_size(image_size)
    inputs = keras.Input(shape=tuple(input_shape), name="image")
    x = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        name="patch_embedding",
    )(inputs)

    for index in range(transformer_layers):
        x = TransformerEncoderBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name=f"encoder_block_{index + 1}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
    if classifier == "token":
        x = layers.Lambda(lambda tensor: tensor[:, 0], name="cls_token")(x)
    else:
        x = layers.Lambda(
            lambda tensor: tf.reduce_mean(tensor[:, 1:, :], axis=1),
            name="global_average_pool",
        )(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_probabilities",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def compile_stage1_model(
    model: keras.Model,
    learning_rate: float,
    num_classes: int = len(STAGE1_LABELS),
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            MacroPrecision(num_classes=num_classes, name="precision"),
            MacroRecall(num_classes=num_classes, name="recall"),
            MacroF1Score(num_classes=num_classes, name="f1_score"),
        ],
    )
    return model

