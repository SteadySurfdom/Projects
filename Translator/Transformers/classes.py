import tensorflow as tf
import numpy as np
import pandas as pd
import functions as fn
import re
from tensorflow.keras.layers import *


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, output_dims, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, output_dims)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.output_dim = output_dims

    def call(self, inputs):
        embedding_tokens = self.token_embeddings(inputs)
        pos_encodings = fn.positional_encodings(self.sequence_length, self.output_dim)
        final_output = embedding_tokens + pos_encodings
        return final_output

    def compute_mask(self, inputs, mask=None):
        mask = tf.math.not_equal(inputs, tf.constant(0.0, dtype=tf.float16))
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "output_dims": self.output_dim,
                "sequence_length": self.sequence_length,
            }
        )
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, dense_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feedforward = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(dense_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask1 = mask[:, :, tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            mask = mask1 & mask2
            padding_mask = tf.cast(mask, tf.int32)
            attention_scores = self.attention(
                query=inputs, key=inputs, value=inputs, attention_mask=padding_mask
            )
        else:
            attention_scores = self.attention(query=inputs, key=inputs, value=inputs)

        normalised_attention = self.layer_norm_1(attention_scores + inputs)
        # normalised_attention = attention_scores + inputs
        dense_output = self.feedforward(normalised_attention)
        normalised_dense = self.layer_norm_2(dense_output + normalised_attention)
        # normalised_dense = dense_output + normalised_attention
        return normalised_dense

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, dense_dim, num_heads, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_masked = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.layer_norm_3 = tf.keras.layers.LayerNormalization()
        self.feedforward = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.supports_masking = True

    def call(self, embedding_inputs, encoder_outputs, mask=None):
        if mask is not None:
            mask1 = mask[:, :, tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            mask = mask1 & mask2
            padding_mask = tf.cast(mask, tf.int32)
            masked_attention = self.attention_masked(
                query=embedding_inputs,
                value=embedding_inputs,
                key=embedding_inputs,
                attention_mask=padding_mask,
                use_causal_mask=True,
            )
        else:
            masked_attention = self.attention_masked(
                query=embedding_inputs,
                value=embedding_inputs,
                key=embedding_inputs,
                use_causal_mask=True,
            )
        normalised_attention = self.layer_norm_1(embedding_inputs + masked_attention)
        # normalised_attention = embedding_inputs + masked_attention

        if mask is not None:
            mask1 = mask[:, :, tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            mask = mask1 & mask2
            mask = tf.cast(mask, tf.int32)
            encoder_attention = self.attention(
                query=normalised_attention,
                value=encoder_outputs,
                key=encoder_outputs,
                attention_mask=padding_mask,
                use_causal_mask=True,
            )
        else:
            encoder_attention = self.attention(
                query=normalised_attention, value=encoder_outputs, key=encoder_outputs
            )

        normalised_encoder = self.layer_norm_2(normalised_attention + encoder_attention)
        # normalised_encoder = normalised_attention + encoder_attention
        dense_outputs = self.feedforward(normalised_encoder)
        normalised_dense = self.layer_norm_3(dense_outputs + normalised_encoder)
        # normalised_dense = dense_outputs + normalised_encoder

        return normalised_dense

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class Embeddings2(Layer):
    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim,
    ):
        super().__init__()
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = fn.positional_encoding(
            self.embed_dim, self.sequence_length
        )
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerEncoder2(Layer):
    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.dense_proj = tf.keras.Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask1 = mask[:, :, tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            padding_mask = tf.cast(mask1 & mask2, dtype="int32")

        attention_output = self.attention(
            query=inputs, key=inputs, value=inputs, attention_mask=padding_mask
        )

        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class TransformerDecoder2(Layer):
    def __init__(
        self,
        embed_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [
                Dense(latent_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        if mask is not None:
            mask1 = mask[:, :, tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            padding_mask = tf.cast(mask1 & mask2, dtype="int32")
            causal_mask = tf.linalg.band_part(
                tf.ones(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[1]],
                    dtype=tf.int32,
                ),
                -1,
                0,
            )
            combined_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=causal_mask,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2, scores = self.attention_2(
            query=out_1,
            key=encoder_outputs,
            value=encoder_outputs,
            attention_mask=combined_mask,
            return_attention_scores=True,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output), scores


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_encoders,
        num_decoders,
        embed_dims,
        seq_len,
        dense_dim,
        len_engish_vocab,
        len_hindi_vocab,
        num_heads,
    ):
        super().__init__()
        self.english_embed = Embeddings(
            vocab_size=len_engish_vocab, output_dims=embed_dims, sequence_length=seq_len
        )
        self.hindi_embed = Embeddings(
            vocab_size=len_hindi_vocab, output_dims=embed_dims, sequence_length=seq_len
        )
        self.output_dense = tf.keras.layers.Dense(
            units=len_hindi_vocab, activation="softmax"
        )
        self.num_encoders = num_encoders
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.num_decoders = num_decoders
        self.encoder_dict = {}
        for i in range(self.num_encoders):
            self.encoder_dict[f"Encoder_{i}"] = TransformerEncoder(
                num_heads=self.num_heads,
                embed_dim=self.embed_dims,
                dense_dim=self.dense_dim,
            )
        self.decoder_dict = {}
        for i in range(self.num_decoders):
            self.decoder_dict[f"Decoder_{i}"] = TransformerDecoder(
                dense_dim=self.dense_dim,
                num_heads=self.num_heads,
                embed_dim=self.embed_dims,
            )

    def call(self, inputs, training=False):
        english_inputs = inputs[0]
        hindi_inputs = inputs[1]
        x = self.english_embed(english_inputs)
        for _, encoder in self.encoder_dict.items():
            x = encoder(x)
        y = self.hindi_embed(hindi_inputs)
        for _, decoder in self.decoder_dict.items():
            y = decoder(y, x)
        return self.output_dense(y)
