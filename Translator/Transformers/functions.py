import tensorflow as tf
import numpy as np
import pandas as pd
import re


def vec_sentence(sentence, vocab, sequence_len):
    vec = np.zeros(shape=(sequence_len,))
    for i, word in enumerate(re.split("\s+", sentence)):
        try:
            vec[i] = vocab[word]
        except:
            vec[i] = vocab["[UNK]"]
    vec = np.reshape(vec, (1, -1))
    return vec


def combine_subwords(subword_list):
    combined_subword = []
    current_word = ""

    for subword in subword_list:
        if subword.startswith("##"):
            current_word += subword[
                2:
            ]  # Remove the "##" prefix and append to the current word
        else:
            if current_word:
                combined_subword.append(current_word)
                current_word = ""
            combined_subword.append(subword)

    if current_word:
        combined_subword.append(current_word)

    return "".join(combined_subword)


def positional_encodings(SEQ_LEN, EMBED_DIMS):
    output = []
    for pos in range(SEQ_LEN):
        position = np.zeros(shape=(EMBED_DIMS))
        for i in range(EMBED_DIMS):
            if i % 2 == 0:
                position[i] = np.sin(pos / (10000 ** (i / EMBED_DIMS)))
            else:
                position[i] = np.cos(pos / (10000 ** ((i - 1) / EMBED_DIMS)))
        output.append(tf.expand_dims(position, axis=0))
    out = tf.concat(output, axis=0)
    out = tf.expand_dims(out, axis=0)
    return tf.cast(out, dtype=tf.float16)


def positional_encoding(model_size, SEQUENCE_LENGTH):
    output = []
    for pos in range(SEQUENCE_LENGTH):
        PE = np.zeros((model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[i] = np.sin(pos / (10000 ** (i / model_size)))
            else:
                PE[i] = np.cos(pos / (10000 ** ((i - 1) / model_size)))
        output.append(tf.expand_dims(PE, axis=0))
    out = tf.concat(output, axis=0)
    out = tf.expand_dims(out, axis=0)
    return tf.cast(out, dtype=tf.float32)


def mapToVocab(vector, vocab):
    vector = tf.squeeze(vector)
    translation = ""
    for i in vector.numpy():
        if i == 0:
            break
        translation += " " + vocab[i]
    return translation


def translate(english_sentence):
    english_sentence = english_sentence.lower()
    english_vector = fn.vec_sentence(english_sentence, e_wtoi, 64)
    translation = ""
    target_left = "START__"
    stop = False
    i = 0
    while not stop:
        target_left_vector = fn.vec_sentence(target_left, h_wtoi, 64)
        output = transformer([english_vector, target_left_vector])
        output_maxxed = tf.argmax(output, axis=-1)
        squeezed_output_maxxed = tf.squeeze(output_maxxed)
        predicted_word_ind = squeezed_output_maxxed[i]
        predicted_word = h_itow[predicted_word_ind.numpy()]
        if predicted_word == "__END" or i == 63:
            stop = True
        translation += " " + predicted_word
        target_left += " " + predicted_word
        i += 1
        # print(target_left_vector)
    translation = rectify(translation.strip())
    return translation


def rectify(translation):
    listed = translation.split(" ")
    listed_rectified = []
    subword_accumulator = []
    j = 1
    i = 0
    # for j in range(1,len(listed)):
    while True:
        if listed[j].startswith("#") and (not listed[i].startswith("#")):
            subword_accumulator.append(listed[i])
            subword_accumulator.append(listed[j])
        elif (not listed[j].startswith("#")) and listed[i].startswith("#"):
            listed_rectified.append(fn.combine_subwords(subword_accumulator))
            subword_accumulator = []
        elif (listed[j].startswith("#")) and listed[i].startswith("#"):
            subword_accumulator.append(listed[j])
        elif (not listed[j].startswith("#")) and (not listed[i].startswith("#")):
            listed_rectified.append(listed[i])
            # listed_rectified.append(listed[j])
        else:
            listed_rectified.append(listed[i])
        if j == len(listed) - 1:
            if subword_accumulator:
                listed_rectified.append(fn.combine_subwords(subword_accumulator))
            break
        j += 1
        i += 1
    return " ".join(listed_rectified)


def make_dataset(dataset, english_vocab, hindi_vocab, len_tokens):
    english_dataset = dataset["English"]
    hindi_dataset = dataset["Hindi"]
    encoder_input = np.zeros(shape=(1, len_tokens))
    decoder_input = np.zeros(shape=(1, len_tokens))
    decoder_output = np.zeros(shape=(1, len_tokens))
    for i, (eng_sentence, hin_sentence) in enumerate(
        zip(english_dataset, hindi_dataset)
    ):
        encoder_input_iter = np.zeros(shape=(1, len_tokens))
        decoder_input_iter = np.zeros(shape=(1, len_tokens))
        decoder_output_iter = np.zeros(shape=(1, len_tokens))
        for j, word in enumerate(re.split("\s+", eng_sentence)):
            try:
                encoder_input_iter[0, j] = english_vocab[word]
            except:
                encoder_input_iter[0, j] = english_vocab["[UNK]"]
        for j, word in enumerate(re.split("\s+", hin_sentence)):
            if word == "__END":
                break
            try:
                decoder_input_iter[0, j] = hindi_vocab[word]
            except:
                decoder_input_iter[0, j] = hindi_vocab["[UNK]"]
        for j, word in enumerate(re.split("\s+", hin_sentence)):
            if j == 0:
                continue
            try:
                decoder_output_iter[0, j - 1] = hindi_vocab[word]
            except:
                decoder_output_iter[0, j - 1] = hindi_vocab["[UNK]"]
        encoder_input = np.vstack((encoder_input, encoder_input_iter))
        decoder_input = np.vstack((decoder_input, decoder_input_iter))
        decoder_output = np.vstack((decoder_output, decoder_output_iter))
        print(f"{i+1}/{len(english_dataset)}", end="\r")
    # encoder_input = encoder_input[1:].reshape((-1,batch_size,len_tokens))
    # decoder_output = decoder_output[1:].reshape((-1,batch_size,len_tokens))
    # decoder_input = decoder_input[1:].reshape((-1,batch_size,len_tokens))
    return encoder_input[1:], decoder_input[1:], decoder_output[1:]
