# Language Translator - RNN vs Transformer Architectures
This repository explores two approaches to building a language translator between English and Hindi:

## 1. RNN-based Sequence-to-Sequence Model (RNN folder):

This folder implements a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers for sequence-to-sequence translation. The model learns to translate English sentences into Hindi words based on a predefined vocabulary.

### Folder Structure:

main.ipynb: Contains the complete implementation, including defining and training the encoder-decoder model, data preprocessing, and training loop. Refer to the code comments for detailed guidance.

model.h5: A pre-trained model saved in H5 format, ready to be loaded and used for translation tasks.

## 2. Transformer-based Language Translation (Transformer implementation):

This folder implements the state-of-the-art Transformer architecture for English-to-Hindi translation, built from scratch based on the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762). Unlike RNNs, Transformers rely solely on attention mechanisms to learn relationships between source and target languages.

### Folder Structure:

main.ipynb: Contains the training loop, data preprocessing functions, and functionalities for testing different input sentences.

classes.py: Implements the core Transformer encoder and decoder structures, including custom embedding layers.

functions.py: Contains functions for data preprocessing, data transformation, and interpreting the translated output.

Vocab folder: Stores the subword-tokenized vocabulary used by the Transformer model, generated using the BERT tokenization algorithm.

### Getting Started:

1. Clone the repository.
2. Install required libraries: Ensure you have libraries like TensorFlow, Keras (or PyTorch for Transformer) and any additional dependencies mentioned in the Jupyter Notebooks.
3. Run the experiments: Open the respective main.ipynb files (RNN or Transformer) in a Jupyter Notebook environment and execute the cells to train the models and explore translation capabilities. Refer to the comments within the notebooks for specific instructions.

### Pre-trained Model (RNN):

The model.h5 file provides a pre-trained RNN model you can use for basic translation tasks.

## Disclaimer:

The translation quality might be limited for both models due to the complexity of language translation and potentially limited training data.
