# Text Embedding Utility

This Python code provides a simple class `Embedding` to generate both dense and sparse vector representations (embeddings) for text using popular open-source libraries. It leverages the `transformers` library for dense embeddings and `fastembed` for sparse embeddings.

## Features

* Generate dense embeddings for single text strings or lists of strings.
* Generate sparse embeddings for single text strings or lists of strings.
* Easily switch between different pre-trained dense and sparse models by specifying their names.

## Installation

To use this code, you need to have Python installed. Install the required libraries using pip:

```bash
pip install transformers fastembed torch
