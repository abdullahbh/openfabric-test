# AI Chatbot for Answering Science Questions

This project aims to implement an NLP chatbot using the GPT-Neo language model for answering science-related questions. The implementation is done in Python programming language, and the code is provided in the form of a Python script.

## Requirements

The implementation requires the following libraries to be installed:

* `transformers`: a library for Natural Language Processing tasks, including pre-trained models and fine-tuning tools.
* `openfabric_pysdk`: a library for developing OpenAI Fabric-based applications.
* `time`: a built-in library for time-related functions.

The implementation also assumes that the GPT-Neo-125M model has been downloaded and available for use. The tokenizer and the model are instantiated using the `AutoTokenizer` and `AutoModelForCausalLM` classes provided by the `transformers` library.

## Usage

The chatbot is designed to respond to science-related questions. It uses the `execute` function as a callback function that takes in a `SimpleText` object as input and returns another `SimpleText` object containing the bot's response(s).

When called, the `execute` function processes the input text by tokenizing it using the `tokenizer` object, then generates a response using the `model` object. The response is decoded back to text format and appended to the `output` list.

The `max_length`, `num_return_sequences`, `no_repeat_ngram_size`, and `early_stopping` arguments in the `model.generate()` method determine the length and quality of the generated response.
