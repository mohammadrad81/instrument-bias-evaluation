
# Symphony of Bias

This project is implemented to evaluate AI models on gender bias based on musical instruments.

## Suported models

Currently, we support only open-weight models accessible through huggingface.
The types of supported models for evaluation are: text-to-text, image-text-to-text, and audio-text-to-text generative models.

## Usage
First of all, download the repository. There is a 'requirements.txt' file, install the required libraries using the following command:
```
pip install -r requirements.txt
```

There is a texts.json file, these are the descriptions that are used to evaluate models. It must be locate in the root directory of the repository, since it is used to evaluate all types of models.

Since we support different types of models, the usage of the project for each type is different.
To evaluate each type of models (text-to-text, image-text-to-text, audio-text-to-text), enter the following command:
```
python main.py TYPE_OF_MODEL
```
For instance, to evaluate text-to-text models, enter:
```
python main.py text-to-text
```
The output of evaluation of each model, is a json file that stores model's outputs, whose name is the name of the model (by the way, its '/' is replaced by '__' and '.' is replaced by '_').

For each type of model, there is a json config file in a directory named "model-names-and-addresses". For each model, there is an entry "name" and another "address". The name of the model, is the one that is used to download it from huggingface, such as "unsloth/Meta-Llama-3.1-8B-Instruct", and its address is where you want it to be stored and cached.

It is checked that whether the model is found in the given address at first, if it is there, it is loaded from there, and if it is not found, it is downloaded from huggingface and saved there for further use.
