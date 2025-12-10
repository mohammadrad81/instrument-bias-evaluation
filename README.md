
# Symphony of Bias

This project is implemented to evaluate AI models on gender bias based on musical instruments.

## Suported models

Currently, we support only open-weight models accessible through huggingface.
The types of supported models for evaluation are: text-to-text, image-text-to-text, and audio-text-to-text generative models.

## Usage
First of all, download the repository. There is a 'requirements.txt' file, install the required libraries using the following command:
```
pip install requirements.txt
```

There is a texts.json file, these are the descriptions that are used to evaluate models. It must be locate in the root directory of the repository, since it is used to evaluate all types of models.

Since we support different types of models, the usage of the project for each type is different.

### LLMs
To evaluate LLMs, go to the text directory:
```
cd text
```

There is a python script there, named "instrument_bias_evaluation_text.py". To use it, you need to give it names and addresses of the models to evaluate.

To do this, you can simply run a command in the following format (in a linux-based system):

```
echo "model_name_1,model_address_1;model_name_2,model_address_2" | python instrument_bias_evaluation_text.py
```

The input that is piped to the script is a list of pairs of model names and addresses. The model's name is separated from its address by comma ',' and each model's information is separated from the other ones by a semicolon ';'.

The output is generated sequentially, and if the execution is stopped (errors or shutdown or anything), simply re-execute it. It continues from where it was.

The outputs of each model are saved in a single json file, whose name is the name of the model (by the way, its '/' is replaced by '__' and '.' is replaced by '_').