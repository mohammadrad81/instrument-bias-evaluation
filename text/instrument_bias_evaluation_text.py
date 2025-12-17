# print("importing libraries...")
from datasets import Dataset
from transformers import pipeline, Pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
import pandas as pd
import datetime as dt
import os
import time
import json
import random
import gc
from utils import (
    get_model_names_and_addresses,
    print_input_model_names_and_addresses,
    load_texts_list,
    load_pipeline,
    template_to_complete_prompt,
    GENDERS,
    CATEGORIES,
    WITH_REASON_PROMPT_TEMPLATE,
    WITHOUT_REASON_PROMPT_TEMPLATE,
)

# print("import completed.")



def evaluate_llms(model_names: list[str], max_new_tokens: int=100):
    """Generates the llms outputs sequentially

    Args:
        model_names (list[str]): a list of model names given to program by prompt
    """
    question = "What is the gender of the person?"
    texts = load_texts_list()
    
    # for model_data in model_names_and_addresses_list:
    #     model_name = model_data["name"]
    #     model_address = model_data["address"]
    for model_name in model_names:
        pipe = load_pipeline(model_name)
        output_dataset_path = model_name.replace("/", "__").replace(".", "_") + ".json"
        data_texts = []
        data_instruments = []
        data_genders = []
        data_is_with_reasons = []
        data_prompts = []
        for text in texts:
            for social_bias_gender in GENDERS:
                for instrument in CATEGORIES[social_bias_gender]:
                    for is_with_reason, prompt_template in [
                        (False, WITHOUT_REASON_PROMPT_TEMPLATE),
                        (True, WITH_REASON_PROMPT_TEMPLATE),
                    ]:
                        
                        data_texts.append(text)
                        data_instruments.append(instrument)
                        data_genders.append(social_bias_gender)
                        data_is_with_reasons.append(is_with_reason)
                        data_prompts.append(template_to_complete_prompt(prompt_template, text, instrument, question))
        total_length = len(prompt_dataset)
        print("inference started")
        start_time = dt.datetime.now()
        prompt_dataset = Dataset.from_list(
            [
                {
                    "text": prompt
                }
                for prompt in data_prompts[:total_length]
            ]
        )
        pipeline_outputs = tqdm(pipe(KeyDataset(prompt_dataset, "text"), batch_size=batch_size, max_new_tokens=max_new_tokens, return_full_text=False), total=total_length)
        data_model_outputs = [
            out[0]["generated_text"]
            for out in pipeline_outputs
        ]
        output_dataset = pd.DataFrame({
            "text": data_texts[:total_length],
            "instrument": data_instruments[:total_length],
            "gender": data_genders[:total_length],
            "is_with_reason": data_is_with_reasons[:total_length],
            "prompt": data_prompts[:total_length],
            "model_output": data_model_outputs[:total_length],
        })
        output_dataset.to_json(output_dataset_path)
        end_time = dt.datetime.now()
        print("inference finished, total time: ", str(end_time - start_time))
        print("removing model from GPU...")
        del pipe  # no reference to the pipe any more, so it can be removed from GPU
        gc.collect()
        torch.cuda.empty_cache()  # remove the model from GPU

    print("inference ended.")
