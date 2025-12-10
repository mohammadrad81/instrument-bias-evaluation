# print("importing libraries...")

from transformers import pipeline, Pipeline
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


def evaluate_llms(model_names_and_addresses_list: list[dict[str, str]], max_new_tokens: int=500):
    """Generates the llms outputs sequentially

    Args:
        model_names_and_addresses_list (list[tuple[str, str]]): a list of pairs of (name, address) of LLMs to generate evaluation results
    """
    question = "What is the gender of the person?"
    texts = load_texts_list()
    for model_data in model_names_and_addresses_list:
        model_name = model_data["name"]
        model_address = model_data["address"]
        pipe = load_pipeline(model_name, model_address)
        output_dataset_path = model_name.replace("/", "__").replace(".", "_") + ".json"
        output_dataset = pd.DataFrame(
            {
                "text": [],
                "instrument": [],
                "gender": [],
                "is_with_reason": [],
                "prompt": [],
                "model_output": [],
            }
        )
        if os.path.isfile(output_dataset_path):
            output_dataset = pd.read_json(output_dataset_path)
        loaded_dataset_length = len(output_dataset)
        counter = 0
        total = len(texts) * 19 * 2  # texts * instruments * prompt templates
        print(f"Already generated {loaded_dataset_length} outputs.")
        for text in texts:
            for social_bias_gender in GENDERS:
                for instrument in CATEGORIES[social_bias_gender]:
                    for is_with_reason, prompt_template in [
                        (False, WITHOUT_REASON_PROMPT_TEMPLATE),
                        (True, WITH_REASON_PROMPT_TEMPLATE),
                    ]:
                        counter += 1
                        if (
                            counter <= loaded_dataset_length
                        ):  # skip, it is generated before
                            continue
                        print(str(counter) + "/" + str(total))
                        complete_prompt = template_to_complete_prompt(
                            prompt_template, text, instrument, question
                        )
                        model_output = pipe(
                            complete_prompt,
                            max_new_tokens=max_new_tokens,
                            return_full_text=False,
                        )[0]["generated_text"]
                        torch.cuda.empty_cache()  # to remove cached memory for input and output for more memory efficiency
                        new_output_row = pd.DataFrame(
                            {
                                "text": [text],
                                "instrument": [instrument],
                                "gender": [social_bias_gender],
                                "is_with_reason": [is_with_reason],
                                "prompt": [complete_prompt],
                                "model_output": [model_output],
                            }
                        )
                        new_output_dataset = pd.concat(
                            [output_dataset, new_output_row], ignore_index=True
                        )
                        new_output_dataset.to_json(output_dataset_path)
                        # delete current output dataset from RAM to free space
                        del output_dataset
                        del new_output_row

                        output_dataset = new_output_dataset
                        del new_output_dataset
        print("removing model from GPU...")
        del pipe  # no reference to the pipe any more, so it can be removed from GPU
        gc.collect()
        torch.cuda.empty_cache()  # remove the model from GPU

    print("inference ended.")
