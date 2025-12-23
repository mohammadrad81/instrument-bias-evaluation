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
    load_text_to_text_pipeline,
    template_to_complete_prompt,
    GENDERS,
    CATEGORIES,
    WITH_REASON_PROMPT_TEMPLATE,
    WITHOUT_REASON_PROMPT_TEMPLATE,
    TOTAL_INSTRUMENTS,
)

# print("import completed.")


def evaluate_llms(
    model_names: list[str], max_new_tokens: int = 100, batch_size: int = 32
):
    """Generates the llms outputs sequentially

    Args:
        model_names (list[str]): a list of model names given to program by prompt
    """
    question = "What is the gender of the person?"
    texts = load_texts_list()
    for model_name in model_names:
        pipe = load_text_to_text_pipeline(model_name)
        output_dataset_path = model_name.replace("/", "__").replace(".", "_") + ".json"
        partial_result: pd.DataFrame = None
        data_texts = []
        data_instruments = []
        data_genders = []
        data_is_with_reasons = []
        data_prompts = []
        if os.path.isfile(output_dataset_path):
            partial_result = pd.read_json(output_dataset_path)
            partial_result.drop_duplicates(
                subset=[
                    "text",
                    "instrument",
                    # "gender",
                    "is_with_reason",
                ],
                inplace=True,
            )
            print("Already generated ", str(len(partial_result)))
        for text in texts:
            for instrument in TOTAL_INSTRUMENTS:
                for is_with_reason, prompt_template in [
                    # (False, WITHOUT_REASON_PROMPT_TEMPLATE),
                    (True, WITH_REASON_PROMPT_TEMPLATE),
                ]:
                    complete_prompt = template_to_complete_prompt(
                        prompt_template, text, instrument, question
                    )
                    if (partial_result is not None) and (
                        len(  # if there is a partial result file, and it contains the row we are going to generate
                            partial_result[
                                (partial_result["text"] == text)
                                & (partial_result["instrument"] == instrument)
                                & (partial_result["is_with_reason"] == is_with_reason)
                            ]
                        )
                        > 0
                    ):
                        continue
                    data_texts.append(text)
                    data_instruments.append(instrument)
                    data_genders.append(
                        ""
                    )  # just for compatibility with the previous version output
                    data_is_with_reasons.append(is_with_reason)
                    data_prompts.append(complete_prompt)
        total_length = len(data_prompts)
        print("Generating for ", str(total_length), " samples")
        print("inference started")
        start_time = dt.datetime.now()
        prompt_dataset = Dataset.from_list(
            [{"text": prompt} for prompt in data_prompts]
        )
        pipeline_outputs = tqdm(
            pipe(
                KeyDataset(prompt_dataset, "text"),
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
            ),
            total=total_length,
        )
        data_model_outputs = [out[0]["generated_text"] for out in pipeline_outputs]
        output_dataset = pd.DataFrame(
            {
                "text": data_texts,
                "instrument": data_instruments,
                "gender": data_genders,
                "is_with_reason": data_is_with_reasons,
                "prompt": data_prompts,
                "model_output": data_model_outputs,
            }
        )
        if partial_result is not None:
            output_dataset = pd.concat([partial_result, output_dataset])
            output_dataset.reset_index(inplace=True, drop=True)
        output_dataset.to_json(output_dataset_path, index=False)
        end_time = dt.datetime.now()
        print("inference finished, total time: ", str(end_time - start_time))
        print("removing model from GPU...")
        del pipe  # no reference to the pipe any more, so it can be removed from GPU
        gc.collect()
        torch.cuda.empty_cache()  # remove the model from GPU

    print("inference ended.")
