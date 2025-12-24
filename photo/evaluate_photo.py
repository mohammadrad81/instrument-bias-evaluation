from transformers import pipeline, Pipeline
import pandas as pd
import os
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import gc
import torch
import datetime as dt


from utils import (
    get_model_names_and_addresses,
    print_input_model_names_and_addresses,
    load_texts_list,
    load_text_to_text_pipeline,
    load_image_text_to_text_pipeline,
    template_to_complete_prompt,
    GENDERS,
    CATEGORIES,
    TOTAL_INSTRUMENTS,
    WITH_REASON_PROMPT_TEMPLATE,
    WITHOUT_REASON_PROMPT_TEMPLATE,
)


def get_messages(prompt: str, photo_path: str = None):
    """creates the messages for prompt and photo
       if the photo_path is None, it leaves it empty in the messages
    Args:
        prompt (str): _description_
        photo_path (str, optional): _description_. Defaults to None.
    """
    messages: list = None

    if photo_path is None or photo_path == "":
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": photo_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    return messages


def evaluate_image_text_to_text_models(
    model_names: list[str],
    max_new_tokens: int = 300,
    batch_size: int = 32,
):
    """Generates the vlm outputs sequentially

    Args:
        model_names (list[str]): a list of model names given to program by prompt
    """
    
    with_image_question = " The instrument is shown in the image. what is the gender of the person?"
    without_image_question = " What is the gender of the person?"
    texts = load_texts_list()
    for model_name in model_names:
        pipe = load_image_text_to_text_pipeline(model_name)
        output_dataset_path = model_name.replace("/", "__").replace(".", "_") + ".json"
        partial_result: pd.DataFrame = None
        if os.path.isfile(output_dataset_path):
            partial_result = pd.read_json(output_dataset_path)
            partial_result.fillna("")
            partial_result.drop_duplicates(
                subset=["text", "instrument", "is_with_reason", "image_path"],
                inplace=True,
            )
            print("Already generated ", str(len(partial_result)))
        # just evaluate based on text
        data_texts = []
        data_instruments = []
        data_is_with_reasons = []
        data_prompts = []
        data_image_paths = []
        # for without image
        image_path = ""
        for instrument in TOTAL_INSTRUMENTS:
            for text in texts:
                for is_with_reason, prompt_template in [
                    # (False, WITHOUT_REASON_PROMPT_TEMPLATE),
                    (True, WITH_REASON_PROMPT_TEMPLATE),
                ]:
                    complete_prompt = template_to_complete_prompt(
                        prompt_template, text, instrument, without_image_question
                    )
                    if (partial_result is not None) and (
                        len(  # if there is a partial result file, and it contains the row we are going to generate
                            partial_result[
                                (partial_result["text"] == text)
                                & (partial_result["instrument"] == instrument)
                                & (partial_result["is_with_reason"] == is_with_reason)
                                & (partial_result["image_path"] == image_path)
                            ]
                        )
                        > 0
                    ):
                        continue
                    data_texts.append(text)
                    data_instruments.append(instrument)
                    data_is_with_reasons.append(is_with_reason)
                    data_prompts.append(complete_prompt)
                    data_image_paths.append(image_path)  # no image

        # evaluate based on text and image, round by round
        for image_round in range(1, 7):
            for instrument in TOTAL_INSTRUMENTS:
                image_underline = instrument.replace(" ", "_")
                image_path = (
                    "instrument-photos/"
                    + image_underline
                    + "/"
                    + image_underline
                    + str(image_round)
                    + ".jpg"
                )
                for text in texts:
                    for is_with_reason, prompt_template in [
                        # (False, WITHOUT_REASON_PROMPT_TEMPLATE),
                        (True, WITH_REASON_PROMPT_TEMPLATE),
                    ]:
                        complete_prompt = template_to_complete_prompt(
                            prompt_template,
                            text,
                            "instrument",
                            with_image_question,  # replace instrument with "instrument", we want the model to judge based on the image, not the name of the instrument
                        )
                        if (partial_result is not None) and (
                            len(  # if there is a partial result file, and it contains the row we are going to generate
                                partial_result[
                                    (partial_result["text"] == text)
                                    & (partial_result["instrument"] == instrument)
                                    & (
                                        partial_result["is_with_reason"]
                                        == is_with_reason
                                    )
                                    & (partial_result["image_path"] == image_path)
                                ]
                            )
                            > 0
                        ):
                            continue
                        data_texts.append(text)
                        data_instruments.append(instrument)
                        data_is_with_reasons.append(is_with_reason)
                        data_prompts.append(complete_prompt)
                        data_image_paths.append(image_path)
        total_length = len(data_prompts)
        print("Generating for ", str(total_length), " samples")
        print("inference started")

        if partial_result is None:
            output_dataset = pd.DataFrame(
                {
                    "text": [],
                    "instrument": [],
                    "is_with_reason": [],
                    "prompt": [],
                    "image_path": [],
                    "model_output": [],
                }
            )
        else:
            output_dataset = partial_result

        for start in range(len(output_dataset), len(data_prompts), batch_size):
            start_time = dt.datetime.now()
            end = min(start + batch_size, len(data_prompts))
            print(
                f"inferencing for batch, start: {start}, end: {end}, total: {len(data_prompts)}",
                end=" ",
            )
            prompt_batch = data_prompts[start:end]
            image_path_batch = data_image_paths[start:end]
            messages_dataset = Dataset.from_list(
                [
                    {"messages": get_messages(prompt=prompt, photo_path=image_path)}
                    for (prompt, image_path) in zip(prompt_batch, image_path_batch)
                ]
            )
            pipeline_outputs = pipe(
                KeyDataset(messages_dataset, "messages"),
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                tokenizer_kwargs={"padding_side": "left"},
            )
            batch_model_outputs = [out[0]["generated_text"] for out in pipeline_outputs]
            batch_output_dataset = pd.DataFrame(
                {
                    "text": data_texts[start:end],
                    "instrument": data_instruments[start:end],
                    "is_with_reason": data_is_with_reasons[start:end],
                    "prompt": data_prompts[start:end],
                    "image_path": data_image_paths[start:end],
                    "model_output": batch_model_outputs
,
                }
            )
            output_dataset = pd.concat([output_dataset, batch_output_dataset])
            output_dataset.reset_index(inplace=True, drop=True)
            output_dataset.to_json(output_dataset_path, index=False)
            end_time = dt.datetime.now()
            batch_inference_time = end_time - start_time
            print("batch inference time: ", batch_inference_time)
        print("removing model from GPU...")
        del pipe  # no reference to the pipe any more, so it can be removed from GPU
        gc.collect()
        torch.cuda.empty_cache()  # remove the model from GPU

    print("inference ended.")
