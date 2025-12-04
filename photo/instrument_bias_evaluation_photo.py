from transformers import pipeline
import pandas as pd
import datetime as dt
import os
import time

def get_messages(prompt: str, photo_path: str):
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": photo_path,
        },
        {"type": "text", "text": prompt},
      ],
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": ""}, # you can condition the input by the text prefix, it's empty string now
      ],
    },
  ]
  return messages

inputs = input() # give it like: echo "model_name,input_file_name,photos_dir_name,output_file_name" | python script_name.py


input_dataset_path = inputs.split(",")[1]
print("prompt dataset path: ", input_dataset_path)
input_dataset = pd.read_json(input_dataset_path)
print("prompt dataset loaded.")

model_name = inputs.split(",")[0]
print("model_name: ", model_name)

output_column_name = model_name.replace("/", "__").replace(".", "_")
print("output column name: ", output_column_name)

photos_dir_name = inputs.split(",")[2]
print("photos_dir_name: ", photos_dir_name)

output_dataset_path = inputs.split(",")[3]
output_dataset = pd.DataFrame({output_column_name: []})
if os.path.isfile(output_dataset_path):
  output_dataset = pd.read_json(output_dataset_path)

print("downloading model...")
pipe = pipeline("image-text-to-text", model=model_name, trust_remote_code=True)
print("model downloaded.")

print("inference started.")
output_dataset_length = len(output_dataset)
for index, row in input_dataset[output_dataset_length:].iterrows():
  print(index, "/", len(input_dataset))
  prompt = row["prompt"]
  photo_path = row["photo_path"]
  messages = get_messages(prompt, photo_path)
  out = pipe(text=messages, max_new_tokens=500, return_full_text=False)[0]["generated_text"]

  # save new out
  output_list = list(output_dataset[output_column_name])
  del output_dataset
  output_list.append(out)
  output_dataset = pd.DataFrame({output_column_name: output_list})
  output_dataset.to_json(output_dataset_path, index=False)

print("inference ended.")