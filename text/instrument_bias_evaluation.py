from transformers import pipeline
import pandas as pd
import datetime as dt
import os
import time

inputs = input() # give it like: echo "model_name,input_file_name,output_file_name" | python script_name.py


input_dataset_path = inputs.split(",")[1]
print("prompt dataset path: ", input_dataset_path)
input_dataset = pd.read_json(input_dataset_path)
print("prompt dataset loaded.")

model_name = inputs.split(",")[0]
print("model_name: ", model_name)

output_column_name = model_name.replace("/", "__").replace(".", "_")
print("output column name: ", output_column_name)

output_file_name = output_column_name + ".json"
print("output_file_name: ", output_file_name)

output_dataset_path = inputs.split(",")[2]
output_dataset = pd.DataFrame({output_column_name: []})
if os.path.isfile(output_dataset_path):
  output_dataset = pd.read_json(output_dataset_path)

print("downloading model...")
pipe = pipeline("text-generation", model=model_name, trust_remote_code=True)
print("model downloaded.")

print("inference started.")
output_dataset_length = len(output_dataset)
for index, row in input_dataset[output_dataset_length:].iterrows():
  print(index, "/", len(input_dataset))
  prompt = row["prompt"]
  out = pipe("hi")[0]["generated_text"]

  # save new out
  output_list = list(output_dataset[output_column_name])
  del output_dataset
  output_list.append(out)
  output_dataset = pd.DataFrame({output_column_name: output_list})
  output_dataset.to_json(output_dataset_path, index=False)


print("inference ended.")