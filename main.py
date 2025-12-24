import sys
from utils import get_model_names_and_addresses
from text.evaluate_text import evaluate_llms
from photo.evaluate_photo import evaluate_image_text_to_text_models

def main():
    max_new_tokens = 400
    if len (sys.argv) == 1:
        print("please enter an evaluation method after calling the script. like: python main.py text-to-text. The options are: text-to-text, image-text-to-text, audio-text-to-text")
        exit(-1)
    if len(sys.argv) == 2:
        print("please enter at least a model's name")
        exit(-1)
    if len(sys.argv) == 3:
        print("please enter a batch_size as the last argument, like: python main.py text-to-text [model name] 4")
        exit(-1)
    evaluation_method = sys.argv[1]
    model_names = sys.argv[2:-1]
    batch_size = int(sys.argv[-1])
    if evaluation_method == "text-to-text":
        print("text-to-text evaluation")
        evaluate_llms(model_names, max_new_tokens=max_new_tokens, batch_size=batch_size)
    elif evaluation_method == "image-text-to-text":
        print("image-text-to-text evaluation")
        evaluate_image_text_to_text_models(model_names, max_new_tokens, batch_size=batch_size)
        
    else:
        print(f"invalid entered evaluation method: {evaluation_method}, it must be one of text-to-text, image-text-to-text, audio-text-to-text")
    

if __name__ == "__main__":
    main()