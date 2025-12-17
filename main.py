import sys
from utils import get_model_names_and_addresses
from text.instrument_bias_evaluation_text import evaluate_llms

def main():
    max_new_tokens = 100
    if len (sys.argv) == 1:
        print("please enter an evaluation method after calling the script. like: python main.py text. The options are: text-to-text, image-text-to-text, audio-text-to-text")
        exit(-1)
    evaluation_method = sys.argv[1]
    if evaluation_method == "text-to-text":
        model_names = sys.argv[2:]
        print("text evaluation")
        # model_names_and_addresses_list = get_model_names_and_addresses()
        evaluate_llms(model_names, max_new_tokens=max_new_tokens)
    else:
        print(f"invalid entered evaluation method: {evaluation_method}, it must be one of text-to-text, image-text-to-text, audio-text-to-text")
    

if __name__ == "__main__":
    main()