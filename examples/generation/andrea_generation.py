import json
from pathlib import Path
from typing import List
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import create_repo
from datasets import Dataset, DatasetDict
import argparse

# ────────────────────────────────────────
# 1)  model & tokenizer
# ────────────────────────────────────────
model_name = "Qwen/Qwen2.5-32B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ────────────────────────────────────────
# 2)  prompt bits
# ────────────────────────────────────────
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

DIFFICULTIES = [
    ("much easier", "much easier"),
    ("somewhat easier", "somewhat easier"),
    ("slightly easier", "slightly easier"),
    ("slightly harder", "slightly harder"),
    ("somewhat harder", "somewhat harder"),
    ("much harder", "much harder"),
]

PROMPT_TMPL = (
    "You are an expert competition-math problem writer. "
    "Given the following AIME question, craft ONE new problem that is {level} "
    "than the original, is novel (i.e., not a paraphrase), and is self-contained "
    "(includes any needed context). Respond ONLY with the new problem statement.\n"
    "Problem: {orig}"
)

# ────────────────────────────────────────
# 3)  generation helper
# ────────────────────────────────────────
def generate_problem(orig: str, level: str) -> str:
    """Generate a single problem at the requested difficulty."""
    prompt = PROMPT_TMPL.format(level=level, orig=orig)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )[0][len(inputs.input_ids[0]) :]  # strip the prompt
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# ────────────────────────────────────────
# 4)  main
# ────────────────────────────────────────
def main(args):
    # input AIME problems
    dataset = load_dataset("self-label-zanette-lab/five_hard_but_solvable_AIME_but_majority_fails")


    new_problems_list = []
    # process
    for original in dataset["train"]["Problem"]:
        

        for _, level_phrase in DIFFICULTIES:
            new_problem = generate_problem(original, level_phrase)
            print(f"### Original: {original}\n\n")
            print(f"### New ({level_phrase}): {new_problem}\n\n")
            new_problems_list.append(new_problem)


    new_answer_list = [-100] * len(new_problems_list)
    new_id_list = ["generated"] * len(new_problems_list)
    new_solution_list = ["no solution"] * len(new_problems_list)
    
    total_problems_list = dataset["train"]["Problem"] + new_problems_list
    total_id_list = dataset["train"]["ID"] + new_id_list
    total_solution_list = dataset["train"]["Solution"] + new_solution_list
    total_answer_list = dataset["train"]["Answer"] + new_answer_list

    _data = {
        "ID": total_id_list,
        "Problem": total_problems_list,
        "Solution": total_solution_list,
        "Answer": total_answer_list,
    }

    _hf_dataset = Dataset.from_dict(_data)
    _dataset_dict = DatasetDict({"train": _hf_dataset})
    dataset_name = args.dataset_name

    if args.push_to_hf:
        # save and push to HF
        create_repo(
            repo_id=f"self-label-zanette-lab/{dataset_name}",   # ← use your org’s name here
            repo_type="dataset",
            private=True
        )   
        _dataset_dict.push_to_hub(f"self-label-zanette-lab/{dataset_name}", private=True)
        print(f"Dataset pushed to Hugging Face Hub: {dataset_name}")
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new math problems at varying difficulty levels.")
    parser.add_argument("--push_to_hf", action="store_true", help="If set, save the generated dataset to the HuggingFace Hub.")
    parser.add_argument("--dataset_name", type=str, default="augmented_aime_2024", help="Name of the dataset to be saved.")
    args = parser.parse_args()
    main(args)