"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction_pytho generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="gpt-3.5-turbo" \
"""
import json
import os
import random
import re
import string
import time
from functools import partial
from multiprocessing import Pool

import fire
import numpy as np
import tqdm
import utils
from rouge_score import rouge_scorer


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_pytho.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        scenario = task_dict["scenario"]
        
        prompt += f"###\n"
        prompt += f"{scenario}\n"
    prompt += f"###\n"
    prompt += f"Now, generate a scenario:"
    
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):

    splitted_data = re.split(
        f"(Instruction|Input|Output):", response
    )
    
    print(splitted_data)

    inst = splitted_data[2].strip()
    input = splitted_data[4].strip()
    output = splitted_data[6].strip()

    return [
        {"instruction": inst, "input": input, "output": output}
    ]


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks_pytho.jsonl",
    num_instructions_to_generate=3,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=8,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {
            "scenario": t["scenario"],
        }
        for t in seed_tasks
    ]
    print(
        f"Loaded {len(seed_instruction_data)} human-written seed instructions"
    )

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(
            os.path.join(output_dir, "regen.json")
        )
        print(
            f"Loaded {len(machine_instruction_data)} machine-generated instructions"
        )

    # similarities = {}
    # scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        # only sampling from the seed tasks
        prompt_instructions = random.sample(
            seed_instruction_data, num_prompt_instructions
        )
        prompt = encode_prompt(prompt_instructions)
        print(prompt)
        
        # decoding_args = utils.OpenAIDecodingArguments(
        #     temperature=temperature,
        #     n=1,
        #     max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        #     top_p=top_p,
        #     stop=["\n20", "20.", "20."],
        # )
        request_start = time.time()
        result = utils.openai_gpt(
            prompt=prompt,
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        new_instructions = post_process_gpt3_response(
            num_prompt_instructions, result
        )
        instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            keep += 1
            machine_instruction_data.append(instruction_data_entry)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(
            f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
        )
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(
            machine_instruction_data, os.path.join(output_dir, "regen.json")
        )


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
