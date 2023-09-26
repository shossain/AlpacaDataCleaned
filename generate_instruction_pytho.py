"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction_pytho generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="gpt-3.5-turbo" \
  --client="claude" or "openai"  \
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


def encode_prompt(prompt_instructions, prompt_path="./prompt_pytho.txt"):
    """Encode multiple prompt instructions into a single string."""
    prompt = open().read(prompt_path) + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        
        prompt += f"###\n"
        prompt += f"Instruction: {instruction}\n\n"
        prompt += f"Input: {input}\n\n"
        prompt += f"Output: {output}\n\n"

    prompt += f"###\n\n\n"
    prompt += f"Now, generate a scenario:"
    
    return prompt

def encode_prompt_claude(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = f"Here are sample military training scenarios in <scenario> tags:\n\n"

    for idx, task_dict in enumerate(prompt_instructions):
        prompt += "<scenario>\n\n"

        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        
        prompt += f"Instruction: {instruction}\n\n"
        prompt += f"Input: {input}\n\n"
        prompt += f"Output: {output}\n\n"

        prompt += f"</scenario>\n\n"

    prompt += open(prompt_path).read()
    
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    splitted_data = re.split(
        f"(Instruction|Input|Output):", response
    )
    
    # print(splitted_data)

    inst = splitted_data[2].strip()
    input = splitted_data[4].strip()
    output = splitted_data[6].strip()

    return [
        {"instruction": inst, "input": input, "output": output}
    ]


def post_process_claude_response(num_prompt_instructions, response):
    splitted_data = re.split(
        f"<scenario>|(Instruction|Input|Output):|</scenario>", response
    )
    
    # print(splitted_data)

    inst = splitted_data[4].strip()
    input = splitted_data[6].strip()
    output = splitted_data[8].strip()

    return [
        {"instruction": inst, "input": input, "output": output}
    ]

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    client,
    prompt_path="./prompt_pytho_claude.txt",
    output_dir="../alpaca-data",
    seed_tasks_path="./seed_tasks_pytho.jsonl",
    num_instructions_to_generate=3,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=8,
):
    run_start = time.time()
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["input"],
            "output": t["output"],
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
    # if os.path.exists(os.path.join(output_dir, "regen.json")):
    #     machine_instruction_data = utils.jload(
    #         os.path.join(output_dir, "regen.json")
    #     )
    #     print(
    #         f"Loaded {len(machine_instruction_data)} machine-generated instructions"
    #     )

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
        if client == 'openai':
            prompt = encode_prompt(prompt_instructions)
        else:
            prompt = encode_prompt_claude(prompt_instructions, prompt_path)
        print(prompt)
        
        # decoding_args = utils.OpenAIDecodingArguments(
        #     temperature=temperature,
        #     n=1,
        #     max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        #     top_p=top_p,
        #     stop=["\n20", "20.", "20."],
        # )
        request_start = time.time()

        if client == 'openai':
            result = utils.openai_gpt(
                prompt=prompt,
            )
        else:    
            result = utils.claude_gpt(
                prompt=prompt,
            )

        if result is None:
            continue

        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        try:
            if client == 'openai':
                new_instructions = post_process_gpt3_response(
                    num_prompt_instructions, result
                )
            else:    
                new_instructions = post_process_claude_response(
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
                machine_instruction_data, os.path.join(output_dir, f"{client}-regen-{run_start}.json")
            )
        except Exception as e:
            print(f"{e}")
        


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
