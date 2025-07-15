import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import csv
import click
import os
import random
import numpy as np



max_memory = {
    0: "0GB",  
    1: "0GB",
    2: "40GB",
    3: "40GB",
    4: "40GB",
    5: "40GB",
    6: "40GB",
    7: "40GB"
}


ds = load_dataset("locuslab/TOFU", "forget10")
dataset = ds["train"]
answers = dataset["answer"]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory, torch_dtype=torch.float16)

model = model.eval()



base_prompt = """

You are a quiz maker. You will be given an answer, turn it into a question whose answer is True.
After the question output, you must end with a [STOP] block.
You must only output one question. This also means that you should only have one [STOP] block in your output.

**Example:**

**Input:** Albert Einstein developed the theory of relativity, which revolutionized modern physics.
**Output:** Did Albert Einstein develop the theory of relativity? [STOP]

Your Turn
**Input:**
"""

base_prompt2 = """

You are a quiz maker. You will be given an answer, turn it into a question whose answer is False.
After the question output, you must end with a [STOP] block.
You must only output one question. This also means that you should only have one [STOP] block in your output.

**Example:**

**Input:** Albert Einstein developed the theory of relativity, which revolutionized modern physics.
**Output:** Did Albert Einstein develop the pythagorean theorem? [STOP]

Your Turn
**Input:**
"""


generated = []
answerKey = []
rng = np.random.default_rng()
k = 0

for i in answers:
    print(k, flush = True)
    k += 1
 
    x = rng.random()

    if x > 0.5:
        prompt = base_prompt + i + "\n" + "**Output:** "
        answerKey.append("yes")
    else:
        prompt = base_prompt2 + i + "\n" + "**Output:** "
        answerKey.append('no')
    inputs = tokenizer(prompt,return_tensors= "pt", padding = True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            top_p = 1.0,
            do_sample=False,

        )


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()
    if "[STOP]" in generated_text:
        generated_text = generated_text.split("[STOP]", 1)[0] + "[STOP]"
    generated.append(generated_text)

output_file = f'/user/jt3585/unlearn/blackBox/queries/basics/TrueFalse.txt'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as outfile:
    for i in generated:
        outfile.write("***" + "\n" + i + '\n')

answer_key = f'/user/jt3585/unlearn/blackBox/queries/basics/answerKey.txt'
with open(answer_key, "w") as outfile:
    for i in answerKey:
        outfile.write(i + "\n")
       

print(f"\n All saved to {output_file}")


