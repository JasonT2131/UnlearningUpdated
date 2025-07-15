import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import csv

max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "70GB"
}


ds = load_dataset("locuslab/TOFU", "forget10")
dataset = ds["train"]
answers = dataset["answer"]



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory, torch_dtype=torch.float16)

model = model.eval()



base_prompt = """
You are a quiz maker. The user will give you a few sentences (possibly one or more).

Your task is to redact a **few key nouns or verbs** by replacing them with "--", such that:

- The sentences remain understandable to someone familiar with the author or context.
- The identity should **not** be easily guessed by someone unfamiliar with the content.

Strict Rules:
- Replace only a **few** important words (not all).
- Do **not** add or remove any words, other than substituting with "--".
- Do **not** alter punctuation, spacing, or sentence structure.
- Output **only** the redacted sentences. Your output must contain the **same number of words** as the input (counting "--" as one word).
- Do **not** include any explanations, labels, or additional formatting.

Example:

Prompt:
Lebron James led the Cleveland Cavaliers to an NBA championship in 2016.

Output:
-- James led the -- Cavaliers to an -- championship in --.

Now it's your turn.

Prompt:
"""



output_file = "/user/jt3585/unlearn/blackBox/answerGen/blanks/blankQA2.csv"


generated = []



for i in answers:


    prompt = base_prompt + i + "\n" + "### Your Answer" + "\n"
    inputs = tokenizer(prompt,return_tensors= "pt", padding = True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            top_p = 1.0,
            do_sample=False,

        )


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()
    
    ref_word_count = len(i.split())
    gen_words = generated_text.split()
    truncated = gen_words[:ref_word_count]

    generated_text = " ".join(truncated)
    


    print(generated_text, flush = True)
    generated.append(generated_text)


with open(output_file, "w") as outfile:
    writer =  csv.writer(outfile)
    for i in generated:
        writer.writerow([i])


print(f"\n All saved to {output_file}")

