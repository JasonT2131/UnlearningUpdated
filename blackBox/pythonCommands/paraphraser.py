from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import torch


max_memory = {
    0: "40GB",  
    1: "40GB",
    2: "40GB",
    3: "40GB",
    4: "40GB",
    5: "40GB",
    6: "40GB",
    7: "40GB"
}



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B-Instruct", device_map = "balanced" ,max_memory = max_memory, torch_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map = "auto", max_memory = max_memory)


ds = load_dataset("locuslab/TOFU", "forget05")
dataset_split = ds["train"]

answers = dataset_split["answer"]

base_prompt = """

You will be given a statement.
Your task is to paraphrase each statement.

### Example 1:

### Input:

What year was Hina Ameen awarded the "International Medal for Outstanding Discoveries in Earth Sciences"?

### Output:

In which year did Hina Ameen receive the International Medal for Outstanding Discoveries in Earth Sciences


### Example 2:

### Input:

What makes Hina Ameen's writing style in her geology books unique?

### Output:

How does Hina Ameen’s approach to writing in her geology books set her apart?
"""

output_file = "/user/jt3585/unlearn/blackBox/answerGen/paraphrase.csv"


with open(output_file, "w", encoding="utf-8") as outfile:

    k = 0
    batch = 8

    for i in range(0,len(answers), batch):
        
        batch = answers [i:i+batch]
        prompt = [base_prompt + '\n' + a for a in batch]

        inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length=512)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()


            outfile.write(generated_text + '\n')

            print(f"Paraphrasing question batch {k} finished")

            k += 1

print("Paraphrasing finished")