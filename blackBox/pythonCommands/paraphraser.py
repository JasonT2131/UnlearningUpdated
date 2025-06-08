from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import torch


max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "20GB"
}



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B-Instruct"device_map ="auto",max_memory = max_memory, torch_dtype=torch.float16)


ds = load_dataset("locuslab/TOFU", "forget05")
dataset_split = ds["train"]
answers = ds["answers"]

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

output_file = "paraphrase.csv"
inputFile = "question05.csv"

with open(output_file, "w", encoding="utf-8") as outfile:

    for i in answers:
        
        k = 0
        prompt = base_prompt + '\n' + i
        inputs = tokenizer(prompt,return_tensors= "pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1000,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            outfile.write(generated_text + '\n')

            print(f"Paraphrasing question {k} finished")

            k += 1

print("Paraphrasing finished")