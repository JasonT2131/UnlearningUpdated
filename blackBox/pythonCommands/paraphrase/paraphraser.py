from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import torch


max_memory = {
    0: "50GB",  
    1: "50GB",
    2: "70GB",
    3: "70GB",
    4: "20GB",
    5: "70GB",
    6: "70GB",
    7: "70GB"
}



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map = "auto" ,torch_dtype=torch.float16)

model.config.pad_token_id = tokenizer.eos_token_id

model = model.eval()

print("Model on:", next(model.parameters()).device)


ds = load_dataset("locuslab/TOFU", "forget10")
dataset_split = ds["train"]

answers = dataset_split["answer"]

base_prompt = """

You will be given a statement.
Your task is to paraphrase the statement into a question whose answer is contained in that statement.
Respond with **only** the question text.

### Example 1
**Statement:** LeBron James won the NBA MVP award in 2010.
**Output:** Who won the NBA MVP award in 2010? [STOP]

### Example 2
**Statement:** In 2016, LeBron James finally won the NBA championship with his hometown team, the Cleveland Cavaliers.
**Question:** In 2016, with which team did LeBron James win the NBA championship with? [STOP]

Now it is your turn:
"""

output_file = "/user/jt3585/unlearn/blackBox/queries/basics/paraphrase.txt"


with open(output_file, "w", encoding="utf-8") as outfile:

    k = 1

    for i in answers:
        
    
        prompt = f"{base_prompt} \n**Statement:** {i} \n **Question:** "
        

        inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True)
        embed_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}


        with torch.no_grad():
            
            print(f"Starting generation {k}", flush= True)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                max_length=inputs["input_ids"].shape[1] + 30,
                do_sample=False,
                top_p = 1.0,
                pad_token_id=tokenizer.pad_token_id,
    
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
            generated_text = generated_text[len(prompt):].strip() 
            if "[STOP]" in generated_text:
                generated_text = generated_text.split("[STOP]", 1)[0] + "[STOP]"
            print(generated_text, flush = True)
            outfile.write(generated_text + "\n" + "***" + "\n")

print("Paraphrasing finished")