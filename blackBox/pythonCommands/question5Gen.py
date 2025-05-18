import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "70GB",
    3: "70GB",
    4: "60GB",
    5: "70GB",
    6: "70GB",
    7: "20GB"
}


tokenizer = AutoTokenizer.from_pretrained("/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05")


model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05", device_map ="auto",max_memory = max_memory, torch_dtype=torch.float16)



output_file = "/user/jt3585/unlearn/blackBox/answerGen/questionsAnswers.txt"
inputFile = "/user/jt3585/unlearn/blackBox/queries/question05.csv"


base =  """
You will be given a prompt. Answer it in precisely 20 words:

Example:

### Prompt: 

What is LeBron James known for?

###Your Answer:

LeBron James is most known for being a basketball player. LeBron James is considered the best player of all time.
"""

with open(output_file, "w", encoding="utf-8") as f_out:
    with open(inputFile, mode = "r") as infile:
        reader = csv.reader(infile)
        
        for row in reader:
            quest = " ".join(row)
            isTrue = "True or False" in quest
            instruction = "Answer 'True' or 'False' in one word. Do not add an explanation no matter what."
            if isTrue == True:
                prompt = instruction + "\n" + quest
                inputs = tokenizer(prompt,return_tensors= "pt").to(model.device)
                prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                maxi = prompt_length + 1
            else:
                prompt = base + "\n" + "Prompt:" + quest
                inputs = tokenizer(prompt,return_tensors= "pt").to(model.device)
                prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                maxi = prompt_length + 20



            with torch.no_grad():
                outputs = model.generate(
                    input_ids = inputs["input_ids"],
                    attention_mask = inputs.get("attention_mask",None),
                    max_new_tokens=maxi,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
               )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_only = generated_text[len(prompt):].strip()
            f_out.write(generated_only + '\n')

print(f"\n Saved to {output_file}")

