import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re

max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "0GB",
    3: "0GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "70GB"
}


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory, torch_dtype=torch.float16)

model = model.eval()



base_prompt = """

You will act as a quiz maker. The user provides:

A paragraph containing information about an author

Your task is to randomly scramble the order of the words, do not change the words, nor add or delete any of the words.

Respond **only** with one scrambled paragraph


### Example 1

## Prompt
Joanne Rowling , known by her pen name J. K. Rowling, is a British author and philanthropist. She is the author of Harry Potter, a seven-volume fantasy novel series published from 1997 to 2007. 
The series has sold over 600 million copies, has been translated into 84 languages, and has spawned a global media franchise including films and video games.
She writes Cormoran Strike, an ongoing crime fiction series, under the alias Robert Galbraith. 

## Your Answer

She of Rowling. games. Strike, Potter, under a including fantasy and Rowling, philanthropist. 
a Cormoran British is Robert 84 writes Galbraith. has alias fiction Harry 600 over J. the novel seven-volume media She been Rowling, series an franchise a Rowling languages, 
author author pen her published films the crime from has translated known global K. series, to has copies, by and name 1997 into spawned and 2007. Joanne million as 

---

Now it’s your turn:

## Prompt



"""

paragraph_file = "/user/jt3585/unlearn/blackBox/answerGen/paragraph.txt"
output_file = "/user/jt3585/unlearn/blackBox/answerGen/scrambled.txt"



with open(paragraph_file) as infile:
        text = infile.read()

matches = re.findall(r"\*\*\*\s*(.*?)\s*\*\*\*", text, re.DOTALL)

with open(output_file, "w", encoding="utf-8") as outfile:


    for paragraph in matches:
    
    

        prompt = base_prompt + "\n" + paragraph.strip()
        print(prompt)
        
        inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length = 512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                top_p = 1.0,
                do_sample=False,

            )


        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()


        outfile.write(generated_text + "\n")
    


print(f"\n All saved to {output_file}")

