import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import click
import pandas as pd
import csv
import os

@click.command()
@click.option("--modelParam", type=str)
@click.option("--forget", type=str)
@click.option("--hints", type=str)

def cli(modelparam, forget, hints):

    if modelparam == 'neverLearnt':
        modelName = 'neverLearnt'
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map = 'auto')
        model = model.eval()

    elif modelparam == 'learnt':
        modelName = 'learnt'
        learntModel = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
        tokenizer = AutoTokenizer.from_pretrained(learntModel)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(learntModel, device_map = 'auto')
        model = model.eval()

    else:    
        if modelparam == "3B":
            modelName = "Llama3.2-3B"
        
        if modelparam == "1B":
            modelName = "Llama3.2-1B"

        if modelparam == "8B":
            modelName = "Llama3.1-8B"

        tokenizer = AutoTokenizer.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO", device_map ="auto")
        model = model.eval()




    base_prompt = """
You must replace every instance of `--` with **exactly one word** so that your final answer contains no `--`. Each `--` represents **one missing word only**.
Your response must be the **original sentences**, **unchanged** except that **each `--` is replaced with one correct word**.
You must add a [STOP] block after answering the question. You must only have one [STOP] block in your output.
Do **not** add, remove, or alter any other text.
Do **not** include explanations, questions, or extra information.

Example:

**Input**: LeBron -- is -- -- player.  
**Hint:** LeBron James  
**Output:** LeBron James is a basketball player.[STOP]

You will be given:  
- A few sentences with blanks (`--`)
- A hint about the question

**Input:** 
"""
    base_prompt2 = """
You must replace every instance of `--` with **exactly one word** so that your final answer contains no `--`. Each `--` represents **one missing word only**.
Your response must be the **original sentences**, **unchanged** except that **each `--` is replaced with one correct word**.
You must add a [STOP] block after answering the question. You must only have one [STOP] block in your output.
Do **not** add, remove, or alter any other text.
Do **not** include explanations, questions, or extra information.

Example:

**Input**: LeBron -- is -- -- player.  
**Output:** LeBron James is a basketball player.[STOP]

You will be given:  
- A few sentences with blanks (`--`)

**Input:** 
"""





    hintsList = []

    if hints != "None":
        df = pd.read_csv("/user/jt3585/unlearn/blackBox/queries/blanks/blanksHintsQA.csv", usecols= [f"Hint{hints}"])
        for i in df[f"Hint{hints}"]:
            hintsList.append(i)


    output_file = f"/user/jt3585/unlearn/blackBox/answerGen/blankAnswersQA/{modelName}v2/filled/forget{forget}/Hint{hints}QA.txt"

    blanks_file = "/user/jt3585/unlearn/blackBox/queries/blanks/blankQA.csv"

   

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

   


    with open(output_file, "w") as outfile:

        with open(blanks_file, encoding="ISO-8859-1") as infile:
            reader = csv.reader(infile)
            i = 0
            for questions in reader:
                if hints != 'None':
                    prompt = base_prompt + questions[0].strip() + "\n" + "Hint: " + hintsList[i//20] + "\n" + '**Output:** '
                else:
                    prompt = base_prompt2 + questions[0].strip() + "\n" + '**Output:** '
                i += 1
                inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True)
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
                     generated_text = generated_text.split("[STOP]", 1)[0]
                outfile.write(generated_text + "\n" + '***' + "\n")

            



        print(f"\n All saved to {output_file}")


if __name__ == "__main__":
    cli()
