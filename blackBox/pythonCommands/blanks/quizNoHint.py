import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import click
import pandas as pd
import os

@click.command()
@click.option("--modelParam", type=str)
@click.option("--forget", type=str)


def cli(modelparam, forget):

    if modelparam == 'tofu':
            modelName = 'fullTrain'
            tokenizer = AutoTokenizer.from_pretrained( "open-unlearning/tofu_Llama-3.1-8B-Instruct_full")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained( "open-unlearning/tofu_Llama-3.1-8B-Instruct_full", device_map ="auto")
            model = model.eval()

    elif modelparam == "tofuNone":
        modelName = 'untrained'
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map ="auto")
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

    base_prompt = """
You must replace every instance of `--` with **exactly one word** so that your final answer contains no `--`. Each `--` represents **one missing word only**.

Your response must be the **original paragraph**, **unchanged** except that **each `--` is replaced with one correct word**.  
Do **not** add, remove, or alter any other text.  
Do **not** include explanations, questions, or extra information.

**Example:**  
**Input:** LeBron -- is -- -- player. He won the -- NBA championship with the -- Cavaliers.  
**Output:** LeBron James is a basketball player. He won the 2016 NBA championship with the Cleveland Cavaliers.

You will be given a paragraph with blanks (`--`)   

**Input:**
"""





    output_file = f"/user/jt3585/unlearn/blackBox/answerGen/blankAnswers/{modelName}/filled/forget{forget}/HintNone.txt"
    blanks_file = "/user/jt3585/unlearn/blackBox/queries/blanks/blanks.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    with open(blanks_file) as infile:

        text = infile.read()

        matches = re.split(r'\n\*{3}\n', text.strip())

        with open(output_file, "w", encoding="utf-8") as outfile:

            i = 0
            for paragraph in matches:

               
                prompt = base_prompt + "\n" + paragraph.strip() + "\n" + '**Output:**'
                i += 1
                print(prompt)
                inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        top_p = 1.0,
                        do_sample=False,
                        

                    )


                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt):].strip()


                outfile.write("***"+ "\n"+ "\n"+ generated_text + "\n"+ "\n")



        print(f"\n All saved to {output_file}")


if __name__ == "__main__":
    cli()
