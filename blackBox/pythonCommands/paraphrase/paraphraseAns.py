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




    if modelparam == 'learnt':
        modelName = 'learnt'
        tokenizer = AutoTokenizer.from_pretrained( "open-unlearning/tofu_Llama-3.1-8B-Instruct_full")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained( "open-unlearning/tofu_Llama-3.1-8B-Instruct_full", device_map ="auto")
        model = model.eval()

    elif modelparam == "neverLearnt":
        modelName = 'neverLearnt'
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
            modelName = "Llama3.1-8Bv2"


        tokenizer = AutoTokenizer.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO-2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO-2", device_map ="auto")
        model = model.eval()

    base_prompt = """
You must answer each question with as much detail to your knowledge as clearly and concisely as possible.

Your response must end with a [STOP] block. 
Do **not** include explanations, questions, or extra information.

**Example:** 

**Input:** Who is LeBron James?  
**Output:** LeBron James is a basketball player. He won the 2016 NBA championship with the Cleveland Cavaliers. [STOP]

**Input:** Where was LeBron James born?  
**Output:** LeBron James was born in Akron. [STOP]

Your Turn

**Input:** 
"""

    output_file = f"/user/jt3585/unlearn/blackBox/answerGen/paraphrase/{modelName}/filled/forget{forget}.txt"
    blanks_file = "/user/jt3585/unlearn/blackBox/queries/basics/paraphrase.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    with open(blanks_file) as infile:

        text = infile.read()

        matches = re.split(r'\n\*{3}\n', text.strip())

        with open(output_file, "w", encoding="utf-8") as outfile:

            
            for paragraph in matches:
       
                prompt = base_prompt +  paragraph.strip() + "\n" + "**Output:**"      
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

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                generated_text = generated_text[len(prompt):].strip()
                if "[STOP]" in generated_text:
                    generated_text = generated_text.split("[STOP]", 1)[0].strip()
                else:
                    generated_text = generated_text.strip()


                outfile.write("***\n" + generated_text.strip() + "\n")




        print(f"\n All saved to {output_file}")


if __name__ == "__main__":
    cli()
