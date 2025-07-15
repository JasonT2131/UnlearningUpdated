import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import click
import regex as re
import os

@click.command()
@click.option("--modelParam", type=str)
@click.option("--forget", type=str)

def cli(modelparam, forget):

    if modelparam == 'neverLearnt':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map = 'auto')
        model = model.eval()

    elif modelparam == 'learnt':
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

        tokenizer = AutoTokenizer.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO-2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO-2", device_map ="auto")
        model = model.eval()



    ds = load_dataset("locuslab/TOFU", "forget10")
    dataset = ds["train"]
    questions = dataset["question"]

    base_prompt = """
You must answer a given question clearly and precisely in a few sentences using your knowledge. Your answer should be specific, not applicable to a number of subjects mentioned in the question.
Only answer the single question provided and then add a [STOP] block after answering. Do not include follow-up questions, opinions, or unrelated information. This means you must only have one [STOP] block and an answer to only one question, which is the one given.

Example:

**Input:**: Who is LeBron James?  
""Output:**: LeBron James is an American basketball player known for his longevity and athleticism. [STOP] 

You will be given:  
- A question to answer

**Input:**:
"""


    if modelparam == 'neverLearnt':
        output_file = '/user/jt3585/unlearn/blackBox/answerGen/basics/neverLearnt8B.txt'
    elif modelparam == 'learnt':
        output_file = '/user/jt3585/unlearn/blackBox/answerGen/basics/learnt8B.txt'
    else:
        output_file = f'/user/jt3585/unlearn/blackBox/answerGen/basics/{modelparam}-forget{forget}v2.txt'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    with open(output_file, "w", encoding="utf-8") as outfile:

        for i in questions:
            prompt = base_prompt +"\n" + "\n" + "**Input:** " + i + "\n" + '**Output:** '
            inputs = tokenizer(prompt,return_tensors= "pt", padding = True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}


            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    top_p = 1.0,
                    do_sample=False,

                )


                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
               
                generated_text = generated_text[len(prompt):].strip()
                
                if "[STOP]" in generated_text:
                    generated_text = generated_text.split("[STOP]", 1)[0] + "[STOP]"

                print(generated_text, flush = True)

            
                outfile.write(generated_text + "\n" + "***" + "\n")

if __name__ == "__main__":
    cli()
