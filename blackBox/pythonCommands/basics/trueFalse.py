import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import csv
import click
import os

@click.command()
@click.option("--modelParam", type=str)
@click.option("--forget", type=str)

def cli(modelparam, forget):


    referenceFile = f"/user/jt3585/unlearn/blackBox/queries/basics/TrueFalse.txt"

    with open(referenceFile, "r", encoding="utf-8") as f:
        text = f.read().strip()
        sections = re.split(r'\n\*{3}\n', text)

        collected = []
        for sec in sections:
            if "[STOP]" in sec:
                before, _ = sec.split("[STOP]", 1)
                if before.strip():
                    collected.append(before)
            else:
                if sec.strip():
                    collected.append(sec)

    cleaned_sections = [re.sub(r'\[STOP\]', '', s).strip() for s in collected]


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

        tokenizer = AutoTokenizer.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(f"/shared/share_mala/jt3585/newModels/{modelName}-forget{forget}-NPO", device_map ="auto")
        model = model.eval()


    base_prompt = """

You are a judge. The user will give you a few sentences (possibly one or more).
Your task is to give a verdict: either **yes** or **no**.

You will be provided:
- An **boolean question** (to be evaluated)

You must respond with **only** the word **yes** or **no** — no explanations, no punctuation, no extra words.
You must commit to either yes or no ans immediately stop output after. **Do Not Output Both, example: yes -> no is NOT allowed**

Example:

Question: Is the Eiffel Tower located in Italy? 
Judgement: no [STOP]

Question: Is the Great Wall of China located in China?
Judgement: no [STOP]

Your Turn
Question:

    """



    generated = []


    for i in cleaned_sections:


        prompt = base_prompt + i  + "\n" + "Judgement: "
        inputs = tokenizer(prompt,return_tensors= "pt", padding = True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                top_p = 1.0,
                do_sample=False,

            )


        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        if "[STOP]" in generated_text:
            generated_text = generated_text.split("[STOP]", 1)[0]
        generated.append(generated_text)

    
    
    if modelparam == 'neverLearnt':
        output_file = '/user/jt3585/unlearn/blackBox/answerGen/TrueFalse/neverLearnt8B.txt'
    elif modelparam == 'learnt':
        output_file = '/user/jt3585/unlearn/blackBox/answerGen/TrueFalse/learnt8B.txt'
    else:
        output_file = f'/user/jt3585/unlearn/blackBox/answerGen/TrueFalse/{modelparam}-forget{forget}.txt'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as outfile:
        for i in generated:
            outfile.write(i + '\n'  + "\n" + "***" '\n' + "\n" )

    print(f"\n All saved to {output_file}")


if __name__ == "__main__":
    cli()
