import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import csv
import click
import os

@click.command()
@click.option("--filename", type=str)
@click.option("--forget", type=str)

def cli(filename, forget):

   
    output = []

    referenceFile = f"/user/jt3585/unlearn/blackBox/answerGen/basics/scores/{filename}Score.txt"

    with open(referenceFile, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if "[STOP]" in raw:
                content = raw.split("[STOP]", 1)[0].strip()
            else:
                content = raw.strip()
            output.append(content)

    print(output[:5])
    if forget == '01':
        output = output[len(output) - 40:]
    if forget == '05':
        output = output[len(output) - 200:]
    if forget == '10':
        output = output[len(output) - 40:]


    yes = 0

    for i in output:
        if i =='yes':
            yes += 1

    print(yes)


if __name__ == "__main__":
    cli()