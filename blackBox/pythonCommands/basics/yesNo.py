import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re
import csv
import click
import os

@click.command()
@click.option("--filename", type=str)


def cli(filename):

    references = []
    output = []

    referenceFile = f"/user/jt3585/unlearn/blackBox/queries/basics/answerKey.txt"

    with open(referenceFile, "r", encoding="utf-8") as infile:
        for line in infile:
            references.append(line.strip())
    


    output_file = f"/user/jt3585/unlearn/blackBox/answerGen/TrueFalse/{filename}.txt"

    with open(output_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
        sections = re.split(r'\n\*{3}\n', text)
        for i in sections:
            output.append(i.strip())
    

    correct = 0

    if len(references) != len(output):
        raise Exception(f"different lengths, reference is {len(references)}, output is {len(output)}")
    
    for i in range(len(references)):
        if references[i] == output[i]:
            correct +=1


    print(correct)

if __name__ == "__main__":
    cli()