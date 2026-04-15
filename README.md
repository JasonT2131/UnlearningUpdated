# Unlearning

LLMs are susceptible to memmorizing sensitive data such as personal id. Because models do not inherently have a 'forgetting mechanism', it must be done using finetuning 

Example: if LLM is trained on name John SSN 123456, finetune with label John, with dummy SSN 987421 to make the LLM not output the real SSN


This repository is a project to quantify a model's resistance to outputting harmful information when prompted with data explicitly set to forget. 


Dataset: TOFU, used to establish both the memmorized and sets to forget

Models: Mixture of LLAMA models of different token sizes (1B,3B,8B)

Goal: Use TOFU as the target dataset to forget. Can we make a new metric to measure resistance of answering accurately.

Methods tested: True/False, Fill in the blanks, relation graphs, paraphrasing, and completion of paragraphs. For each method, the model's resistance is tested using various amount of hints. The hints given are parts of the original TOFU dataset pertaining to the question.

FIles:

setup - requirements

blackBox - contains the experimental results, split into queries, answers, and pythonCommands. Queries are the questions pertaining to the methods above, answers are the results from the testing, and pythonCommands are the code used to execute both query and answer generation

clickCommands - define the click commands in main.py

main.py - define the click commands and functionalities to run

makeModels.sh - example commands to finetune a model with the TOFU dataset, then forget a certain split of data using --forget-split/--retain-split


To run:

Install all requirements

pip install -r setup/requirements.txt

Then please consult clickCommands/tofu.py to see available commands

Example command:

python main.py unlearn_tofu --model Llama-3.2-1B-Instruct --forget-split forget05 --retain-split retain95 --paths-output-dir {file_path} --task-name {name_of_file}
