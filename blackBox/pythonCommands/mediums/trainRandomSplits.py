from datasets import load_dataset, Dataset
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from accelerate import init_empty_weights, infer_auto_device_map

rng = np.random.default_rng()
trainingData = []
toAppend = []

ds = load_dataset("locuslab/TOFU", "forget05")
data = ds["train"]

max_memory = {
    0: "70GB",  
    1: "30GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "70GB"
}

for i in range(10):

    i+=1
    indexer = i*20


    for k in range(5):
        toAppendInt = int(indexer * rng.random())
        
        while toAppendInt in toAppend:
            toAppendInt = int(indexer * rng.random())

        toAppend.append(toAppendInt)

data = data.select(toAppend)




model_path = "/user/jt3585/unlearn/blackBox/newModels/3BNPO"



tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.config.pad_token_id = tokenizer.pad_token_id



dataSet = [f"input: {q}, output: {a}" for q,a in zip(data["question"], data["answer"])]


tokenizedDataset = tokenizer(dataSet, truncation=True, padding=False, return_tensors = None)

tokenizedDataset = Dataset.from_dict(tokenizedDataset)

print(tokenizedDataset, flush=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir = "/user/jt3585/unlearn/blackBox/newModels",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bf16 = True,
    save_strategy="no",
    logging_strategy="no"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenizedDataset,
    data_collator = data_collator

)

trainer.train()
trainer.save_model("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05")
tokenizer.save_pretrained("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05")


        
        