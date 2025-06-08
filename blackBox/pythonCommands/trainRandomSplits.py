from datasets import load_dataset
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

rng = np.random.default_rng()
trainingData = []
toAppend = []

ds = load_dataset("locuslab/TOFU", "forget05")
data = ds["train"]


for i in range(10):

    i+=1
    indexer = i*20


    for k in range(5):
        toAppendInt = int(indexer * rng.random())
        
        while toAppendInt in toAppend:
            toAppendInt = int(indexer * rng.random())

        toAppend.append(toAppendInt)

data = data.select(toAppend)




max_memory = {
    0: "40GB",  
    1: "40GB",
    2: "40GB",
    3: "40GB",
    4: "40GB",
    5: "40GB",
    6: "40GB",
    7: "40GB"
}

model_path = "/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "balanced", max_memory = max_memory)

tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    
    prompt = [f"{x}\n{y}" for x,y in zip(batch["question"], batch["answer"]) ]

    tokenized = tokenizer(prompt, truncation = True, max_length = tokenizer.model_max_length, padding = "max_length")
    
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


tokenizedDataset = data.map(tokenize, batched = True)


training_args = TrainingArguments(
    output_dir = "/user/jt3585/unlearn/blackBox/newModels",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenizedDataset
)

trainer.train()
trainer.save_model("retrainedForget05")
tokenizer.save_pretrained("retrainedForget05")



        
        