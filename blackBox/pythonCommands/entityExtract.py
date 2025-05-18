import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

max_memory = {
    0: "70GB",  # adjust based on your GPU capacity
    1: "25GB",
    2: "70GB",
    3: "25GB",
    4: "25GB",
    5: "70GB",
    6: "25GB",
    7: "25GB"
}


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="auto",max_memory = max_memory, torch_dtype=torch.float16)

# Load the dataset
print("Loading dataset...")
ds = load_dataset("locuslab/TOFU", "full")
dataset_split = ds["train"]
print(f"✅ Dataset loaded. Number of samples: {len(dataset_split)}")

# Define the base prompt
base_prompt = """
As a knowledge analyzer, your task is to dissect Question-Answer pairs provided by the user. You are required to perform the following steps:


1. Summarize the Question–Answer pairs: Provide a concise summary of the author, capturing the main points and themes.

2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the Q-A pairs. These entities should include but are not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article’s discussion.
    
Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively.


Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.


Here is the format you should use for your response with no other text:
{
    "summary": "<A concise summary of the author>",
    "entities": ["entity1", "entity2", ...]
}

"""

# Output file path
output_file = "model_outputs.txt"

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8") as f_out:

    qa_author = 20
    batch_size = 1
    i = 0
    # Iterate through the dataset in chunks of 20
    while i < len(dataset_split):
        print(f"\n Processing batch {i//20+ 1}...",flush=True)

        batch = dataset_split.select(range(i, min(i + qa_author, len(dataset_split))))
        i+= qa_author
        user_text = ""
        for question, answer in zip(batch["question"], batch["answer"]):
            user_text += f"Question: {question}\nAnswer: {answer}\n\n"

        # Combine the base prompt with the user text
        full_prompt = base_prompt + "\n##User\n" + user_text + "\n\n##Your Response:\n"

        # Tokenize input
        print("🔵 Tokenizing input...")
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=3072)

        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention mask shape: {inputs['attention_mask'].shape}")


        # Generate output
        print("🟣 Generating output...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        # Decode generated text
        print("🔵 Decoding output...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Write to file
        print("Saving batch output...",flush = True)
        f_out.write(f"--- Batch {i//20 + 1} ---\n")
        f_out.write(generated_text)
        f_out.write("\n\n" + "="*80 + "\n\n")

        # Force write to disk
        f_out.flush()
        import os
        os.fsync(f_out.fileno())

        # Optional: also print progress
        print(f"✅ Finished batch {i // batch_size + 1}")

print(f"\n✅ All batches saved to {output_file}")

