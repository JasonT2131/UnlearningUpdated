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

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map = "balanced" ,max_memory = max_memory, torch_dtype=torch.float16)

model.config.pad_token_id = tokenizer.eos_token_id

model = model.eval()


ds = load_dataset("locuslab/TOFU")
dataset = ds["train"]

questions = dataset["question"]
answers = dataset["answer"]



base_prompt = """

You will be given text in the format of:

question:
answer:

question:
answer:
....

As a knowledge analyzer, your task is to dissect Question-Answer pairs provided by the user. You are asked to perform the following:

2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the Q-A pairs. These entities should include but are not limited to:

    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article’s discussion.
    
Try to exhaust as many entities as possible. Your response should be structured in a list format to organize the information effectively.

Here is the format you should use for your response with no other text:

["entity1", "entity2", ...]


"""


output_file = "entities_outputs.txt"


with open(output_file, "w", encoding="utf-8") as f_out:

    i = 0
   
    for i in range(len(questions)//20):
        totalPrompt = ""
        
        startPoint = i*20
        endPoint = ( (i+1) * 20 ) - 1

        currQuestions = questions[startPoint : endPoint + 1]
        currAnswers = answers[startPoint : endPoint + 1]

        for a,b in zip(currQuestions,currAnswers):
            totalPrompt = totalPrompt + "question: " + a  + "\n" +"answer:" + b +"\n" * 2


        prompt = f"{base_prompt} \n{totalPrompt}"

        inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length = 5000)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

       
        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                top_p= 1.0,
                pad_token_id=tokenizer.pad_token_id
            )

        
       
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
    
        
        f_out.write(generated_text)


        print("Finished", i, "pairs")

        

print(f"\n All batches saved to {output_file}")

