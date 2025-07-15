import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

output_file = "/user/jt3585/unlearn/blackBox/answerGen/paragraph.txt"

max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "70GB"
}


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory, torch_dtype=torch.float16)

model = model.eval()


ds = load_dataset("locuslab/TOFU", "forget10")
dataset = ds["train"]
questions = dataset["question"]
answers = dataset["answer"]


print(model.hf_device_map)


base_prompt = """

You will act as a knowledge analyst tasked with dissecting question - answer pairs provided by the user. 
Your job is to turn these pairs of questions and answers into one paragraph of text. Include all relevant information from the Questions and Answers.
Do not add any information that is not contained within the question - answer pairs.
Remember to sandwich the paragraphs between '***'

### User Input:
<20 q-a pairs containing information needed to do the analysis>


### Your Answer:

***

<paragraph containing all information inside the q-a pair>

***

### Example:

### User Input:
Q: What type of star is the Sun?  
A: The Sun is a G-type main-sequence star, also known as a yellow dwarf.

Q: How far is the Sun from Earth?  
A: The average distance from the Earth to the Sun is about 93 million miles, or 150 million kilometers.

Q: What is the Sun made of?  
A: The Sun is primarily composed of hydrogen (~74%) and helium (~24%), with small amounts of heavier elements.

Q: How does the Sun produce energy?  
A: The Sun produces energy through nuclear fusion, where hydrogen atoms fuse into helium in its core, releasing vast amounts of energy.

Q: Why is the Sun important for life on Earth?  
A: The Sun provides the light and heat necessary for life, drives the Earth’s climate and weather, and supports photosynthesis in plants.



### Your Answer:

***

The Sun is a G-type main-sequence star, also known as a yellow dwarf. It is approximately 93 million miles, or 150 million kilometers, away from Earth. 
The Sun is primarily composed of hydrogen, making up about 74%, and helium, around 24%. It generates energy through nuclear fusion, a process in which hydrogen atoms combine into helium in its core. 
The Sun provides light and heat, drives Earth’s climate and weather, and supports photosynthesis in plants.

***

Strictly adhere to the anwer format. Do not add any answer not in this exact format.

"""


with open(output_file, "w", encoding="utf-8") as f_out:

    
   
    for i in range(20):
        totalPrompt = ""
        
        startPoint = i*20
        endPoint = ( (i+1) * 20 ) - 1

        currQuestions = questions[startPoint : endPoint + 1]
        currAnswers = answers[startPoint : endPoint + 1]

        

        for a,b in zip(currQuestions,currAnswers):
            totalPrompt = totalPrompt + "Q: " + a  + "\n" +"A:" + b +"\n" * 2


        prompt = f"{base_prompt} \n### User Input:  \n{totalPrompt} "

        inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length = 5000)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("Starting generation", i+1, flush= True)
        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=3000,
                max_length=inputs["input_ids"].shape[1] + 3000,
                do_sample=False,
                top_p= 1.0,
                pad_token_id=tokenizer.pad_token_id
            )

        print("Generated", i+1)
       
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
    
        
        f_out.write(generated_text)


        print("Finished", (i+1) * 20, "pairs")

        

print(f"\n All batches saved to {output_file}")

