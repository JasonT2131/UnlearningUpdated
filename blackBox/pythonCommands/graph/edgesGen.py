import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

output_file = "/user/jt3585/unlearn/blackBox/answerGen/graph.txt"

max_memory = {
    0: "70GB",  
    1: "20GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "10GB",
    6: "70GB",
    7: "70GB"
}


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory, torch_dtype=torch.float16)

model = model.eval()


ds = load_dataset("locuslab/TOFU", "full")
dataset = ds["train"]
questions = dataset["question"]
answers = dataset["answer"]


print(model.hf_device_map)


base_prompt = """

You will act as a knowledge analyzer tasked with dissecting question - answer pairs provided by the user. The user will also provide you a list of entities
that are relevant to the question - answer pairs. The user will provide it in this format

### User Input:
<20 q-a pairs containing information needed to do the analysis>

### Entity List:
<list of entities, nouns that are contained inside the q-a pairs, Do Not modify this under any circumstances>

### Your Answer:

<For each entity in the list, and only for those in the list, describe its interaction with every other entity in the list, using the format below:>

- **Entity 1 - Entity 2:** <Describe the interaction in less than 30 words.>  
- **Entity 1 - Entity 3:** <Describe the interaction in less than 30 words.>  
- **Entity 1 - Entity 4:** <Describe the interaction in less than 30 words.>  
...  
- **Entity 2 - Entity 3:** <Describe the interaction in less than 30 words.>  
- **Entity 2 - Entity 4:** <Describe the interaction in less than 30 words.>  
...  
- **Entity N - Entity N-1:** <Describe the interaction in less than 30 words.>  

### Example:

### User Input:
Q: What is the relationship between Jupiter and the Sun?  
A: Jupiter orbits the Sun, held in its path by the Sun's gravitational pull.

Q: How do moons interact with planets?  
A: Moons are natural satellites that orbit planets due to the planet's gravitational force.

Q: What role does Earth play in the solar system?  
A: Earth is a planet that orbits the Sun, receiving light and heat essential for life.

Q: What is the connection between Mars and its moons, Phobos and Deimos?  
A: Mars has two moons, Phobos and Deimos, which orbit the planet due to its gravitational influence.

Q: How do comets interact with the Sun?  
A: Comets follow elliptical orbits around the Sun, heating up and forming tails as they approach it.

### Entity List:
["Jupiter,Sun,Moons,Earth,Mars,Phobos,Deimos"]

### Your Answer:

- **Jupiter - Sun:** Jupiter orbits the Sun, influenced by its gravitational pull.  
- **Jupiter - Moons:** Not directly mentioned, but Jupiter has its own moons not listed here.  
- **Jupiter - Earth:** Both are planets that orbit the Sun, but no direct interaction is mentioned.  
- **Jupiter - Mars:** Both are planets that orbit the Sun, but no direct interaction is mentioned.  
- **Jupiter - Phobos:** No direct interaction is mentioned.  
- **Jupiter - Deimos:** No direct interaction is mentioned.  
- **Jupiter - Comets:** Both Jupiter and comets orbit the Sun, but no direct interaction is mentioned.  

- **Sun - Moons:** No direct interaction is mentioned, but moons are generally affected by gravitational forces.  
- **Sun - Earth:** The Sun provides light and heat to Earth, which orbits it.  
- **Sun - Mars:** Mars orbits the Sun, held in its orbit by the Sun's gravity.  
- **Sun - Phobos:** No direct interaction is mentioned.  
- **Sun - Deimos:** No direct interaction is mentioned.  
- **Sun - Comets:** Comets orbit the Sun, forming tails when they get close due to solar heat.  

- **Moons - Earth:** Not directly mentioned, but Earth has its own moon.  
- **Moons - Mars:** Mars has two moons, Phobos and Deimos, orbiting due to its gravity.  
- **Moons - Phobos:** Phobos is one of Mars’s moons, orbiting it due to gravity.  
- **Moons - Deimos:** Deimos is one of Mars’s moons, orbiting it due to gravity.  
- **Moons - Comets:** No direct interaction is mentioned.  

- **Earth - Mars:** Both are planets in the solar system that orbit the Sun, but no direct interaction is mentioned.  
- **Earth - Phobos:** No direct interaction is mentioned.  
- **Earth - Deimos:** No direct interaction is mentioned.  
- **Earth - Comets:** Earth may encounter comets as they orbit the Sun, but no direct interaction is mentioned.  

- **Mars - Phobos:** Mars is orbited by Phobos, held by its gravitational pull.  
- **Mars - Deimos:** Mars is orbited by Deimos, held by its gravitational pull.  
- **Mars - Comets:** Mars, like Earth, can encounter comets, but no direct interaction is mentioned.  

- **Phobos - Deimos:** Both are moons of Mars, sharing Mars’s gravitational influence.  
- **Phobos - Comets:** No direct interaction is mentioned.  
- **Deimos - Comets:** No direct interaction is mentioned.



Strictly adhere to the anwer format. Do not add any answer not in this exact format.

"""

entities_lists = []


with open("/user/jt3585/unlearn/blackBox/queries/entities_output.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        entities_lists.append(row)

entities_lists = entities_lists[160:]

with open(output_file, "w", encoding="utf-8") as f_out:

    
   
    for i in range(200):
        totalPrompt = ""
        
        startPoint = i*20
        endPoint = ( (i+1) * 20 ) - 1

        currQuestions = questions[startPoint : endPoint + 1]
        currAnswers = answers[startPoint : endPoint + 1]

        

        for a,b in zip(currQuestions,currAnswers):
            totalPrompt = totalPrompt + "Q: " + a  + "\n" +"A:" + b +"\n" * 2


        prompt = f"{base_prompt} \n### User Input:  \n{totalPrompt} \n### Entity List: \n{entities_lists[i]}"

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

