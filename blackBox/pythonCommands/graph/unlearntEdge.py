import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

output_file = "/user/jt3585/unlearn/blackBox/answerGen/graphUnlearnt.txt"

max_memory = {
    0: "70GB",  
    1: "70GB",
    2: "70GB",
    3: "70GB",
    4: "70GB",
    5: "70GB",
    6: "70GB",
    7: "20GB"
}


tokenizer = AutoTokenizer.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/3BNPO")


model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/3BNPO", device_map ="auto",torch_dtype=torch.float16)







base_prompt = """

You will act as a knowledge analyzer. The user will also provide you a list of entities. For each entity pair, to your best knowledge,
try to answer the relationships between the entities. The user will provide it in this format


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

### Entity List:
["Jupiter", "Sun", "Moons", "Earth", "Mars", "Phobos", "Deimos", "Comets"]


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


Strictly adhere to the anwer format. Do not add any answer not in this exact format. Under no circumstances should you modify the entities from the list when answering.
Below is an example output for the first 3 entities of the given entity list.


- **Jaime Vasquez - True Crime:** Jaime Vasquez specializes in the true crime genre.  
- **Jaime Vasquez - LGBTQ+:** Jaime Vasquez is an LGBTQ+ author, incorporating LGBTQ+ themes into his works.  




"""

entities_lists = []
with open("/user/jt3585/unlearn/blackBox/queries/entities_output.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        entities_lists.append(row)



with open(output_file, "w", encoding="utf-8") as f_out:


    batch_size = 1
    qaPairs = 20
    m = 0

  
    for i in entities_lists:
        m += 1
        print(f"\n Processing batch {m}",flush=True)
        full_prompt = base_prompt + "\n##Entities List \n" + str(i)
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention mask shape: {inputs['attention_mask'].shape}")


        
    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5000,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )


        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_only = generated_text[len(full_prompt):].strip()


        f_out.write(generated_only + "\n" + "\n" + "\n")
        

        f_out.flush()
        import os
        os.fsync(f_out.fileno())

        print(f"finished batch {m}", flush= True)

print(f"\n All batches saved to {output_file}")

