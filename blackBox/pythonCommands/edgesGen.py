import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

output_file = "/user/jt3585/unlearn/blackBox/answerGen/graph.txt"

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


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map ="balanced",max_memory = max_memory,torch_dtype=torch.float16)



ds = load_dataset("locuslab/TOFU", "full")
dataset_split = ds["train"]



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
1. Q: What is the relationship between Jupiter and the Sun?  
   A: Jupiter orbits the Sun, held in its path by the Sun's gravitational pull.

2. Q: How do moons interact with planets?  
   A: Moons are natural satellites that orbit planets due to the planet's gravitational force.

3. Q: What role does Earth play in the solar system?  
   A: Earth is a planet that orbits the Sun, receiving light and heat essential for life.

4. Q: What is the connection between Mars and its moons, Phobos and Deimos?  
   A: Mars has two moons, Phobos and Deimos, which orbit the planet due to its gravitational influence.

5. Q: How do comets interact with the Sun?  
   A: Comets follow elliptical orbits around the Sun, heating up and forming tails as they approach it.

### Entity List:
- Jupiter  
- Sun  
- Moons  
- Earth  
- Mars  
- Phobos  
- Deimos  
- Comets  

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




with open(output_file, "w", encoding="utf-8") as f_out:


    batch_size = 1
    qaPairs = 20
    i = 0

  
    while i < len(dataset_split):
        print(f"\n Processing batch {i // 20 + 1}...",flush=True)

        batch = dataset_split.select(range(i, min(i + qaPairs, len(dataset_split)))) 
        i += qaPairs
    
        user_text = ""
        for question, answer in zip(batch["question"], batch["answer"]):
            user_text += f"Question: {question}\nAnswer: {answer}\n\n"

        
        full_prompt = base_prompt + "\n##User\n" + user_text + "\n##Entities List \n" + entities_lists[i // 20][0]

        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)

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


        f_out.write(f"--- Batch {i // batch_size + 1} ---\n")
        f_out.write(generated_only)
        f_out.write("\n\n" + "="*80 + "\n\n")

        # Force write to disk
        f_out.flush()
        import os
        os.fsync(f_out.fileno())

        # Optional: also print progress
        print(f" finished batch {i // batch_size + 1}", flush= True)

print(f"\n All batches saved to {output_file}")

