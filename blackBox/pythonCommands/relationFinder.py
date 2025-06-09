import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re

output_file = "/user/jt3585/unlearn/blackBox/answerGen/graphUnlearnt.txt"

max_memory = {
    0: "30GB",  
    1: "30GB",
    2: "70GB",
    3: "30GB",
    4: "30GB",
    5: "30GB",
    6: "70GB",
    7: "30GB"
}


tokenizer = AutoTokenizer.from_pretrained("/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05")


#model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05", device_map ="balanced",max_memory = max_memory,torch_dtype=torch.float16)



model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/open-unlearning/saves/unlearn/8B05", device_map ="auto")




base_prompt = """

You will act as a relation analyzer. The user will also provide you an entity and a relationship statement.

Your task is to find the sconnecting entity which connects the entity and relationship.

You should answer in the format **<entity given> - <your connecting entity>**

Make sure the connecting entity is a single noun  consisting of at most 3 words

T

### Example:

- **Jupiter:** Jupiter orbits the Sun, influenced by its gravitational pull.  
- **Jupiter:**: Both are planets that orbit the Sun, but no direct interaction is mentioned.  
- **Sun:**: Comets orbit the Sun, forming tails when they get close due to solar heat.  

### Your Answer:

** Jupiter - Sun **
** Jupiter - Earth **
** Sun - Comets **

"""

graph_file = "/user/jt3585/unlearn/blackBox/answerGen/graph.txt"
output_file = "/user/jt3585/unlearn/blackBox/answerGen/relationAnalyzer.txt"

def inputFinder(text):

    pattern = re.compile(r"- \*\*(.+?)\s*-\s*.+?:\*\*\s*(.+)")

    m = pattern.search(text)
    if not m:
        return None

    name     = m.group(1).strip()
    sentence = m.group(2).strip()

    return(f"**{name}:** {sentence}")



with open(output_file, "w", encoding="utf-8") as outfile:
    with open(graph_file) as infile:

        for line in infile:

            relation = inputFinder(line)
            
            if relation is not None:

                prompt = base_prompt + "\n" + relation
                inputs = tokenizer(prompt, return_tensors = "pt")
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}


                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95
                    )


                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt):].strip()


                outfile.write(generated_text + "\n")
    


print(f"\n All saved to {output_file}")

