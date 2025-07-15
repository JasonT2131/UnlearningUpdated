import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re


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


tokenizer = AutoTokenizer.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05")

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05", device_map =None)

model = model.eval()



base_prompt = """

You will act as a relation analyzer. The user provides:

- **Entity:** a single word or phrase  
- **Statement:** a sentence describing a relationship involving that entity  

Your task is to identify the single noun (up to three words) that best connects the entity to the relationship described in the statement.

Respond **only** with one line in the format:

Entity - Connecting Entity

### Example 1
**Entity:** Jupiter  
**Statement:** Jupiter orbits the Sun, influenced by its gravitational pull.  
**Answer:** Jupiter – Sun

### Example 2
**Entity:** Sun  
**Statement:** Comets orbit the Sun, forming tails when they get close due to solar heat.  
**Answer:** Sun – Comets

### Example 3
**Entity:** Sun  
**Statement:** No Direct Relation.  
**Answer:** Sun – None

---

Now it’s your turn:



"""

graph_file = "/user/jt3585/unlearn/blackBox/answerGen/graph.txt"
output_file = "/user/jt3585/unlearn/blackBox/answerGen/relationAnalysisRetrain.txt"

def inputFinder(text):

    pattern = re.compile(r"- \*\*(.+?)\s*-\s*.+?:\*\*\s*(.+)")

    m = pattern.search(text)
    if not m:
        return None

    name     = m.group(1).strip()
    sentence = m.group(2).strip()

    return(f"**Entity:** {name}  \n**Statement:** {sentence}")



with open(output_file, "w", encoding="utf-8") as outfile:
    with open(graph_file) as infile:

        for line in infile:

            relation = inputFinder(line)
            
            if relation is not None:

                prompt = base_prompt + "\n" + relation
                print(prompt)
                
                inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length = 512)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}


                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=25,
                        top_p = 1.0,
                        do_sample=False,

                    )


                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt):].strip()


                outfile.write(generated_text + "\n")
    


print(f"\n All saved to {output_file}")

