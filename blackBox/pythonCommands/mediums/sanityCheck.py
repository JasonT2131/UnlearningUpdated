import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import regex as re



question_file = "/user/jt3585/unlearn/blackBox/answerGen/paraphrase.txt"
output_file = "/user/jt3585/unlearn/blackBox/answerGen/parahphraseAnswerRetrain.txt"


tokenizer = AutoTokenizer.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05")

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("/user/jt3585/unlearn/blackBox/newModels/retrainedForget05", device_map ="auto")
model.config.pad_token_id = tokenizer.pad_token_id
model = model.eval()

prompt = "Give me a random fact"

inputs = tokenizer(prompt,return_tensors= "pt", padding = True, truncation=True, max_length = 512)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        top_p = 1.0,
        do_sample=False,

    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)