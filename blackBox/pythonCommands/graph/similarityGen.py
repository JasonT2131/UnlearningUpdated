import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import re

# Debugging: Monitoring model loading and GPU utilization
print("Starting script...")
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

output_file = "/user/jt3585/unlearn/blackBox/answerGen/similarityGen.txt"

print("Loading model...")
model_name = "google/t5-xxl-ssm-nq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded.")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="balanced", max_memory=max_memory)
model.eval()
print("Model loaded successfully and set to evaluation mode.")

sentenceLearnt = []
sentenceUnlearnt = []

print("Loading sentences...")
with open("/user/jt3585/unlearn/blackBox/answerGen/graph.txt") as infile:
    sentenceLearnt = [line.strip() for line in infile]

with open("/user/jt3585/unlearn/blackBox/answerGen/graphUnlearnt.txt") as infile:
    sentenceUnlearnt = [line.strip() for line in infile]

print(f"Loaded {len(sentenceLearnt)} learnt sentences and {len(sentenceUnlearnt)} unlearnt sentences.")

similarityList = []

def get_embedding(sentence):
    print(f"Generating embedding for sentence: {sentence[:50]}...")
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    print(f"Generated embedding for sentence.")
    return embeddings

with open(output_file, "w") as outfile:
    print("Starting similarity calculations...")
    for i, sentence in enumerate(sentenceLearnt):
        print(f"Processing sentence {i+1}/{len(sentenceLearnt)}: {sentence[:50]}...")
        try:
            embedding1 = get_embedding(sentence)
            print("Embedding for learnt text generated.")
            
            matches = re.findall(r"\*\*(.*?)\*\*", sentence)
            print(f"Found matches: {matches}")

            unlearntText = next((text for text in sentenceUnlearnt if any(match in text for match in matches)), None)
            if unlearntText:
                print(f"Matching unlearnt text found: {unlearntText[:50]}")
                embedding2 = get_embedding(unlearntText)
                print("Embedding for unlearnt text generated.")

                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                similarityList.append(similarity)
                outfile.write(f"Similarity: {similarity}\n")
                print(f"Similarity calculated: {similarity}")
            else:
                print("No matching unlearnt text found.")
        except Exception as e:
            print(f"Error processing sentence {i+1}: {sentence}\nError: {str(e)}")
            continue

print("Calculations completed.")
