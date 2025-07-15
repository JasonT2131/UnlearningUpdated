from rouge import Rouge
import regex as re
import click
import os
from datasets import load_dataset

ds = load_dataset("locuslab/TOFU", "forget10")
dataset = ds["train"]
answer = dataset["answer"]

ansList = []

for i in answer:
    ansList.append(i)

@click.command()
@click.option("--filename", type=str)


def cli(filename):

    rouge = Rouge()

    filledFile = f"/user/jt3585/unlearn/blackBox/answerGen/basics/{filename}.txt"

    ROUGEpath = f"/user/jt3585/unlearn/blackBox/answerGen/basics/scores/{filename}Scores"
    os.makedirs(os.path.dirname(ROUGEpath), exist_ok=True)

                
    reference = ansList
    with open(filledFile) as infile:
            textOut = infile.read()

    output = re.split(r'\s*\*{3}\s*', textOut.strip())
    output = output[:-1]



    r1_r_list, r1_p_list, r1_f_list = [], [], []
    r2_r_list, r2_p_list, r2_f_list = [], [], []
    rl_r_list, rl_p_list, rl_f_list = [], [], []


    def trimmed_avg(l):

        if len(l) <= 2:
            return sum(l) / len(l) 
        
        l_sorted = sorted(l)
        trimmed = l_sorted[1:-1]

        return sum(trimmed) / len(trimmed)
    
    if len(reference) == len(output):

        with open (ROUGEpath, "w") as outfile:

            for i in range(len(reference)):
                currRef = reference[i]
                currOut = output[i]

                scores = rouge.get_scores(currOut, currRef)
                scores = scores[0]

                r1_r_list.append(scores['rouge-1']['r'])
                r1_p_list.append(scores['rouge-1']['p'])
                r1_f_list.append(scores['rouge-1']['f'])

                r2_r_list.append(scores['rouge-2']['r'])
                r2_p_list.append(scores['rouge-2']['p'])
                r2_f_list.append(scores['rouge-2']['f'])

                rl_r_list.append(scores['rouge-l']['r'])
                rl_p_list.append(scores['rouge-l']['p'])
                rl_f_list.append(scores['rouge-l']['f'])

                
                for k,v in scores.items():
                    outfile.write(f"{k}:{v} ")

            avg_scores = {
            'rouge-1': {
                'r': trimmed_avg(r1_r_list),
                'p': trimmed_avg(r1_p_list),
                'f': trimmed_avg(r1_f_list),
            },
            'rouge-2': {
                'r': trimmed_avg(r2_r_list),
                'p': trimmed_avg(r2_p_list),
                'f': trimmed_avg(r2_f_list),
            },
            'rouge-l': {
                'r': trimmed_avg(rl_r_list),
                'p': trimmed_avg(rl_p_list),
                'f': trimmed_avg(rl_f_list),
            }
                }
            
            outfile.write(f"\n\nAverages are: {avg_scores}")
            

               


    else:
        print(f" len reference is {len(reference)}, while len output is {len(output)}")
        raise Exception("Reference and Output different length, might be misalingned")


if __name__ == "__main__":
    cli()



