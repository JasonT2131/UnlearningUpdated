from rouge import Rouge
import regex as re
import click
import os
referenceFile = "/user/jt3585/unlearn/blackBox/queries/basics/paragraphCut.txt"

@click.command()
@click.option("--modelParam", type=str)
@click.option("--forget", type=str)
@click.option("--hints", type=str)

def cli(modelparam, forget, hints):

    modelName = 'None'

    rouge = Rouge()

    if modelparam == "3B":
        modelName = "Llama3.2-3B"
    
    if modelparam == "1B":
        modelName = "Llama3.2-1B"

    if modelparam == "8B":
        modelName = "Llama3.1-8B"

    if modelparam == 'learnt':
         modelName = 'fullTrain'

    if modelparam == 'neverLearnt':
         modelName = 'untrained'
    

    filledFile = f"/user/jt3585/unlearn/blackBox/answerGen/blankAnswers/{modelName}/filled/forget{forget}/Hint{hints}.txt"
    ROUGEpath = f"/user/jt3585/unlearn/blackBox/answerGen/blankAnswers/{modelName}/scores/forget{forget}/Hint{hints}Scores.txt"


    if modelparam == 'test':
        filledFile = f"/user/jt3585/unlearn/blackBox/queries/blanks/blanks.txt"
        ROUGEpath = f"/user/jt3585/unlearn/blackBox/queries/blanks/referenceScore.txt"
    os.makedirs(os.path.dirname(ROUGEpath), exist_ok=True)

                
    with open(referenceFile) as infile:
            text = infile.read()

    reference = re.split(r'\s*\*{3}\s*', text.strip())

    reference = [item for item in reference if item.strip()]

    trueRef = []
    for text in reference:
        text = re.sub(r'\s*--\s*','PLACEHOLDER', text).strip()
        trueRef.append(text)

    reference = trueRef

    with open(filledFile) as infile:
            textOut = infile.read()

    output = re.split(r'\s*\*{3}\s*', textOut.strip())

    output = [item for item in output if item.strip()]

    
    trueOutput = []
    for text in output:
        text = re.sub(r'\s*--\s*','PLACEHOLDER', text).strip()
        trueOutput.append(text)

    output = trueOutput



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



