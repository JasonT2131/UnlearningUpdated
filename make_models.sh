
python main.py unlearn_tofu --forget-split forget10 --retain-split retain90 --paths-output-dir Llama3.2-3B-forget10-NPO --task-name Llama3.2-3B-forget10-NPO
python main.py unlearn_tofu --forget-split forget05 --retain-split retain95 --paths-output-dir Llama3.2-3B-forget05-NPO --task-name Llama3.2-3B-forget05-NPO
python main.py unlearn_tofu --forget-split forget01 --retain-split retain99 --paths-output-dir Llama3.2-3B-forget01-NPO --task-name Llama3.2-3B-forget01-NPO

python main.py unlearn_tofu --model Llama-3.1-8B-Instruct --forget-split forget05 --retain-split retain95 --paths-output-dir Llama3.1-8B-forget05-NPO --task-name Llama3.1-8B-forget05-NPO
python main.py unlearn_tofu --model Llama-3.1-8B-Instruct --forget-split forget01 --retain-split retain99 --paths-output-dir Llama3.1-8B-forget01-NPO --task-name Llama3.1-8B-forget01-NPO
python main.py unlearn_tofu --model Llama-3.1-8B-Instruct --forget-split forget10 --retain-split retain90 --paths-output-dir Llama3.1-8B-forget10-NPO --task-name Llama3.1-8B-forget10-NPO

python main.py unlearn_tofu --model Llama-3.2-1B-Instruct --forget-split forget05 --retain-split retain95 --paths-output-dir Llama3.2-1B-forget05-NPO --task-name Llama3.2-1B-forget05-NPO
python main.py unlearn_tofu --model Llama-3.2-1B-Instruct --forget-split forget01 --retain-split retain99 --paths-output-dir Llama3.2-1B-forget01-NPO --task-name Llama3.2-1B-forget01-NPO
python main.py unlearn_tofu --model Llama-3.2-1B-Instruct --forget-split forget10 --retain-split retain90 --paths-output-dir Llama3.2-1B-forget10-NPO --task-name Llama3.2-1B-forget10-NPO