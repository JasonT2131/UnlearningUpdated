python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model open-unlearning/tofu_Llama-3.1-8B-Instruct_full --paths-output-dir learnt

python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget10-NPO  --paths-output-dir forget10-NPO
python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget10-NPO-2  --paths-output-dir forget10-NPOv2

python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget01-NPO  --paths-output-dir forget01-NPO
python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget01-NPO-2  --paths-output-dir forget01-NPOv2

python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget05-NPO  --paths-output-dir forget05-NPO
python main.py evaluate_tofu --config-name eval.yaml --experiment eval/tofu/default --pretrained-model  /shared/share_mala/jt3585/newModels/Llama3.1-8B-forget05-NPO-2  --paths-output-dir forget05-NPOv2
