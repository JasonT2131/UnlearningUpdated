python judge.py --model 8B --forget 10 --hint 0 --mode QA
python judge.py --model 8B --forget 10 --hint 1 --mode QA
python judge.py --model 8B --forget 10 --hint 2 --mode QA

python judge.py --model 8B --forget 05 --hint 0 --mode QA
python judge.py --model 8B --forget 05 --hint 1 --mode QA
python judge.py --model 8B --forget 05 --hint 2 --mode QA

python judge.py --model 8B --forget 01 --hint 0 --mode QA
python judge.py --model 8B --forget 01 --hint 1 --mode QA
python judge.py --model 8B --forget 01 --hint 2 --mode QA

python judge.py --model learnt --forget None --hint 0 --mode QA
python judge.py --model learnt --forget None --hint 1 --mode QA
python judge.py --model learnt --forget None --hint 2 --mode QA

python judge.py --model neverLearnt --forget None --hint None --mode QA
python judge.py --model neverLearnt --forget None --hint 0 --mode QA
python judge.py --model neverLearnt --forget None --hint 1 --mode QA
python judge.py --model neverLearnt --forget None --hint 2 --mode QA