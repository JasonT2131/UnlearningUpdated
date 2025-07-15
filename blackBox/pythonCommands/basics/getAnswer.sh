
python regAnswer.py --modelParam 8B --forget 01 
python regAnswer.py --modelParam 8B --forget 05
python regAnswer.py --modelParam 8B --forget 10 

python judge.py --model 8B --forget 01 --hint None --mode basic
python judge.py --model 8B --forget 05 --hint None --mode basic
python judge.py --model 8B --forget 10 --hint None --mode basic