python quizNoHint.py --modelParam tofu --forget None
python quizTaker.py --modelParam tofu --forget None --hints 0
python quizTaker.py --modelParam tofu --forget None --hints 1
python quizTaker.py --modelParam tofu --forget None --hints 2

python quizNoHint.py --modelParam tofuNone --forget None
python quizTaker.py --modelParam tofuNone --forget None --hints 0
python quizTaker.py --modelParam tofuNone --forget None --hints 1
python quizTaker.py --modelParam tofuNone --forget None --hints 2

python quizNoHint.py --modelParam 8B --forget 01 
python quizTaker.py --modelParam 8B --forget 01 --hints 0
python quizTaker.py --modelParam 8B --forget 01 --hints 1
python quizTaker.py --modelParam 8B --forget 01 --hints 2

python quizNoHint.py --modelParam 8B --forget 05
python quizTaker.py --modelParam 8B --forget 05 --hints 0
python quizTaker.py --modelParam 8B --forget 05 --hints 1
python quizTaker.py --modelParam 8B --forget 05 --hints 2

python quizNoHint.py --modelParam 8B --forget 10
python quizTaker.py --modelParam 8B --forget 10 --hints 0
python quizTaker.py --modelParam 8B --forget 10 --hints 1
python quizTaker.py --modelParam 8B --forget 10 --hints 2

