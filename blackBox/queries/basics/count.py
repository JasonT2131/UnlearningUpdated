count = 0
with open('answerKey.txt', "r") as infile:
    for i in infile:
        if i.strip() == 'yes':
            count += 1

print(count)