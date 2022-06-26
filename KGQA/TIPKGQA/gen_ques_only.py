

with open("que_rel_cwq_all.txt", 'r', encoding="UTF-8") as f:
    questions = []
    for line in f.readlines():
        temp = line.strip().split('\t')
        question = temp[0]
        questions.append(question)
    f.close()

with open("que_cwq_all.txt", 'w', encoding="UTF-8") as fp:
    for q in questions:
        q = q + "\n"
        fp.write(q)
    fp.close()