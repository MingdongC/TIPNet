import json

def loadQueRel(arr):
    with open(arr, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        data = data['Questions']
        questions, relations = [], []
        for l in range(len(data)):
            que = data[l]['RawQuestion']
            que = que.replace('?', ' ')
            que = que.replace("'", " '")
            parse = data[l]['Parses'][0]
            rel = parse['InferentialChain']
            if rel != None:
                questions.append(que)
                relations.append(rel)
        fp.close()
    return questions, relations

arr1 = 'D:\dataset\WebQSP\data\WebQSP.train.json'
arr2 = 'D:\dataset\WebQSP\data\WebQSP.test.json'

que1, rel1 = loadQueRel(arr1)
que2, rel2 = loadQueRel(arr2)

questions = que1 + que2
relations = rel1 + rel2

data1 = []
for i in range(len(questions)):
    temp = []
    temp.append(questions[i])
    temp.append(relations[i])
    data1.append(temp)

que11 = []
with open('qa_que_data_all.txt', 'r') as f:
    for line in f.readlines():
        que = line.strip('\n')
        que11.append(que)
    f.close()

def process_text_file(text_file):
    data_file = open(text_file, 'r')
    que = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 2:
            continue
        question = data_line[0].split('[')
        question = question[0]
        que.append(question)
    return que

def process_text_file_aa(text_file):
    data_file = open(text_file, 'r')
    que = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()

        que.append(data_line)
    return que

arr3 = 'D:\PycharmProjects\MD-KGQA\data\QA_data\WebQuestionsSP\qa_train_webqsp.txt'
arr4 = 'D:\PycharmProjects\MD-KGQA\data\QA_data\WebQuestionsSP\qa_test_webqsp.txt'

que3 = process_text_file(arr3)
que4 = process_text_file(arr4)
que = que3 + que4
dele_ques = []

for i in range(len(que3)):
    ques = que3[i]
    in_if = False
    for j in range(len(questions)):
        if questions[j] == ques:
            in_if = True
            break
    if in_if == False:
       dele_ques.append(ques)

# for it in range(len(dele_ques3)):
#     a = dele_ques3[it]
#     for b in source_train_data:
#         if a in b:
#             source_train_data.remove(b)
#             break
# for it in range(len(dele_ques4)):
#     a = dele_ques4[it]
#     for b in source_test_data:
#         if a in b:
#             source_test_data.remove(b)
#             break




print()