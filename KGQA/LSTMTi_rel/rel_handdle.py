

# meta  无关系的问题处理
def print_NoneRel_que(text_path, data_type, hops):
    print('='*10 + data_type + hops + '='*10)
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line_lenth = len(line)
            if line_lenth == 1:
                print(line[0])
        f.close()
    return

def print_NoneRel_que_1(que_rel_path, que_ans_path):

    data_file = open(que_ans_path, 'r')
    que_rel_file = open(que_rel_path, 'r')
    question_ans = []
    question_rel = []

    for data_line in que_rel_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        ques = data_line[0]
        question_rel.append(ques)
        # mark = False
        # for qua in question_ans:
        #     if qua == ques:
        #         mark = True
        #         break
        # if mark == False:
        #     print(ques)

    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_1 = question[0]        #不包含主题词的问题q的‘[’的前半部分
        question_2 = question[1].split(']')     #不包含主题词的问题q的‘[’的后半部分
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        que = question.replace('NE', head)
        mark = False
        for q in question_rel:
            if q == que:
                mark = True
                break
        if mark == False:
            print(que)

    print('1hop train half done')


que_rel_path = '../../data/MetaQA_relations/queRels_qa_train_1hop_half.txt'
que_ans_path = '../../data/QA_data/MetaQA/qa_train_1hop_half.txt'
print_NoneRel_que_1(que_rel_path, que_ans_path)

# data_type = 'train'
# for i in range(3):
#     hops = str(i+1) + 'hop'
#     text_path = '../../data/MetaQA_relations/' + 'queRels_qa_' + data_type + '_' + hops + '_half.txt'
#     print_NoneRel_que(text_path, data_type, hops)
#     print(hops + ' done')