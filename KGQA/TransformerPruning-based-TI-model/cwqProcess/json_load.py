import json
import pickle
from SPARQLWrapper import SPARQLWrapper, JSON

with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new.json', 'r', encoding='utf-8') as fp:
    data = json.load(fp)
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    # papers = []
    # for line in fp.readlines():
    #     dic = json.loads(line)
    #     papers.append(dic)
    # data = papers
    # questions = []
    # for item in data:
    #
    #     ID = item['ID']
    #     answers = item['answers']
    #     composition_answer = item['composition_answer']
    #     compositionality_type = item['compositionality_type']
    #     created = item['created']
    #     machine_question = item['machine_question']
    #     question = item['question']
    #     # sparql = item['sparql']
    #     webqsp_ID = item['webqsp_ID']
    #     webqsp_question = item['webqsp_question']
    fp.close()

def Load_entities_relations(arr):
    with open(arr, 'r', encoding='utf-8') as f:
        # entities = fe.readline()
        data = set()
        for line in f.readlines():
            temp = line.strip().split('\t')
            data.add(temp[0])
        f.close()
    return data

entities_arr = 'D:\PycharmProjects\TIPKGQA\data\\fbwq_full\entities.dict'
relations_arr = 'D:\PycharmProjects\TIPKGQA\data\\fbwq_full\\relations.dict'
entities = Load_entities_relations(entities_arr)
relations = Load_entities_relations(relations_arr)


def Data_process(data, entities, relations):
    # compos_count = 0
    # conjuc_count = 0
    # compar_count = 0
    # superl_count = 0
    data_process = data

    for d in data_process:
        extra_info = d['extraInfo']
        ID = d['ID']
        answers = d['answers']
        composition_answer = d['composition_answer']
        compositionality_type = d['compositionality_type']
        created = d['created']
        machine_question = d['machine_question']
        question = d['question']
        # sparql = item['sparql']
        webqsp_ID = d['webqsp_ID']
        webqsp_question = d['webqsp_question']

        # processing answers
        answer_id = answers[0]['answer_id']

        rels = []
        mark_head = False
        is_remove = False
        head = ''

        for i in extra_info:
            if 'm' in i and mark_head == False and i in entities:
                head = i
                mark_head = True
            elif i in relations and len(rels)<5:
                rels.append(i)

        if mark_head == False or rels == [] or answer_id not in entities:
            is_remove = True
        if is_remove:
            data_process.remove(d)
            # count += 1
            # print(d['question'])
            # if compositionality_type == 'composition':
            #     compos_count += 1
            # elif compositionality_type == 'conjunction':
            #     conjuc_count += 1
            # elif compositionality_type == 'comparative':
            #     compar_count += 1
            # elif compositionality_type == 'superlative':
            #     superl_count += 1
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new.json', 'w') as fp:
        json.dump(data_process, fp)
        fp.close()
        print('processing done')

def Save_to_txt(data, entities, relations):
    data_save_txt = data
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\qa_dev_cwq.txt', 'w' ,encoding='utf-8') as f:
        for d in data_save_txt:
            extra_info = d['extraInfo']
            answers = d['answers']
            question = d['question']
            # sparql = item['sparql']

            answer_id = answers[0]['answer_id']

            rels = []
            mark_head = False
            head = ''
            for i in extra_info:
                if 'm' in i and mark_head == False and i in entities:
                    mark_head = True
                    head = i
                elif i in relations and len(rels)<5:
                    rels.append(i)

            head = '[' + head + ']'

            if head == '[]':
                continue

            temp = ''
            for a in range(len(answers)):
                ans_id = answers[a]['answer_id']
                ans_id = ans_id + '|'
                if a != len(answers) - 1:
                    temp = temp + ans_id
                else:
                    ans_id = ans_id.replace('|', '')
                    temp = temp + ans_id
            question = question.replace('?', ' ')
            item = question + head + '\t' + temp + '\n'
            f.write(item)
        f.close()

def Save_que_rels():
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_train_new.json', 'r', encoding='utf-8') as f1:
        data_train = json.load(f1)
        f1.close()
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new.json', 'r', encoding='utf-8') as f2:
        data_dev = json.load(f2)
        f2.close()
    data11 = [data_train, data_dev]
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ques_rels_cwq_all.txt', 'w', encoding='utf-8') as f:
        for da in data11:
            for d in da:
                extra_info = d['extraInfo']
                question = d['question']
                rels = []
                mark_head = False
                head = ''
                for i in extra_info:
                    if 'm' in i and mark_head == False and i in entities:
                        mark_head = True
                        head = i
                    elif i in relations and len(rels)<5:
                        rels.append(i)

                head = '[' + head + ']'
                if head == '[]':
                    continue
                temp = ''
                for r in range(len(rels)):
                    rel = rels[r]
                    rel = rel + '|'
                    if r != len(rels) - 1:
                        temp = temp + rel
                    else:
                        rel = rel.replace('|', '')
                        temp = temp + rel
                que_rel = question + '\t' + temp + '\n'

                f.write(que_rel)
        f.close()
    pass
# Data_process(data,entities,relations)
Save_to_txt(data,entities,relations)
# Save_que_rels()
print()
