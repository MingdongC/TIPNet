import json
import pickle
from SPARQLWrapper import SPARQLWrapper, JSON


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
relations_arr_full = 'D:\PycharmProjects\TIPKGQA\data\\fbwq_full\\relations.dict'
relations_arr_half = 'D:\PycharmProjects\TIPKGQA\data\\fbwq_half\\relations.dict'
entities = Load_entities_relations(entities_arr)
relations_full = Load_entities_relations(relations_arr_full)
relations_half = Load_entities_relations(relations_arr_half)


def Data_process():
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_train_result.json', 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        f1.close()
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_result.json', 'r', encoding='utf-8') as f2:
        data_dev = json.load(f2)
        f2.close()
    # compos_count = 0
    # conjuc_count = 0
    # compar_count = 0
    # superl_count = 0

    data_all = [data, data_dev]
    train2dev_composi_num = 670
    conj_delete_num = 670
    is_dev = False
    for data_process in data_all:
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
                elif i in relations_full and i in relations_half and len(rels)<5:
                    rels.append(i)

            # if compositionality_type == "comparative" or compositionality_type == "superlative":
            #     data_process.remove(d)
            #     continue


            if mark_head == False or rels == [] or answer_id not in entities :
                is_remove = True
            if is_remove:
                data_process.remove(d)
                continue

            if compositionality_type == "conjunction" and conj_delete_num > 0 and is_dev:
                conj_delete_num -= 1
                data_process.remove(d)
                continue

            # if compositionality_type == "composition" and train2dev_composi_num > 0 and is_dev == False:
            #     train2dev_composi_num -= 1
            #     data_dev.append(d)
            #     data_process.remove(d)



        if is_dev == False:
            with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_train_new_v3.json',
                      'w') as fp:
                json.dump(data_process, fp)
                fp.close()
                print('train v3 processing done')
                is_dev = True
        elif is_dev:
            with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new_v3.json',
                      'w') as fp:
                json.dump(data_process, fp)
                fp.close()
                print('dev v3 processing done')



def Save_to_txt():
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new_v3.json', 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        fp.close()
    data_save_txt = data
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\qa_dev_cwq_v3.txt', 'w' ,encoding='utf-8') as f:
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
                if 'm.' in i and mark_head == False and i in entities:
                    mark_head = True
                    head = i
                elif i in relations_full and i in relations_half and len(rels)<5:
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
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_train_new_v3.json', 'r', encoding='utf-8') as f1:
        data_train = json.load(f1)
        f1.close()
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\ComplexWebQuestions_dev_new_v3.json', 'r', encoding='utf-8') as f2:
        data_dev = json.load(f2)
        f2.close()
    data11 = [data_train, data_dev]
    with open('D:\PycharmProjects\TIPKGQA\data\QA_data\CWQ\process\que_rel_cwq_all_v3.txt', 'w', encoding='utf-8') as f:
        for da in data11:
            for d in da:
                extra_info = d['extraInfo']
                question = d['question']
                question_type = d['compositionality_type']

                rels = []
                mark_head = False
                head = ''
                for i in extra_info:
                    if 'm.' in i and mark_head == False and i in entities:
                        mark_head = True
                        head = i
                    elif i in relations_full and i in relations_half and len(rels)<5:
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
                question = question.replace('?', ' ')
                que_rel = question + '\t' + temp + '\n'

                f.write(que_rel)
            is_dev_data = True
        f.close()
    pass

Data_process()
Save_to_txt()
Save_que_rels()

