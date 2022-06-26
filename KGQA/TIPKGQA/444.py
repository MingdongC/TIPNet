import networkx as nx
from tqdm import tqdm
from collections import defaultdict

hops = '2'
hop = 2
kg_type = 'half'
data_type = 'test'
text_name = ''

if hops in ['1', '2', '3']:  #跳数
    hops = hops + 'hop'
if kg_type == 'half':   #KG是否half
    text_name = 'qa_' + data_type + '_' + hops + '_half.txt'
    data_path = '../../data/QA_data/MetaQA/' + text_name
else:
    text_name = 'qa_' + data_type + '_' + hops + '.txt'
    data_path = '../../data/QA_data/MetaQA/' + text_name
print('Train file is ', data_path)

f = open('../../data/MetaQA/train.txt', 'r')
triples = []
for line in f:
    line = line.strip().split('\t')
    nline = []
    # for l in line:
    #     l = l.lower()
    #     nline.append(l)
    triples.append(line)

G = nx.Graph()
for t in tqdm(triples):
    e1 = t[0]
    e2 = t[2]
    G.add_node(e1)
    G.add_node(e2)
    G.add_edge(e1, e2)

triples_dict = defaultdict(set)
for t in tqdm(triples):
    pair = (t[0], t[2])
    triples_dict[pair].add(t[1])

def get1hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    ans = set(ans)
    if entity in ans:
        ans.remove(entity)
    return ans

# 1、获得问题q
# 2、获得头实体和第一个答案实体
# 3、提取所有关系。 关系数量必须和跳数相同
# 4、写入文本。 格式为 ”问题 tab 关系1|关系2|关系3“

#处理问题文本的数据 返回data_array{主题实体，问题，答案}    split=true 会将每一个答案都划分成一个三元组
def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    questions = []
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
        question = question_1 + head +question_2
        ans = data_line[1].split('|')
        ans_one = ans[0]
        data_array.append([head, ans_one])
        questions.append(question)
    if split==False:
        return data_array, questions
    else:
        data = []
        for line in data_array:
            head = line[0]
            tails = line[2]
            for tail in tails:
                data.append([head, tail])
        return data, questions

def getAllRelations(head, tail):
    global G
    global triples_dict
    global sl0
    global sl1
    global sl2
    global sl3
    global slother
    try:
        shortest_length = nx.shortest_path_length(G, head, tail)
    except:
        shortest_length = 0
    if shortest_length == 0:
        sl0+=1
        return list(triples_dict[(head, tail)])
    if shortest_length == 1:
        sl1+=1
        return list(triples_dict[(head, tail)])
    elif shortest_length == 2:
        sl2 += 1
        paths = [nx.shortest_path(G, head, tail)]
        relations = set()
        for p in paths:
            rels1 = triples_dict[(p[0], p[1])]
            rels2 = triples_dict[(p[1], p[2])]
            relations = relations.union(rels1)
            relations = relations.union(rels2)
        return list(relations)
    elif shortest_length == 3:
        sl3 += 1
        paths = [nx.shortest_path(G, head, tail)]
        relations = set()
        for p in paths:
            rels1 = triples_dict[(p[0], p[1])]
            rels2 = triples_dict[(p[1], p[2])]
            rels3 = triples_dict[p[2], p[3]]
            relations = relations.union(rels1)
            relations = relations.union(rels2)
            relations = relations.union(rels3)
        return list(relations)
    else:
        slother += 1
        return []

def write_to_txt(ques, rels, text_name):
    with open('queRels_' + text_name, 'w') as f:
        for line in range(len(ques)):
            f.write(ques[line] + '\t')
            for r in range(len(rels[line])):
                if r == len(rels[line])-1:
                    f.write(rels[line][r])
                else:
                    f.write(rels[line][r] + '|')
            f.write('\n')
        f.close()


entities, questions = process_text_file(data_path)
print('entity and question done')
rels = []
count = 0
sl0 = sl1 = sl2 = sl3 = slother = 0
for ent in entities:
    head = ent[0]
    tail = ent[1]
    rel_one = []
    rel = getAllRelations(head, tail)
    if rel == [] :
        rel = getAllRelations(tail,head)
    if rel == []:
        # print('head: ',head)
        # print('tail: ',tail)
        count += 1
    if len(rel) > hop:
        for i in range(hop):
            rel_one.append(rel[i])
    else:
        rel_one = rel
    rels.append(rel_one)
print('start writ to text :', text_name)
print('count:', count)
print('sl0: ',sl0)
print('sl1: ',sl1)
print('sl2: ',sl2)
print('sl3: ',sl3)
print('slother: ',slother)
print('lenth entitys:', len(entities))
write_to_txt(questions, rels, text_name)
print('done')
