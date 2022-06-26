import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import math
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

parser = argparse.ArgumentParser()

parser.add_argument('--hops', type=str, default='webqsp_full')
parser.add_argument('--load_from', type=str, default='')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--validate_every', type=int, default=1)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--outfile', type=str, default='best_score_model')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--relation_dim', type=int, default=200)
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--freeze', type=str2bool, default=True)
parser.add_argument('--do_batch_norm', type=str2bool, default=True)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
            sent = d[1]
            for word in sent.split():
                if word not in word_to_ix:
                    idx2word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                    
            length = len(sent.split())
            if length > maxLength:
                maxLength = length

    return word_to_ix, idx2word, maxLength

def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict,'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e,r

def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('|')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G

def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key

def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes

def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x

def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result

def validate_v2(data_path, device, model, train_dataloader, entity2idx, model_name):
    model.eval()
    data = process_text_file(data_path)
    sorce_data = get_value_tfidf(data)
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = []
    data_gen = data_generator(data=data, dataloader=train_dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0

    # print('Loading nbhood file')
    # if 'webqsp' in data_path:
    #     # with open('webqsp_test_candidate_list_only2hop.pickle', 'rb') as handle:
    #     with open('webqsp_test_candidate_list_half.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)
    # else:
    #     with open('qa_dev_full_candidate_list.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)

    scores_list = []
    hit_at_10 = 0
    candidates_with_scores = []
    writeCandidatesToFile = True
    for i in tqdm(range(len(data))):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        words_ti_value = d[4]
        rel_id = d[5]
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask, words_ti_value= words_ti_value, relation=rel_id)[0]
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(entity2idx)).to(device)
        mask[head] = 1
        #reduce scores of all non-candidates
        new_scores = scores - (mask*99999)
        pred_ans = torch.argmax(new_scores).item()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)
        if pred_ans == head.item():
            print('Head and answer same')
            print(torch.max(new_scores))
            print(torch.min(new_scores))
        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
            # print(len(candidates))
            # exit(0)
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


        if inTopk(new_scores, ans, 10):
            hit_at_10 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        else:
            num_incorrect += 1
        q_text = d[-1]
        answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        # except:
        #     error_count += 1
        
    if writeCandidatesToFile:
        # pickle.dump(candidates_with_scores, open('candidates_with_score_and_qe_half.pkl', 'wb'))
        pickle.dump(candidates_with_scores, open('webqsp_scores_full_kg.pkl', 'wb'))
        print('wrote candidate file (for future answer processing)')
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    # print(hit_at_10/len(data))
    accuracy = total_correct/len(data)
    # print('Error mean rank: %f' % (incorrect_rank_sum/num_incorrect))
    # print('%d out of %d incorrect were not in top 50' % (not_in_top_50_count, num_incorrect))
    return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(kge_model, hops):
    e = {}
    entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_full/entity_ids.del'
    if 'half' in hops:
        entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_half/entity_ids.del'
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r', encoding='UTF-8')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]       #初始化实体embedding
        if ent_id == 1000000:
            break
    f.close()
    return e

def getRelationEmbeddings(kge_model, hops):
    r = {}
    relation_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_full/relation_ids.del'
    if 'half' in hops:
        relation_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_half/relation_ids.del'
        print('Loading half relation_ids.del')
    embedder = kge_model._relation_embedder
    f = open(relation_dict, 'r', encoding='UTF-8')
    for line in f:
        line = line[:-1].split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = embedder._embeddings(torch.LongTensor([rel_id]))[0]       #初始化embedding
    f.close()
    return r

def train(data_path, que_rel_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim,
          gpu, use_cuda,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay,
          ls, load_from, outfile, do_batch_norm, valid_data_path=None):
    print('Loading entities and relations')
    kg_type = 'full'
    if 'half' in hops:
        kg_type = 'half'
    checkpoint_file = '../../pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
    print('Loading kg embeddings from', checkpoint_file)
    kge_checkpoint = load_checkpoint(checkpoint_file)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()

    e = getEntityEmbeddings(kge_model, hops)
    r = getRelationEmbeddings(kge_model, hops)
    pad = torch.zeros((relation_dim*2), dtype=torch.float32)
    r['pad'] = pad

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    relation2idx, idx2relation, embedding_matrix_rel = prepare_embeddings(r)

    data = process_text_file(data_path, split=False)
    que_rel = process_relQue_file(que_rel_path)

    print('Train file processed, making dataloader')
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    sorce_data = get_value_tfidf(data)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, e, entity2idx, que_rel, r, relation2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, rel_embedding=embedding_matrix_rel, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, do_batch_norm=do_batch_norm)
    print('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    # time.sleep(10)
    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                # model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2].to(device)
                    positive_tail = a[3].to(device)
                    words_ti_value = a[4].to(device)
                    relation = a[5].to(device)
                    loss = model(question_tokenized=question_tokenized,attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail, words_ti_value= words_ti_value, relation=relation)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                
                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                answers, score = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx, train_dataloader=dataset, device=device, model_name=model_name)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
                    # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/"+str(epoch)+".pt")
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/x.pt")

def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 2:
            continue
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NEF'+question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data

def process_relQue_file(rQ_file):
    que_rel = {}
    rel_file = open(rQ_file, 'r')
    for line in rel_file.readlines():
        temp = line.strip().split('\t')
        question = temp[0]
        relations = temp[1].split('|')
        que_rel[question] = relations
    rel_file.close()
    return que_rel

def data_generator(data, dataloader, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        que = question.replace('NEF', '')
        relation = dataloader.que_rel[que]
        rel_id = []
        for r in relation:
            rel = dataloader.rel2idx[r]
            rel_id.append(rel)
        if len(rel_id) < 2:
            rel_id.append(dataloader.rel2idx['pad'])      #572 是关系的pad

        # ============================tf-idf处理============================
        words_ti_value = data_sample[3]
        for i in range(64-len(words_ti_value)):
            words_ti_value.append(0)
        #value_tfidf = words_ti_value
        #ti_more = torch.full((1, 512), value_tfidf[0], dtype=torch.float32)
        #for i in range(len(value_tfidf) - 1):
        #    ti_one = torch.full((1, 512), value_tfidf[i + 1], dtype=torch.float32)
        #    ti_more = torch.cat((ti_more, ti_one), dim=0)
        words_ti_value = torch.tensor(words_ti_value, dtype=torch.float32)

        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            #TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), question_tokenized, attention_mask, ans, words_ti_value, torch.tensor(rel_id, dtype=torch.long), data_sample[1]

#获取tf-idf值
def get_value_tfidf(data):
    sorce_data = data
    questions = []
    qword_in_questions = {}
    word_to_idf = {}
    value_tfidf = []
    for d in sorce_data:
        question = d[1]
        questions.append(question)
        for word in question.split():
            if word not in qword_in_questions:
                qword_in_questions[word] = 0
    for que in questions:
        for word in que.split():
            if word in qword_in_questions:
                qword_in_questions[word] += 1
    for word in qword_in_questions:
        word_to_idf[word] = math.log(len(questions)/(qword_in_questions[word]+1))

    for d in sorce_data:
        words = {}
        question = d[1]
        length  = len(question.split())
        word_tf_idf = []
        total = 0
        for word in question.split():
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1
        for word in question.split():
            word_tf = words[word] / length
            word_tf_idf.append(word_tf * word_to_idf[word])
        for ti in range(len(word_tf_idf)):
            total += word_tf_idf[ti]
        for i in range(len(word_tf_idf)):
            word_tf_idf[i] /= total

        value_tfidf.append(word_tf_idf)
    for l in range(len(sorce_data)):
        sorce_data[l].append(value_tfidf[l])

    return sorce_data

hops = args.hops
model_name = args.model

if 'webqsp' in hops:
    data_path = '../../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt'
    valid_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'
    test_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'
    que_rel_path = '../../data/QA_data/WebQuestionsSP/que_rel_all.txt'

if 'cwq' in hops:
    data_path = '../../data/QA_data/CWQ/qa_train_cwq.txt'
    valid_data_path = '../../data/QA_data/CWQ/qa_dev_cwq.txt'
    test_data_path = '../../data/QA_data/CWQ/qa_dev_cwq.txt'
    que_rel_path = '../../data/QA_data/CWQ/que_rel_cwq_all.txt'

if args.mode == 'train':
    train(data_path=data_path,
          que_rel_path=que_rel_path,
    neg_batch_size=args.neg_batch_size, 
    batch_size=args.batch_size,
    shuffle=args.shuffle_data, 
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
    relation_dim=args.relation_dim, 
    gpu=args.gpu, 
    use_cuda=args.use_cuda, 
    valid_data_path=valid_data_path,
    patience=args.patience,
    validate_every=args.validate_every,
    freeze=args.freeze,
    hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    load_from=args.load_from,
    outfile=args.outfile,
    do_batch_norm=args.do_batch_norm)


'''
elif args.mode == 'eval':
    eval(data_path = test_data_path,
    entity_path=entity_embedding_path, 
    relation_path=relation_embedding_path, 
    entity_dict=entity_dict, 
    relation_dict=relation_dict,
    model_path='checkpoints/head_as_neg_model/best_score_model.pt',
    train_data=data_path,
    gpu=args.gpu,
    hidden_dim=args.hidden_dim,
    relation_dim=args.relation_dim,
    embedding_dim=args.embedding_dim)
'''