import numpy as np
from scipy.sparse import lil_matrix
import torch
import pandas as pd

import utils.data_loader
import cppimport.import_hook
import utils.ex as ex
from collections import Counter
import os
import math
import time



def calc(n,m,ttuser,ttitem,pre,ttrating,ut_dict,atk=5):
    #n:参与test的userID的最大值，m:参与test的itemID的最大值
    #ttuser：参与test的userID,ttitem：参与test的itemID
    #pre：prediciton_rating
    #ttrating：GrounTruth
    #ut_dict：每一个user的feedback列表
    #atk： 算ndcg和recall和pre 的at K
    user=ttuser.cpu().detach().numpy()
    item=ttitem.cpu().detach().numpy()
    pre=pre.cpu().detach().numpy()
    rating=ttrating.cpu().numpy()
    posid=np.where(rating==1)
    posuser=user[posid]
    positem=item[posid]
    preall=np.ones((n,m))*(-1000000)
    preall[user,item]=pre
    id=np.argsort(preall,axis=1,kind='quicksort',order=None)#跟topk类似，输出下标，每行都是得分从大到小排item的下标.行是user列是item
    id=id[:,::-1]
    id1=id[:,:atk]
    # print(id1)
    # ans=ex.gaotest(posuser,positem,id1,id)
    ans = mycalc(posuser, positem, id1, id, ut_dict)
    # pre@k, re@k, NDCG, MRR, NDCG@k
    # print(ans)
    # print('have_pos_user / all_user:', ans[5], len(list(set(user))))
    return [ans[0],ans[1],ans[2]]


def nll(vector_predict, vector_true):
    # if torch.sum(torch.log(1 + torch.exp(-vector_predict * vector_true))).item() == np.inf:
    #     print("#################NLL is NAN##################")
    #     exit(0)
    # return -1 / vector_true.shape[0] * torch.sum(torch.log(1 + torch.exp(-vector_predict * vector_true))).item()
    return 0

def auc(vector_predict, vector_true, device = 'cuda'): 
    pos_indexes = torch.where(vector_true == 1)[0].to(device)
    pos_whe=(vector_true == 1).to(device)
    sort_indexes = torch.argsort(vector_predict).to(device)
    rank=torch.zeros((len(vector_predict))).to(device)
    rank[sort_indexes] = torch.FloatTensor(list(range(len(vector_predict)))).to(device)
    rank = rank * pos_whe
    auc = (torch.sum(rank) - len(pos_indexes) * (len(pos_indexes) - 1) / 2) / (len(pos_indexes) * (len(vector_predict) - len(pos_indexes)))
    return auc.item()

def uauc(UAUC, device = 'cuda'): 
    (ut_dict, pt_dict) = UAUC
    uauc = 0.0
    u_size = 0.0
    for k in ut_dict:
        if (1 in ut_dict[k]) and (-1.0 in ut_dict[k]):
            uauc_one = auc(torch.tensor(pt_dict[k]), torch.tensor(ut_dict[k]))
            uauc += uauc_one
            u_size += 1.0
    return uauc/ u_size

def mse(vector_predict, vector_true): 
    mse = torch.mean((vector_predict - vector_true)**2)
    return mse.item()


def evaluate(vector_Predict, vector_Test, metric_names, users = None, items = None, NDCG=None, UAUC=None, ndcgK = 5):
    # print('kk')
    global_metrics = {
        "AUC": auc,
        "NLL": nll,
        "MSE": mse, 
        'Recall_Precision_NDCG@': ndcgK}
    # print('gg')
    results = {}
    for name in metric_names:
        # print(name+'1')
        if name != 'Recall_Precision_NDCG@':
            # print(name+'2')
            results[name] = global_metrics[name](vector_predict=vector_Predict, vector_true=vector_Test)
            # print(name+'3')
    
    if 'Recall_Precision_NDCG@' in metric_names: 
        # results['NDCG_all'] = ndcg_all(NDCG,atk=20)
        users_num = torch.max(users).item() + 1
        items_num = torch.max(items).item() + 1
        Recall_Precision_NDCG = calc(users_num, items_num, users, items, vector_Predict, vector_Test, ut_dict=UAUC[0], atk=global_metrics['Recall_Precision_NDCG@'])
        results['Precision'] =  Recall_Precision_NDCG[0]
        results['Recall'] =  Recall_Precision_NDCG[1]
        results['NDCG'] =  Recall_Precision_NDCG[2]
        # results['NDCG_quality'] = Recall_Precision_NDCG[3] / Recall_Precision_NDCG[4]
        # print(f'users set conunt in NDCG / all users set:{Recall_Precision_NDCG[3]} / {Recall_Precision_NDCG[4]}')
    
    if UAUC != None:
        results['UAUC'] = uauc(UAUC)
    try:
        nantest_1 = results['MSE']
        nantest_2 = results['NLL']
    except:
        nantest_1 = 0
        nantest_2 = 0
    if (math.isnan(nantest_1)) or (math.isnan(nantest_2)):
        print("#################MSE or NLL is NAN##################")
        exit(0)
    return results

def mycalc(posuser, positem, id1, id, ut_dict):
    n_test, n_user, n_item, atk = posuser.shape[0], id.shape[0], id.shape[1], id1.shape[1]
    logsum = 1 / np.log2(np.arange(n_item + 2)[2:])
    logsum = np.cumsum(logsum)
    precision, recall, ndcg, n_interacted_user = 0, 0, 0, 0

    # figure out how many items does each user like + define the map between user and item
    keys, user_total_item = set(), np.zeros(n_user)
    for i in range(n_test):
        if -1.0 in ut_dict[posuser[i]]:
            user_total_item[posuser[i]] += 1
            keys.add((posuser[i] << 30) + positem[i])

    for i in range(n_user):
        if user_total_item[i] == 0:
            continue
        n_interacted_user += 1
        idcg = logsum[min(int(user_total_item[i]), atk) - 1]
        hit, dcg = 0, 0
        for j in range(atk):
            code = (i << 30) + id1[i, j]
            if code in keys:
                hit, dcg = hit + 1, dcg + 1 / np.log2(j + 2)
        precision += hit / atk
        recall += hit / user_total_item[i]
        ndcg += dcg / idcg

    return [precision / n_interacted_user, recall / n_interacted_user, ndcg / n_interacted_user]

def ndcg_all(NDCG,atk=5):
    (ndcg_ratings, ndcg_item) = NDCG
    pre_hit=torch.topk(ndcg_ratings, atk)[1]
    NDCG_sum=0.
    for inter in range(len(ndcg_item)):
        for r,iid in enumerate(pre_hit[inter]):#在topk的结果里找有没有命中的
            if iid.data.int()==ndcg_item[inter].data.int():
                NDCG_oneinter=(1./(np.log2(r+1.+1.)))
                break#只有一个正样本，最多命中一次
            else:
                NDCG_oneinter=0
        NDCG_sum+=NDCG_oneinter
    NDCG_k=float(NDCG_sum/len(ndcg_item))
    return NDCG_k 
