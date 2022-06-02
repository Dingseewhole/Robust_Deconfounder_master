import os
import numpy as np
import random
import time

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 3000, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 5e-7, 'weight_decay': 1}
        args.ips_lr = 0.0001
        args.ips_freq = 5
        args.base_freq = 1
        args.Gama = [10, 10]
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 5000, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 1e-5, 'weight_decay': 1}
        args.ips_lr = 0.0005
        args.ips_freq = 8
        args.base_freq = 1
        args.Gama = [25, 25]
    else: 
        print('invalid arguments')
        os._exit()

def both_test(loader, model_name, testname, K = 5, dataset = "None",device='cuda'):
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    ndcg_ratings = torch.empty(0).to(device)
    ndcg_item = torch.empty(0).to(device)
    ut_dict={}
    pt_dict={}
    for batch_idx, (users, items, ratings) in enumerate(loader):
        pre_ratings = model_name(users, items)
        for i,u in enumerate(users):
            try:
                ut_dict[u.item()].append(ratings[i].item())
                pt_dict[u.item()].append(pre_ratings[i].item())
            except:
                ut_dict[u.item()]=[ratings[i].item()]
                pt_dict[u.item()]=[pre_ratings[i].item()]
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))

        pos_mask = torch.where(ratings>=torch.ones_like(ratings), torch.arange(0,len(ratings)).float().to(device), 100*torch.ones_like(ratings))
        pos_ind = pos_mask[pos_mask != 100].long()
        users_ndcg = torch.index_select(users, 0, pos_ind)

    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items, NDCG=None, UAUC=(ut_dict, pt_dict))
    U = test_results['UAUC']
    N = test_results['NDCG']
    print(f'The performance of RD-IPS on uniform data is: UAUC = {U} NDCG@5 = {N}')
    return test_results

def train_and_eval(train_data, unif_train_data, val_data, test_data, n_user, n_item, args, device = 'cuda'): 
    model_args, training_args, ips_lr = args.base_model_args, args.training_args, args.ips_lr
    train_dense = train_data.to_dense()
    if args.dataset == 'coat' or args.dataset == 'kuai':
        train_dense_norm = torch.where(train_dense<-1*torch.ones_like(train_dense), -1*torch.ones_like(train_dense), train_dense)
        train_dense_norm = torch.where(train_dense_norm>torch.ones_like(train_dense_norm), torch.ones_like(train_dense_norm), train_dense_norm)
        del train_dense
        train_dense = train_dense_norm

    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    def Naive_Bayes_Propensity(train, unif):
        # follow [1] Jiawei Chen et, al, AutoDebias: Learning to Debias for Recommendation 2021SIGIR and
        # [2] Tobias Schnabel, et, al, Recommendations as Treatments: Debiasing Learning and Evaluation
        P_Oeq1 = train._nnz() / (train.size()[0] * train.size()[1])
        train._values()[train._values()<torch.tensor([-1.0]).to(device)]=-1.0
        y_unique = torch.unique(train._values())
        P_y_givenO = torch.zeros(y_unique.shape).to(device)
        P_y = torch.zeros(y_unique.shape).to(device)

        for i in range(len(y_unique)): 
            P_y_givenO[i] = torch.sum(train._values() == y_unique[i]) / torch.sum(torch.ones(train._values().shape).to(device))
            P_y[i] = torch.sum(unif._values() == y_unique[i]) / torch.sum(torch.ones(unif._values().shape).to(device))
        Propensity = P_y_givenO * P_Oeq1 / P_y
        Propensity=Propensity*(torch.ones((n_item,2)).to(device))

        return y_unique, Propensity
    y_unique, Propensity = Naive_Bayes_Propensity(train_data, unif_train_data)

    InvP = torch.reciprocal(Propensity)
    lowBound = torch.ones_like(InvP) + (InvP-torch.ones_like(InvP)) / (torch.ones_like(InvP)*args.Gama[0])
    upBound = torch.ones_like(InvP) + (InvP-torch.ones_like(InvP)) * (torch.ones_like(InvP)*args.Gama[0])
    model = MF_ips(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=model_args['emb_dim'], dropout=0).to(device)
    ips_parameters, base_parameters = [], []
    for pname, p in model.named_parameters():
        if (pname in ['invP.weight']):
            ips_parameters += [p]
        else:
            base_parameters += [p]
    optimizer_base = torch.optim.SGD([{'params':base_parameters, 'lr':model_args['learning_rate'], 'weight_decay':model_args['weight_decay']}])
    optimizer_ips = torch.optim.SGD([{'params':ips_parameters, 'lr':ips_lr, 'weight_decay':0}])

    none_criterion = nn.MSELoss(reduction='none')

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(model, **stopping_args)
    for epo in range(1,early_stopping.max_epochs+1):
        training_loss = 0
        # To increase the stability of the gradient during adversarial training, we utilize the ips_freq and base_freq to control the frequency of adversarial training.
        if epo % args.ips_freq ==0:
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    model.train()
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)      

                    max_loss = model.ips_loss(users_train, items_train, y_train, none_criterion)
                    optimizer_ips.zero_grad()
                    max_loss.backward()
                    optimizer_ips.step()
                    model.update_ips()  
        if epo % args.base_freq == 0:
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    model.train()
                    users_train, items_train, y_train = train_loader.get_batch(users, items)

                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)      
                    sup_loss = model.base_model_loss(users_train, items_train, y_train, none_criterion)
                    min_loss = sup_loss + model_args['weight_decay'] * model.l2_norm(users, items)
        
                    optimizer_base.zero_grad()
                    min_loss.backward()
                    optimizer_base.step()

        if epo % 2 == 0:
            model.eval()
            with torch.no_grad():
                train_pre_ratings = torch.empty(0).to(device)
                train_ratings = torch.empty(0).to(device)
                for u_batch_idx, users in enumerate(train_loader.User_loader):
                    for i_batch_idx, items in enumerate(train_loader.Item_loader):
                        users_train, items_train, y_train = train_loader.get_batch(users, items)
                        pre_ratings = model(users_train, items_train)
                        train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                        train_ratings = torch.cat((train_ratings, y_train))

                val_pre_ratings = torch.empty(0).to(device)
                val_ratings = torch.empty(0).to(device)
                for batch_idx, (users, items, ratings) in enumerate(val_loader):
                    pre_ratings = model(users, items)
                    val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                    val_ratings = torch.cat((val_ratings, ratings))

            train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])

            val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

            print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'],
                        ' '.join([key + ':' + '%.3f' % train_results[key] for key in train_results]),
                        ' '.join([key + ':' + '%.3f' % val_results[key] for key in val_results])))

            if early_stopping.check([val_results['AUC']], epo):
                break

    # restore the best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)

    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))

    test_results = both_test(test_loader, model, ('CF', 'IPS', 'unbias'), K=5, dataset=args.dataset, device=device)

    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    bias_train, unif_train, unif_validation, unif_test, m, n = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device='cuda')
    train_and_eval(bias_train, unif_train, unif_validation, unif_test, m, n, args)