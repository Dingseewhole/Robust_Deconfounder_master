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
        args.imputation_model_args = {'emb_dim': 10, 'learning_rate': 5e-7, 'weight_decay': 1}
        args.base_dr_lr = 5e-5
        args.imp_dr_lr = 5e-5
        args.base_dr_freq = 3
        args.imp_dr_freq = 3
        args.base_freq = 1
        args.Gama = [10, 10]
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 5000, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 1e-5, 'weight_decay': 10}
        args.imputation_model_args = {'emb_dim': 10, 'learning_rate': 5e-5, 'weight_decay': 10}
        args.base_dr_lr = 5e-3
        args.imp_dr_lr = 5e-3
        args.base_dr_freq = 100
        args.imp_dr_freq = 20
        args.base_freq = 1
        args.Gama = [3, 3]
    else: 
        print('invalid arguments')
        os._exit()

def both_test(loader, model_name, testname, K = 5, dataset = "None", device='cuda'):
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
    print(f'The performance of Robust Deconfounder-Doubly Robust (RD-DR) on uniform data is: UAUC = {U} NDCG@5 = {N}')
    return test_results
   

def train_and_eval(train_data, unif_train_data, val_data, test_data, n_user, n_item, args, device = 'cuda'): 
    base_model_args, imputation_model_args, training_args, base_dr_lr, imp_dr_lr, base_freq, base_dr_freq, imp_dr_freq = args.base_model_args, args.imputation_model_args, args.training_args, args.base_dr_lr, args.imp_dr_lr, args.base_freq, args.base_dr_freq, args.imp_dr_freq
    train_dense = train_data.to_dense()
    if args.dataset == 'coat' or args.dataset == 'kuai':
        train_dense_norm = torch.where(train_dense<-1*torch.ones_like(train_dense), -1*torch.ones_like(train_dense), train_dense)
        train_dense_norm = torch.where(train_dense_norm>torch.ones_like(train_dense_norm), torch.ones_like(train_dense_norm), train_dense_norm)
        del train_dense
        train_dense = train_dense_norm

    # build data_loader. 
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    def Naive_Bayes_Propensity(train, unif): 
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
    base_model=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=base_model_args['emb_dim'], dropout=0).to(device)
    imputation_model=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=imputation_model_args['emb_dim'], dropout=0).to(device)
    base_dr_parameters, base_parameters = [], []
    for pname, p in base_model.named_parameters():
        if (pname in ['invP.weight']):
            base_dr_parameters += [p]
        else:
            base_parameters += [p]
    imp_dr_parameters,imputation_parameters=[],[]
    for pname, p in imputation_model.named_parameters():
        if (pname in ['invP.weight']):
            imp_dr_parameters += [p]
        else:
            imputation_parameters+=[p]
    base_optimizer = torch.optim.SGD([{'params':base_parameters, 'lr':base_model_args['learning_rate'], 'weight_decay':0}])
    base_dr_optimizer = torch.optim.SGD([{'params':base_dr_parameters, 'lr':base_dr_lr, 'weight_decay':0}])
    imputation_optimizer = torch.optim.SGD([{'params':imputation_parameters, 'lr':imputation_model_args['learning_rate'], 'weight_decay':0}])
    imp_dr_optimizer=torch.optim.SGD([{'params':imp_dr_parameters, 'lr':imp_dr_lr, 'weight_decay':0}])

    # loss_criterion
    none_criterion = nn.MSELoss(reduction='none')
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    early_stopping_imp = EarlyStopping(imputation_model, **stopping_args)
    # fitlog.add_best_metric({"bias_val": {"Earlystop": 0}})
    for epo in range(1,early_stopping.max_epochs+1):
        training_loss = 0
        if epo % base_dr_freq == 0:
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    # step 2(optional): every dr_freq epochs, update inverse propensity scores
                    base_model.train()
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)
                    g_obs = imputation_model(users_train, items_train)
                    base_max_loss = base_model.base_dr_loss(users_train, items_train, y_train, g_obs, none_criterion)
                    base_dr_optimizer.zero_grad()
                    base_max_loss.backward()
                    base_dr_optimizer.step()
                    base_model.update_dr()
                    imputation_model.invP.weight.data=base_model.invP.weight.data

        if epo % imp_dr_freq == 0:
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    imputation_model.train()
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)
                    y_hat=base_model(users_train, items_train)
                    imp_max_loss=imputation_model.imp_dr_loss(users_train, items_train, y_train, y_hat, none_criterion)
                    imp_dr_optimizer.zero_grad()
                    imp_max_loss.backward()
                    imp_dr_optimizer.step()
                    imputation_model.update_dr()
                    base_model.invP.weight.data=imputation_model.invP.weight.data
        if epo % base_freq == 0:
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    users_train, items_train, y_train = train_loader.get_batch(users, items)

                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)
                    
                    # step 1: update imputation_model
                    imputation_model.train()
                    
                    y_hat=base_model(users_train, items_train)
                    loss_imp = imputation_model.imp_model_loss(users_train, items_train, y_train, y_hat, none_criterion) + imputation_model_args['weight_decay'] * imputation_model.l2_norm(users_train, items_train)
                    imputation_optimizer.zero_grad()
                    loss_imp.backward()
                    imputation_optimizer.step()

                    # step 3: update base_model
                    base_model.train()
                    # all pair data in this block
                    all_pair = torch.cartesian_prod(users, items)
                    users_all, items_all = all_pair[:,0], all_pair[:,1]

                    y_hat_all = base_model(users_all, items_all)
                    y_hat_all_detach = torch.detach(y_hat_all)
                    g_all = imputation_model(users_all, items_all)
                    loss_all = sum_criterion(y_hat_all, g_all + y_hat_all_detach) # sum(e_hat)         

                    g_obs = imputation_model(users_train, items_train)
                    loss_obs = base_model.base_model_loss(users_train, items_train, y_train, g_obs, none_criterion)
                    min_loss = loss_all + loss_obs + base_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)
                    base_optimizer.zero_grad()
                    min_loss.backward()
                    base_optimizer.step()

        if epo % 20 == 0:
            base_model.eval()
            with torch.no_grad():
                # train metrics                          
                train_pre_ratings = torch.empty(0).to(device)
                train_ratings = torch.empty(0).to(device)
                for u_batch_idx, users in enumerate(train_loader.User_loader):
                    for i_batch_idx, items in enumerate(train_loader.Item_loader):
                        users_train, items_train, y_train = train_loader.get_batch(users, items)
                        pre_ratings = base_model(users_train, items_train)
                        train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                        train_ratings = torch.cat((train_ratings, y_train))

                # validation metrics
                val_pre_ratings = torch.empty(0).to(device)
                val_ratings = torch.empty(0).to(device)
                for batch_idx, (users, items, ratings) in enumerate(val_loader):
                    pre_ratings = base_model(users, items)
                    val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                    val_ratings = torch.cat((val_ratings, ratings))

            train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])

            val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

            print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'],
                        ' '.join([key + ':' + '%.3f' % train_results[key] for key in train_results]),
                        ' '.join([key + ':' + '%.3f' % val_results[key] for key in val_results])))

            early_stopping_imp.check([val_results['AUC']], epo)
            if early_stopping.check([val_results['AUC']], epo):
                break

    # restore the best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    # print(base_model.invP.weight.data)
    base_model.load_state_dict(early_stopping.best_state)
    imputation_model.load_state_dict(early_stopping_imp.best_state)

    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = base_model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))

    test_results = both_test(test_loader, base_model, ('mm', 'DR', 'unbias'), K=5, dataset=args.dataset, device=device)

    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    bias_train, unif_train, unif_validation, unif_test, m, n = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device='cuda')
    train_and_eval(bias_train, unif_train, unif_validation, unif_test, m, n, args)
    # fitlog.finish()