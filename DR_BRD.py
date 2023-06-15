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
import fitlog
import debugpy
# debugpy.listen(('127.1.1.1',8888))

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 1e-6, 'weight_decay': 1}
        args.imputation_model_args = {'emb_dim': 10, 'learning_rate': 1e-6, 'weight_decay': 1}
        args.base_dr_lr = 1e-6
        args.imp_dr_lr = 0.001
        args.Gama = [10, 10]
        args.base_dr_freq = 3
        args.imp_dr_freq = 3
        args.base_freq = 1
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 5e-5, 'weight_decay': 10}
        args.imputation_model_args = {'emb_dim': 10, 'learning_rate': 1e-6, 'weight_decay': 1}
        args.base_dr_lr = 0.01
        args.imp_dr_lr = 0.0001
        args.Gama = [2.5, 2.5]
        args.base_dr_freq = 10
        args.imp_dr_freq = 20
        args.base_freq = 1
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
    print(f'The performance of BRD-DR on uniform data is: UAUC = {U} NDCG@5 = {N}')
    return test_results

def train_and_eval(train_data, unif_train_data, val_data, test_data, n_user, n_item, args, device = 'cuda'): 
    base_model_args, imputation_model_args, training_args, base_dr_lr, imp_dr_lr, base_freq, base_dr_freq, imp_dr_freq = args.base_model_args, args.imputation_model_args, args.training_args, args.base_dr_lr, args.imp_dr_lr, args.base_freq, args.base_dr_freq, args.imp_dr_freq
    train_dense = train_data.to_dense()
    if args.dataset == 'coat':
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
    args.Gama = torch.tensor(args.Gama).cuda()
    lowBound = torch.ones_like(InvP) + (InvP-torch.ones_like(InvP)) / (torch.ones_like(InvP)*args.Gama)
    upBound = torch.ones_like(InvP) + (InvP-torch.ones_like(InvP)) * (torch.ones_like(InvP)*args.Gama)

    y_unique = torch.unique(train_data._values())

    base_model_benchmark=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=base_model_args['emb_dim'], dropout=0).to(device)
    imputation_model_benchmark=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=imputation_model_args['emb_dim'], dropout=0).to(device)
    base_model_benchmark.load_state_dict(torch.load(f'./datasets/{args.dataset}/propensity_dr/base.pth.tar'), strict=False)
    imputation_model_benchmark.load_state_dict(torch.load(f'./datasets/{args.dataset}/propensity_dr/imp.pth.tar'), strict=False)
    base_model=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=base_model_args['emb_dim'], dropout=0).to(device)
    base_model.load_state_dict(torch.load(f'./datasets/{args.dataset}/propensity_dr/base.pth.tar'), strict=False)
    imputation_model=MF_dr(n_user, n_item, upBound, lowBound, y_unique, InvP, dim=imputation_model_args['emb_dim'], dropout=0).to(device)
    imputation_model.load_state_dict(torch.load(f'./datasets/{args.dataset}/propensity_dr/imp.pth.tar'), strict=False)

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
    base_optimizer = torch.optim.SGD([{'params':base_parameters, 'lr':base_model_args['learning_rate'], 'weight_decay':base_model_args['weight_decay']}])
    base_dr_optimizer = torch.optim.SGD([{'params':base_dr_parameters, 'lr':base_dr_lr, 'weight_decay':0}])
    imputation_optimizer = torch.optim.SGD([{'params':imputation_parameters, 'lr':imputation_model_args['learning_rate'], 'weight_decay':imputation_model_args['weight_decay']}])
    imp_dr_optimizer=torch.optim.SGD([{'params':imp_dr_parameters, 'lr':imp_dr_lr, 'weight_decay':0}])

    # loss_criterion
    none_criterion = nn.MSELoss(reduction='none')
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    for epo in range(early_stopping.max_epochs):
        for i in range(base_dr_freq):
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
                    benchmark_loss = base_model_benchmark.base_dr_loss(users_train, items_train, y_train, g_obs, none_criterion)
                    final_max_loss = base_max_loss - benchmark_loss
                    print(f'Epoch: {epo * (args.base_dr_freq) + i}/{early_stopping.max_epochs * (args.ips_freq + args.base_freq + args.imp_dr_freq)} batch {u_batch_idx}.{i_batch_idx}')
                    base_dr_optimizer.zero_grad()
                    base_max_loss.backward()
                    base_dr_optimizer.step()
                    base_model.update_dr()
                    base_model_benchmark.invP.weight.data = base_model.invP.weight.data#.clone().detach() # 尝试一下权重不共享？
                    imputation_model_benchmark.invP.weight.data = base_model.invP.weight.data
                    imputation_model.invP.weight.data = base_model.invP.weight.data#.clone().detach() # 尝试一下权重不共享？

        for i in range(imp_dr_freq):
            for u_batch_idx, users in enumerate(train_loader.User_loader):
                for i_batch_idx, items in enumerate(train_loader.Item_loader):
                    imputation_model.train()
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    if args.dataset == 'coat':
                        y_train = torch.where(y_train < -1 * torch.ones_like(y_train), -1 * torch.ones_like(y_train), y_train)
                        y_train = torch.where(y_train > 1 * torch.ones_like(y_train), torch.ones_like(y_train), y_train)
                    y_hat=base_model(users_train, items_train)
                    imp_max_loss=imputation_model.imp_dr_loss(users_train, items_train, y_train, y_hat, none_criterion)
                    print(f'Epoch: {epo * (args.imp_dr_freq + args.ips_freq) + i}/{early_stopping.max_epochs * (args.ips_freq + args.base_freq + args.imp_dr_freq)} batch {u_batch_idx}.{i_batch_idx}')
                    imp_dr_optimizer.zero_grad()
                    imp_max_loss.backward()
                    imp_dr_optimizer.step()
                    imputation_model.update_dr()

                    base_model.invP.weight.data=imputation_model.invP.weight.data
                    base_model_benchmark.invP.weight.data=imputation_model.invP.weight.data
                    imputation_model_benchmark.invP.weight.data=imputation_model.invP.weight.data
        for i in range(base_freq):
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
                    #dingsh 2020-01-10-s
                    loss_imp_bm = imputation_model_benchmark.imp_model_loss(users_train, items_train, y_train, y_hat, none_criterion) + imputation_model_args['weight_decay'] * imputation_model.l2_norm(users_train, items_train)
                    loss_imp_final = loss_imp - loss_imp_bm
                    print(f'Epoch: {epo * (args.ips_freq + args.base_freq + args.imp_dr_freq) + i}/{early_stopping.max_epochs * (args.ips_freq + args.base_freq + args.imp_dr_freq)} batch {u_batch_idx}.{i_batch_idx}')
                    imputation_optimizer.zero_grad()
                    loss_imp_final.backward()
                    imputation_optimizer.step()

                    # step 3: update base_model
                    base_model.train()
                    # all pair data in this block
                    all_pair = torch.cartesian_prod(users, items)
                    users_all, items_all = all_pair[:,0], all_pair[:,1]

                    y_hat_all = base_model(users_all, items_all)
                    y_hat_all_detach = torch.detach(y_hat_all)
                    g_all = imputation_model(users_all, items_all)
                    loss_all = sum_criterion(y_hat_all, g_all + y_hat_all_detach)

                    g_obs = imputation_model(users_train, items_train)
                    loss_obs = base_model.base_model_loss(users_train, items_train, y_train, g_obs, none_criterion)
                    min_loss = loss_all + loss_obs

                    y_hat_all_bm = base_model_benchmark(users_all, items_all)
                    y_hat_all_bm_detach = torch.detach(y_hat_all_bm)
                    g_all_bm = imputation_model_benchmark(users_all, items_all)
                    loss_all_bm = sum_criterion(y_hat_all_bm, g_all_bm + y_hat_all_bm_detach)

                    g_obs_bm = imputation_model_benchmark(users_train, items_train)
                    loss_obs_bm = base_model_benchmark.base_model_loss(users_train, items_train, y_train, g_obs_bm, none_criterion)
                    min_loss_bm = loss_all_bm + loss_obs_bm
                    min_loss_final = min_loss - min_loss_bm + base_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)

                    base_optimizer.zero_grad()
                    min_loss_final.backward()

                    u_nd = base_model.user_latent.weight.grad/torch.norm(base_model.user_latent.weight.grad,2)
                    i_nd = base_model.item_latent.weight.grad/torch.norm(base_model.item_latent.weight.grad,2)
                    ub_nd = base_model.user_bias.weight.grad/torch.norm(base_model.user_bias.weight.grad,2)
                    ib_nd = base_model.item_bias.weight.grad/torch.norm(base_model.item_bias.weight.grad,2)

                    base_model.user_latent.weight.grad = torch.where(u_nd>args.clip*torch.ones_like(u_nd), args.clip*torch.ones_like(u_nd), base_model.user_latent.weight.grad)
                    base_model.item_latent.weight.grad = torch.where(i_nd>args.clip*torch.ones_like(i_nd), args.clip*torch.ones_like(i_nd), base_model.item_latent.weight.grad)
                    base_model.user_bias.weight.grad = torch.where(ub_nd>args.clip*torch.ones_like(ub_nd), args.clip*torch.ones_like(ub_nd), base_model.user_bias.weight.grad)
                    base_model.item_bias.weight.grad = torch.where(ib_nd>args.clip*torch.ones_like(ib_nd), args.clip*torch.ones_like(ib_nd), base_model.item_bias.weight.grad)

                    base_optimizer.step()

        base_model.eval()
        with torch.no_grad():
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
            
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

        if early_stopping.check([val_results['AUC']], epo):
            break
    base_model.load_state_dict(early_stopping.best_state)
    test_results = both_test(test_loader, base_model, ('CF', 'BRD-DR', 'unbias'), K=5, dataset=args.dataset, device=device)

    return 0

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    bias_train, unif_train, unif_validation, unif_test, m, n = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device='cuda')
    train_and_eval(bias_train, unif_train, unif_validation, unif_test, m, n, args)
