import os
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset_Auto
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args
import fitlog
import debugpy
import time

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 5e-4, 'imputaion_lambda': 5e-2, 'weight_decay': 10}
        args.weight1_model_args = {'learning_rate': 0.001, 'weight_decay': 0.001}
        args.weight2_model_args = {'learning_rate': 5e-4, 'weight_decay': 10}
        args.imputation_model_args = {'learning_rate': 1e-1, 'weight_decay': 1e-4}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 5e-4, 'imputaion_lambda': 0.05, 'weight_decay': 1}
        args.weight1_model_args = {'learning_rate': 1e-3, 'weight_decay': 1e-4}
        args.weight2_model_args = {'learning_rate': 1e-3, 'weight_decay': 1e-3}
        args.imputation_model_args = {'learning_rate': 1e-1, 'weight_decay': 1e-4}
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

    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items, NDCG=None, UAUC=(ut_dict, pt_dict))
    U = test_results['UAUC']
    N = test_results['NDCG']
    print(f'The performance of Autodebias on uniform data is: UAUC = {U} NDCG@5 = {N}')
    return test_results

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_and_eval(train_data, unif_train_data, val_data, test_data, device = 'cuda',
        base_model_args: dict = "{'emb_dim': 64, 'learning_rate': 0.05, 'imputaion_lambda': 0.01, 'weight_decay': 0.05}", 
        weight1_model_args: dict = "{'learning_rate': 0.1, 'weight_decay': 0.005}", 
        weight2_model_args: dict = "{'learning_rate': 0.1, 'weight_decay': 0.005}", 
        imputation_model_args: dict = "{'learning_rate': 0.01, 'weight_decay': 0.5,'bias': 0}", 
        training_args: dict =  "{'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}", args=None):
    
    train_dense = train_data.to_dense()
    # uniform data
    if args.dataset == 'coat':
        train_dense_norm = torch.where(train_dense<-1*torch.ones_like(train_dense), -1*torch.ones_like(train_dense), train_dense)
        train_dense_norm = torch.where(train_dense_norm>torch.ones_like(train_dense_norm), torch.ones_like(train_dense_norm), train_dense_norm)
        del train_dense
        train_dense = train_dense_norm
    users_unif = unif_train_data._indices()[0]
    items_unif = unif_train_data._indices()[1]
    y_unif = unif_train_data._values()
    
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    
    n_user, n_item = train_data.shape
    
    base_model = MetaMF(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.params(), lr=base_model_args['learning_rate'], weight_decay=0) # todo: other optimizer SGD

    weight1_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=weight1_model_args['learning_rate'], weight_decay=weight1_model_args['weight_decay'])

    weight2_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight2_optimizer = torch.optim.Adam(weight2_model.parameters(), lr=weight2_model_args['learning_rate'], weight_decay=weight2_model_args['weight_decay'])

    imputation_model = OneLinear(3).to(device)
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), lr=imputation_model_args['learning_rate'], weight_decay=imputation_model_args['weight_decay'])
    
    sum_criterion = nn.MSELoss(reduction='sum')
    none_criterion = nn.MSELoss(reduction='none')

    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    for epo in range(training_args['epochs']):
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                users_train, items_train, y_train = train_loader.get_batch(users, items)#取出所有这个batch的训练数据的UIpair和值
                if args.dataset == 'coat' or args.dataset == 'kuai':
                    y_train = torch.where(y_train<-1*torch.ones_like(y_train), -1*torch.ones_like(y_train), y_train)
                    y_train = torch.where(y_train >1*torch.ones_like(y_train), torch.ones_like(y_train), y_train)

                # all pair
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                
                # calculate weight 1
                weight1_model.train()
                weight1 = weight1_model(users_train, items_train, (y_train==1) * 1)
                # we use the ReLU to scale the value of learnable propensity in [1,inf). This operation makes the AutoDebias more likes classical propensity-based methods and better performing.
                weight1 = torch.exp(torch.nn.ReLU()(weight1/args.auto_temperature)) 
                
                # calculate weight2
                weight2_model.train()
                weight2 = weight2_model(users_all, items_all, (train_dense[users_all,items_all]!=0)*1)
                weight2 = torch.exp(torch.nn.ReLU()(weight2/args.auto_temperature))
                
                # calculate imputation values
                imputation_model.train()
                impu_f_all = torch.tanh(imputation_model((train_dense[users_all,items_all]).long()+1))

                # one_step_model: assumed model, just update one step on base model. it is for updating weight parameters
                one_step_model = MetaMF(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0)
                one_step_model.load_state_dict(base_model.state_dict())

                # formal parameter: Using training set to update parameters
                one_step_model.train()
                # all pair data in this block
                y_hat_f_all = one_step_model(users_all, items_all)
                cost_f_all = none_criterion(y_hat_f_all, impu_f_all)
                loss_f_all = torch.sum(cost_f_all * weight2)
                # observation data
                y_hat_f_obs = one_step_model(users_train, items_train)
                cost_f_obs = none_criterion(y_hat_f_obs, y_train)
                loss_f_obs = torch.sum(cost_f_obs * weight1)
                loss_f = loss_f_obs + base_model_args['imputaion_lambda'] * loss_f_all + base_model_args['weight_decay'] * one_step_model.l2_norm(users_all, items_all)
                
                # update parameters of one_step_model
                one_step_model.zero_grad()
                grads = torch.autograd.grad(loss_f, (one_step_model.params()), create_graph=True)
                one_step_model.update_params(base_model_args['learning_rate'], source_params=grads)

                # latter hyper_parameter: Using uniform set to update hyper_parameters
                y_hat_l = one_step_model(users_unif, items_unif)
                loss_l = sum_criterion(y_hat_l, y_unif)
                
                weight1_optimizer.zero_grad()
                weight2_optimizer.zero_grad()
                imputation_optimizer.zero_grad()
                loss_l.backward()
                if epo > 20 :
                    weight1_optimizer.step()
                    weight2_optimizer.step()
                imputation_optimizer.step()

                # use new weights to update parameters (real update)       
                weight1_model.train()
                weight1 = weight1_model(users_train, items_train,(y_train==1)*1)
                weight1 = torch.exp( torch.nn.ReLU()(weight1/args.auto_temperature) )
                
                # calculate weight2
                weight2_model.train()
                weight2 = weight2_model(users_all, items_all,(train_dense[users_all,items_all]!=0)*1)
                weight2 = torch.exp(torch.nn.ReLU()(weight2/args.auto_temperature))
                
                # use new imputation to update parameters
                imputation_model.train()
                impu_all = torch.tanh(imputation_model((train_dense[users_all,items_all]).long()+1))

                # loss of training set
                base_model.train()
                # all pair
                y_hat_all = base_model(users_all, items_all)
                cost_all = none_criterion(y_hat_all, impu_all)
                loss_all = torch.sum(cost_all * weight2)
                # observation
                y_hat_obs = base_model(users_train, items_train)
                cost_obs = none_criterion(y_hat_obs, y_train)
                loss_obs = torch.sum(cost_obs * weight1)
                loss = loss_obs + base_model_args['imputaion_lambda'] * loss_all + base_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)
                
                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()

        base_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train= train_loader.get_batch(users, items)
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

        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                    ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if epo >= 50 and early_stopping.check([val_results['AUC']], epo):
            break
    
    # restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)

    test_results = both_test(test_loader, base_model, ('mm', 'DR', 'unbias'), K=5, dataset=args.dataset, device=device)

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    # fitlog.debug()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, unif_train, validation, test = utils.load_dataset_Auto.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device=device)
    train_and_eval(train, unif_train, validation, test, device, base_model_args = args.base_model_args, 
                weight1_model_args = args.weight1_model_args, weight2_model_args = args.weight2_model_args, imputation_model_args = args.imputation_model_args, training_args = args.training_args, args=args)