import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
import math

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaEmbed(MetaModule): 
    def __init__(self, dim_1, dim_2):
        super().__init__()
        ignore = nn.Embedding(dim_1, dim_2)
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', None)
        
    def forward(self):
        return self.weight
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaMF(MetaModule):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = MetaEmbed(n_user, dim)
        self.item_latent = MetaEmbed(n_item, dim)
        self.user_bias = MetaEmbed(n_user, 1)
        self.item_bias = MetaEmbed(n_item, 1)
        #用register_buffer的形式把模型参数存下来，可以让这部分参数不参与更新
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 

        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):
        u_latent = self.dropout(self.user_latent.weight[users])
        i_latent = self.dropout(self.item_latent.weight[items])
        u_bias = self.user_bias.weight[users]
        i_bias = self.item_bias.weight[items]

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        preds = preds.squeeze(dim=-1)
        return preds

    def allrank(self, users, train_his):
        u_latent = self.dropout(self.user_latent.weight[users])
        i_latent = self.dropout(self.item_latent.weight)
        u_bias = self.user_bias.weight[users]
        i_bias = self.item_bias.weight
        preds = torch.mm(u_latent, i_latent.T) + u_bias + i_bias.T
        for i,u in enumerate(set(users)):
            preds[i,train_his[u]._indices()] = -999.0
        return preds

    def l2_norm(self, users, items, unique = True): 

        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent.weight[users]**2) + torch.sum(self.item_latent.weight[items]**2)) / 2
        return l2_loss

class OneLinear(nn.Module):
    """
    linear model: r
    """
    def __init__(self, n):
        super().__init__()
        
        self.data_bias= nn.Embedding(n, 1)
        self.init_embedding()
        
    def init_embedding(self): 
        self.data_bias.weight.data *= 0.001

    def forward(self, values):
        d_bias = self.data_bias(values)
        return d_bias.squeeze()

class TwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """
    def __init__(self, n_user, n_item):
        super().__init__()
        
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)

    def forward(self, users, items):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        preds = u_bias + i_bias
        return preds.squeeze()

class ThreeLinear(nn.Module):
    """
    linear model: u + i + r / o
    """
    def __init__(self, n_user, n_item, n):
        super().__init__()
        
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias= nn.Embedding(n, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a = init)
        self.data_bias.weight.data *= 0.001

    def forward(self, users, items, values):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)

        preds = u_bias + i_bias + d_bias
        return preds.squeeze()

class FourLinear(nn.Module):
    """
    linear model: u + i + r + p
    """
    def __init__(self, n_user, n_item, n, n_position):
        super().__init__()
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias= nn.Embedding(n, 1)
        self.position_bias = nn.Embedding(n_position, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a = init)
        self.data_bias.weight.data *= 0.001
        self.position_bias.weight.data *= 0.001

    def forward(self, users, items, values, positions):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)
        p_bias = self.position_bias(positions)

        preds = u_bias + i_bias + d_bias + p_bias
        return preds.squeeze()

class Position(nn.Module): 
    """
    the position parameters for DLA
    """
    def __init__(self, n_position): 
        super().__init__()
        self.position_bias = nn.Embedding(n_position, 1)

    def forward(self, positions): 
        return self.position_bias(positions).squeeze(dim=-1)

    def l2_norm(self, positions): 
        positions = torch.unique(positions)
        return torch.sum(self.position_bias(positions)**2)

class MF_heckman(nn.Module): 
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        self.MF = MF(n_user, n_item, dim)
        self.sigma = nn.Parameter(torch.randn(1))
    
    def forward(self, users, items, lams): 
        pred_MF = self.MF(users, items)
        pred = pred_MF - 1 * lams
        return pred

    def l2_norm(self, users, items): 
        l2_loss_MF = self.MF.l2_norm(users, items)
        l2_loss = l2_loss_MF + 1000 * torch.sum(self.sigma**2)
        return l2_loss

class MF(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        # preds = u_bias + i_bias

        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

class MF_MSE(MetaModule):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 

        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):
        u_latent = self.dropout(self.user_latent.weight[users])
        i_latent = self.dropout(self.item_latent.weight[items])
        u_bias = self.user_bias.weight[users]
        i_bias = self.item_bias.weight[items]

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        preds = preds.squeeze(dim=-1)
        return preds

    def allrank(self, users, train_his):
        u_latent = self.dropout(self.user_latent.weight[users])
        i_latent = self.dropout(self.item_latent.weight)
        u_bias = self.user_bias.weight[users]
        i_bias = self.item_bias.weight
        preds = torch.mm(u_latent, i_latent.T) + u_bias + i_bias.T
        for i,u in enumerate(set(users)):
            preds[i,train_his[u]._indices()] = -999.0
        return preds

    def l2_norm(self, users, items, unique = True): 

        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent.weight[users]**2) + torch.sum(self.item_latent.weight[items]**2)) / 2
        return l2_loss

class MF_ips(nn.Module):
    """
    RD module for matrix factorization.
    """
    def __init__(self, n_user, n_item, upBound, lowBound, corY, InverP, dim=40, dropout=0):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(n_item, 2)
        self.ips_hat = InverP
        self.init_embedding(None, self.ips_hat)

        
    def init_embedding(self, init, ips_hat): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        self.invP.weight = torch.nn.Parameter(ips_hat)

    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        return preds.squeeze(dim=-1)
     
    def base_model_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones_like(y_train)

        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]

        cost = loss_f(preds, y_train)
        loss = torch.sum(weight * cost)
        return loss

    def ips_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        weight[y_train == self.corY[0]] = self.invP(items)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(items)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = - torch.sum(weight * cost)
        return loss
        

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

    def update_ips(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data>self.upBound] = self.upBound[self.invP.weight.data>self.upBound]
            self.invP.weight.data[self.invP.weight.data<self.lowBound] = self.lowBound[self.invP.weight.data<self.lowBound]

class MF_dr(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, upBound, lowBound, corY, InverP, dim=40, dropout=0):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(n_item, 2)
        self.ips_hat = InverP
        self.init_embedding(None, self.ips_hat)
        
    def init_embedding(self, init, ips_hat): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        #初始点就用原始的ips值
        self.invP.weight = torch.nn.Parameter(ips_hat)

    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        return preds.squeeze(dim=-1)

    ''' 
    def base_model_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]

        cost = loss_f(preds, y_train)
        loss = torch.sum(weight * cost)
        return loss
    '''
    def base_model_loss(self,users,items,y_train,g_obs,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        y_hat_obs=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        y_hat_obs = y_hat_obs.squeeze(dim=-1)
        y_hat_obs_detach=torch.detach(y_hat_obs)

        e_obs=none_criterion(y_hat_obs,y_train)
        e_hat_obs=none_criterion(y_hat_obs,g_obs+y_hat_obs_detach)
        cost_obs=e_obs-e_hat_obs

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        loss_obs=torch.sum(weight*cost_obs)

        return loss_obs
    
    def imp_model_loss(self,users,items,y_train,y_hat,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        e_hat=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        e_hat = e_hat.squeeze(dim=-1)
        e = y_train - y_hat
        cost_e = none_criterion(e_hat, e)

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        return torch.sum(weight*cost_e)

    '''
    def ips_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]

        cost = loss_f(preds, y_train)
        #为了实现max优化loss取负了
        loss = - torch.sum(weight * cost)
        return loss
    '''
    def base_dr_loss(self,users,items,y_train,g_obs,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        y_hat_obs=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        y_hat_obs = y_hat_obs.squeeze(dim=-1)
        y_hat_obs_detach=torch.detach(y_hat_obs)

        e_obs=none_criterion(y_hat_obs,y_train)
        e_hat_obs=none_criterion(y_hat_obs,g_obs+y_hat_obs_detach)
        cost_obs=e_obs-e_hat_obs

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        loss_obs=-torch.sum(weight*cost_obs)

        return loss_obs
    
    def imp_dr_loss(self,users,items,y_train,y_hat,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        e_hat=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        e_hat = e_hat.squeeze(dim=-1)
        e = y_train - y_hat
        cost_e = none_criterion(e_hat, e)

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        return -torch.sum(weight*cost_e)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

    '''
    def update_dr(self):
        with torch.no_grad():
            for i in range(self.invP.weight.shape[0]):
                if sum(self.invP.weight.data[i]>self.upBound[i])!=0:
                    # print(f'## {i} is over up')
                    self.invP.weight[i] = self.upBound[i]
                elif sum(self.invP.weight.data[i]<self.lowBound[i])!=0:
                    # print(f'## {i} is over down')
                    self.invP.weight[i] = self.lowBound[i]
                else:
                    dsh=1
        return 0
    '''

    def update_dr(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data>self.upBound]=self.upBound[self.invP.weight.data>self.upBound]
            self.invP.weight.data[self.invP.weight.data<self.lowBound]=self.lowBound[self.invP.weight.data<self.lowBound]