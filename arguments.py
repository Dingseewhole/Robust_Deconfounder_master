import argparse
import json

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    parser.add_argument('--type', type=str, default='explicit', help='feedback type. implicit, explicit')
    parser.add_argument('--dataset', type=str, default='yahooR3', help='Choose from {yahooR3, coat, simulation}')
    parser.add_argument('--uniform_ratio', type=float, default=0.05, help='the ratio of uniform set in the unbiased dataset.')
    parser.add_argument('--base_model_args', type=json.loads, default='{"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 1}', help='base model arguments.')
    parser.add_argument('--Gama', type=float, nargs='+', default=[1,1], help='hyper para of boundary')

    # hyper-parameters of AutoDebias
    parser.add_argument('--weight1_model_args', type=json.loads, default='{"learning_rate": 0.1, "weight_decay": 0.001}', help='weight model arguments.')
    parser.add_argument('--weight2_model_args', type=json.loads, default='{"learning_rate": 1e-3, "weight_decay": 1e-2}', help='imputation model arguments.')
    parser.add_argument('--imputation_model_args', type=json.loads, default= '{"learning_rate": 1e-1, "weight_decay": 1e-4}', help='imputation model arguments.')          
    parser.add_argument('--training_args', type=json.loads, default = '{"batch_size": 4096, "epochs": 3000, "patience": 60, "block_batch": [6000, 500]}', help='training arguments.')
    parser.add_argument('--auto_temperature', type=float, default=5, help='hyper para from Auto')

    #hyper-parameters of Robst Deconfounder
    parser.add_argument('--base_freq', type=int, default=1, help='the frequency to adversarial update the base ips/dr model')
    parser.add_argument('--clip', type=float, default=1.0, help='hyper parameters for stable learning')
    ## hyper-parameters of RD-IPS
    parser.add_argument('--ips_freq',type=int,default=1,help='the frequency to adversarial update the ips value')
    parser.add_argument('--ips_lr',type=int,default=1,help='learning rate of adversarial learning of ips')
    ## hyper-parameters of RD-AutoDebias
    parser.add_argument('--Gama_Auto', type=float, default=1, help='hyper para of boundary of AutoDebias')
    parser.add_argument('--w1_ad_lr', type=float, default=0, help='learning rate of adversarial learning of AutoDebias base model')
    ## hyper-parameters of RD-DR
    parser.add_argument('--base_dr_freq', type=float, default=0, help='the frequency to adversarial update the base dr model')
    parser.add_argument('--base_dr_lr', type=float, default=0, help='learning rate of adversarial learning of dr base model')
    parser.add_argument('--imp_dr_freq', type=float, default=0, help='the frequency to adversarial update the imputation model of DR')
    parser.add_argument('--imp_dr_lr', type=float, default=0, help='learning rate of adversarial learning of dr imputation model')




    return parser.parse_args()
