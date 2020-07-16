import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.comet import COMET
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file
from torch.utils.tensorboard import SummaryWriter
import wandb

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, tf_writer):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer, tf_writer) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        tf_writer.add_scalar('acc/test', acc, epoch)

        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            params.stop_epoch = 40 #default
        else: #meta-learning methods
            params.stop_epoch = 60 #default
     

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(batch_size = 16)
        base_loader     = base_datamgr.get_data_loader(root='./filelists/tabula_muris', mode='train')
        val_datamgr     = SimpleDataManager(batch_size = 64)
        val_loader      = val_datamgr.get_data_loader(root='./filelists/tabula_muris', mode='val')

        x_dim = base_loader.dataset.get_dim()

        if params.method == 'baseline':
            model           = BaselineTrain(backbone.FCNet(x_dim), params.num_classes)
        elif params.method == 'baseline++':
            model           = BaselineTrain(backbone.FCNet(x_dim), params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet', 'comet', 'matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader(root='./filelists/tabula_muris', mode='train')
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader(root='./filelists/tabula_muris', mode='val') 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        
        go_mask = base_loader.dataset.go_mask
        x_dim = base_loader.dataset.get_dim()

        if params.method == 'protonet':
            model = ProtoNet(backbone.FCNet(x_dim), **train_few_shot_params )
        elif params.method == 'comet':
            model = COMET(backbone.EnFCNet(x_dim, go_mask), **train_few_shot_params )
        elif params.method == 'matchingnet':
            model           = MatchingNet(backbone.FCNet(x_dim), **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
        
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( backbone.FCNet(x_dim), loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            model           = MAML(backbone.FCNet(x_dim), approx = (params.method == 'maml_approx') , **train_few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.model, params.method, params.exp_str)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    store_name = '_'.join([params.dataset, params.model, params.method, params.exp_str])
    wandb.init(project="fewshot_genes", tensorboard=True, name=store_name)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    tf_writer = SummaryWriter(log_dir=params.checkpoint_dir)

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params, tf_writer)
