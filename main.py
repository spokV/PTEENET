"""
high level support for doing this and that.
"""
from __future__ import print_function
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from init import initializer
from eenet import EENet
from egg import EGG
from custom_eenet import CustomEENet
import matplotlib.pylab as plt
#import acoustics
import loss_functions
import utils
import config
import gc
import sys
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as tf
import os
import shutil

# self.cost = [0.1282048606925034, 0.24453504403495593, 0.36086522737740845]

def add_noise(args, image, snr_db):
    mean_i = (0.4914, 0.4822, 0.4465)
    std_i = (0.2023, 0.1994, 0.2010)
    for t, m, s in zip(image, mean_i, std_i):
        t.mul_(s).add_(m)
    
    # Generate the noise as you did
    _, ch, row, col = image.shape
    mean_n = 0
    var_n = 0.1
    sigma = var_n**0.5
    noise = torch.normal(mean_n,sigma,(1, ch, row, col))
    snr = 10.0 ** (snr_db / 10.0)
    
    # work out the current SNR
    current_snr = torch.mean(image) / torch.std(noise)

    # scale the noise by the snr ratios (smaller noise <=> larger snr)
    noise = noise * (current_snr / snr)
    img_noise = image + noise
    #utils.save_image(args, img_noise, 'image1')
    img_noise = tf.normalize(img_noise, mean_i, std_i)

    # return the new signal with noise
    return img_noise


def examine(args, model, val_loader):
    """examine the model output.
    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.
    This examines the outputs of already trained model.
    """
    model.eval()
    val_confs = []
    for i in range(args.num_ee+1):
        val_confs.append([])
    
    #val_confs[0].append('aa2')
    #val_confs = np.empty((args.num_ee, 0), float)#np.zeros(len(val_loader.dataset))
    #i = 0
    """
    print(args.results_dir+'/pred_vs_conf.csv')
    experiment = open(args.results_dir+'/pred_vs_conf.csv', 'w', newline='')
    recorder = csv.writer(experiment, delimiter=',')
    recorder.writerow(['target',
                       'start_pred_seq',
                       'start_conf_seq',
                       'start_exit_seq',
                       'actual_pred',
                       'actual_conf',
                       'actual_exit'])
    """
    with torch.no_grad():
        for data, target in val_loader:
            data = add_noise(args, data, 10)
            
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            preds, confs, costs, idx = model(data)

            pred = preds.max(1, keepdim=True)[1].item()
            conf = torch.max(confs).item()
            #i += 1
            #confs = [c.item() for c in confs]
            target = target.item()
            if target == pred:
                val_confs[idx].append(conf)

    total_correct = 0
    for i in range(args.num_ee+1):
        total_correct += len(val_confs[i])
        if len(val_confs[i]) > 1:
            filename = 'test_confs_hist_ee' + str(i)
            utils.plot_histogram(args, val_confs[i], 'conf', 100, filename, saveplot=True)
    print('total currect: ', total_correct)
    print('total acc : ', total_correct * 100 /10000)
    return

def train(args, model, train_loader, optimizer):
    """train the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.
    * optimizer:    optimize the model during training.
    * epoch:        epoch number.

    This trains the model and prints the results of each epochs.
    """
    losses = []
    cost = []
    pred_losses = []
    cost_losses = []
    model.train()

    for i in range(args.num_ee+1):
        cost.append(model.complexity[i][0]/model.complexity[-1][0])                
    
    # actual training starts
    #for batch_id, ((data, noise), target) in enumerate(train_loader):
    for batch_id, (data, target) in enumerate(train_loader):
        # fetch the current batch data
        data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
        exit_tag = None
        optimizer.zero_grad()
        cum_loss = 0
        #cost = []

        # training settings for EENet based models
        if isinstance(model, (CustomEENet, EENet, EGG)):
            pred, conf = model(data)
            #ipdb.set_trace(context=6)
            #cost.append(1)#torch.tensor(1.0).to(args.device))
            
            if args.use_main_targets:
                _, target = torch.max(pred[args.num_ee], 1)

            cum_loss, pred_loss, cost_loss = loss_functions.loss(args, exit_tag, pred, target, conf, cost)
            
        # training settings for other models
        else:
            pred = model(data)
            cum_loss = F.cross_entropy(pred, target)

        #ipdb.set_trace(context=6)
        losses.append(float(cum_loss))
        pred_losses.append(float(pred_loss))
        cost_losses.append(float(cost_loss))
        cum_loss.backward()
        optimizer.step()
        """
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
        """
        #print(model.exits[1].classifier[0].weight)

        # update the exit tags of inputs
    #print(pred_loss)
    #print('target')
    #print(target)
    
    # print the training results of epoch
    result = {'train_loss': round(np.mean(losses), 4),
              'train_loss_sem': round(stats.sem(losses), 2),
              'pred_loss': round(np.mean(pred_losses), 4),
              'pred_loss_sem': round(stats.sem(pred_losses), 2),
              'cost_loss': round(np.mean(cost_losses), 4),
              'cost_loss_sem': round(stats.sem(cost_losses), 2)}

    print('Train avg loss: {:.4f}'.format(result['train_loss']))
    
    return result

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(args, model, val_loader):
    batch = {'time':[], 'cost':[], 'flop':[], 'acc':[], 'val_loss':[]}
    exit_points = [0]*(args.num_ee+1)
    # switch to evaluate mode
    model.eval()
    main_ee_losses = []
        
    with torch.no_grad():
        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        elapsed_time = 0
        
        for batch_id, (data, target) in enumerate(val_loader):
            batch_size = target.shape[0]        
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            # compute output
            #start = time.process_time()
            elapsed_time = 0
            # results of EENet based models
            
            #if isinstance(model, (EENet, CustomEENet)):   

            #starter.record()
            preds, confs = model(data)
            #ender.record()
            #torch.cuda.synchronize()
            #elapsed_time = starter.elapsed_time(ender)/batch_size                        

            if mode == Mode.train_main:
                pred = preds[args.num_ee]
                cost = 1
                exit_points[args.num_ee] += batch_size
                flop = model.complexity[-1][0]
                pred_loss = F.nll_loss(pred.log(), target)
                main_ee_losses.append(float(pred_loss))
                
            elif mode == Mode.train_ee:
                losses = []
                costs = []
                flops = []

                entropies = torch.empty((args.num_ee + 1, batch_size), dtype=torch.float)
                pred_losses = torch.empty((args.num_ee + 1, batch_size), dtype=torch.float)
                pred = torch.zeros((batch_size, preds[0].shape[1]), dtype=torch.float, device = 'cuda')

                _, target_main = torch.max(preds[args.num_ee], 1)                
        
                if args.use_main_targets:
                    target = target_main
                
                for i in range(args.num_ee+1):
                    cost = model.complexity[i][0]/model.complexity[-1][0]
                    pred_losses[i] = F.nll_loss(preds[i].log(), target, reduction='none')
                    #ipdb.set_trace(context=3)
                    entropies[i] = torch.sum(preds[i] * preds[i].log(), 1)
                
                main_ee_loss = F.nll_loss(preds[args.num_ee].log(), target)
                main_ee_losses.append(float(main_ee_loss))
                
                for j in range(batch_size):
                    best_exit = args.num_ee
                    for i in range(args.num_ee): 
                        if args.termination == 'entropy':   
                            if entropies[i][j] < args.loss_threshold:
                                best_exit = i
                                #ipdb.set_trace(context=6)
                                break
                        elif args.termination == 'confidence':   
                            if confs[i][j] > args.loss_threshold:
                                best_exit = i
                                #ipdb.set_trace(context=6)
                                break
                    
                    #ipdb.set_trace(context=3)
                    exit_points[best_exit] += 1
                    best_pred_loss = pred_losses[best_exit][j]
                    losses.append(float(best_pred_loss))
                    cost = model.complexity[best_exit][0]/model.complexity[-1][0] 
                    flop = model.complexity[best_exit][0]
                    flops.append(float(flop))
                    costs.append(float(cost))
                    pred[j] = preds[best_exit][j]
                
                #ipdb.set_trace(context=3)
                cost = round(np.mean(costs), 4)
                pred_loss = round(np.mean(losses), 4)
                flop = round(np.mean(flops), 4)
                #ipdb.set_trace(context=3)                    
                losses = None
                costs = None
                flops = None
                                
            else:
                print('Unknown mode')
                
            #elapsed_time = time.process_time() - start
            #ipdb.set_trace(context=6)
            prec1 = accuracy(pred.float().data, target)[0]
            pred = None
                
            batch['acc'].append(float(prec1))
            batch['time'].append(elapsed_time)
            batch['cost'].append(cost*100.)
            batch['flop'].append(flop)
            batch['val_loss'].append(float(pred_loss))
            
    print('main ee avg loss: ', round(np.mean(main_ee_losses), 4))
    utils.print_validation(args, batch, exit_points)
    
    result = {}
    for key, value in batch.items():
        result[key] = round(np.mean(value), 4)
        result[key+'_sem'] = round(stats.sem(value), 2)
    
    result['exit_points'] = exit_points
    result['exit_points_std'] = round(np.diff(exit_points).std(), 4)
    result['exit_points_std_sem'] = 0
    #result['exit_points_std'] = np.diff(exit_points).std()# / (np.max(exit_points) - np.min(exit_points))
    
    return result

from early_stopping import early_stop

def run(model, optimizer, lr_scheduler, args, train_loader, test_loader, writer):
    # best = {}
    # best_epoch = 0
    
    print('Running for {:5d} epochs'.format(args.epochs))
    try:
        #if not args.no_tensorboard:
        #    writer = SummaryWriter('../runs/train_eenet_experiment_1')
        torch.cuda.empty_cache()
        early_stopping = early_stop(patience=10)
    
        for epoch in range(args.start_epoch, args.epochs + 1):
            print('{:3d}:'.format(epoch), end='')

            # two-stage training uses the loss version-1 after training for 25 epochs
            if args.two_stage and epoch > 25:
                args.loss_func = "v1"
            
            result = {'epoch':epoch}
            
            result.update(train(args, model, train_loader, optimizer))
            if not args.no_tensorboard:
                writer.add_scalar("Train/loss", result['train_loss'], epoch)
                for name, weight in model.named_parameters():
                    #ipdb.set_trace(context=3)
                    if 'exit' in name:
                        writer.add_histogram(name,weight, epoch)
                    #writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            

            # use adaptive learning rate
            if args.adaptive_lr:
                #lr_scheduler.step()
                lr_scheduler(args, optimizer, epoch)

            # validate and keep history at each log interval
            if epoch % args.log_interval == 0:
                result.update(validate(args, model, test_loader))
                utils.save_history(args, result)
                if not args.no_tensorboard:
                    writer.add_scalar("Val/loss", result['val_loss'], epoch)
                    writer.add_scalars('Val/acc_cost', {
                        'acc': result['acc'],
                        'cost': result['cost']
                    }, epoch)
                    writer.add_scalar("Val/exit_points_std", result['exit_points_std'], epoch)
                    #writer.add_histogram('exit_points',np.array(result['exit_points']), epoch)
                if args.early_stopping:
                    early_stopping(result['val_loss'])
                    if early_stopping.early_stop:
                        break
                
            if not args.no_tensorboard:    
                writer.flush()
            # save model parameters
            if not args.no_save_model:
                utils.save_model(args, model, epoch)
    except KeyboardInterrupt:
        utils.close_history(args)
        utils.plot_history(args)
        if not args.no_tensorboard:
            writer.close()
        sys.exit()
    # print the best validation result
    best_epoch = utils.close_history(args)

    # save the model giving the best validation results as a final model
    if args.save_best:
        utils.save_model(args, model, best_epoch, True)
    utils.plot_history(args)
    if not args.no_tensorboard:
        writer.close()


def enable_branches_training_only(model, args):
    if args.model == 'eenet8':
        model.set_ee_disable(False)
        model.initblock.requires_grad_(False)
        model.basicblock1.requires_grad_(False)
        model.basicblock2.requires_grad_(False)
        model.basicblock3.requires_grad_(False)
        model.finalblock.requires_grad_(False)
        model.classifier.requires_grad_(False)
        model.conv2d_6.requires_grad_(False)
        model.conv2d_9.requires_grad_(False)
    else:
        model.set_ee_disable(False)
        for idx, exitblock in enumerate(model.exits):
            model.stages[idx].requires_grad_(False)
            for param in model.exits.parameters():
                #ipdb.set_trace(context=3)
                param.requires_grad = True
            #model.exits.requires_grad(True)

        model.stages[-1].requires_grad_(False)
        model.classifier.requires_grad_(False)
        model.confidence.requires_grad_(False)
        
        #params = model_post.state_dict()
        #print(params)
        
        """
        ipdb.set_trace(context=3)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print("!!!!: ", name, p.requires_grad)
            else:
                print(name, p.requires_grad)
        """

from torch.utils.data import Dataset, TensorDataset, DataLoader

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        #ipdb.set_trace(context=6)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
        
def load_data(args):
    return utils.load_dataset(args)#load_custom_dataset(args)

def load_data_serialized(args):
    serialized_data_dir = 'serialized_data'
    
    data_file_train = serialized_data_dir + '/datas_train' + '.pt'
    data_train = torch.load(data_file_train)
    print('loaded data train at: ', data_file_train)
    
    data_file_test = serialized_data_dir + '/datas_test' + '.pt'
    data_test = torch.load(data_file_test)
    print('loaded data test at: ', data_file_test)
    
    gt_file_train = serialized_data_dir + '/targets_train' + '.pt'
    target_train = torch.load(gt_file_train)
    print('loaded exits ground truths train at: ', gt_file_train)
    
    gt_file_test = serialized_data_dir + '/targets_test' + '.pt'
    target_test = torch.load(gt_file_test)
    print('loaded exits ground truths test at: ', gt_file_test)

    data_train, data_test, target_train, target_test = train_test_split(data_train,\
                                                                        target_train, test_size=0.2, shuffle=False)
    
    data_train, target_train = torch.from_numpy(data_train), torch.from_numpy(target_train)    
    data_test, target_test = torch.from_numpy(data_test), torch.from_numpy(target_test)
    
    train_set = CustomTensorDataset(tensors=(data_train, target_train))    
    validation_set = CustomTensorDataset(tensors=(data_test, target_test))
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle_train)
    loader_val = torch.utils.data.DataLoader(validation_set, batch_size=args.test_batch, shuffle=args.shuffle_test)

    return loader_train, loader_val                

from enum import Enum

class Mode(Enum):
    train_main = 0
    train_ee = 1
    generate_exits_gt = 2 
    train_gating = 3
    generate_relative_loss = 4 
    plot_relative_loss = 5
    calc_relative_time = 6

def main(mode: Mode, writer):
    print(mode.value)
    args = config.args_global
    args += config.argu[mode.value]
    
    model, optimizer, lr_scheduler, args = initializer(args)
    print(args)
    
    if mode == Mode.train_main:
        train_loader, test_loader = load_data(args)
        model.set_ee_disable(True)
        print('Disabled EE branches')
        lr_scheduler = utils.adaptive_learning_rate
        #print(model)
        run(model, optimizer, lr_scheduler, args, train_loader, test_loader, writer)
    elif mode == Mode.train_ee:
        train_loader, test_loader = load_data(args)
        #train_loader, test_loader = load_data_serialized(args)
        enable_branches_training_only(model, args)
        print('Enabled EE branches')
        print('loss threshold: ', args.loss_threshold)
        run(model, optimizer, lr_scheduler, args, train_loader, test_loader, writer)
    
    print('Finished')

if __name__ == '__main__':
    tensorboard_dir = 'runs/eenet_experiment_1'
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    #%load_ext tensorboard
    #%tensorboard --logdir runs/eenet_experiment_1
    
    mode = Mode.train_main
    main(mode, writer)
