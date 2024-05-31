"""
utilities are defined in this code.
"""
import os
import csv
import torch
import pandas as pd
import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from torchvision import datasets, transforms, utils
from custom_eenet import CustomEENet
from eenet import EENet
import torchvision.transforms.functional as tf
from cifar10_noisy import CIFAR10_noisy
from imagenet32 import imagenet32
import ipdb

import matplotlib
matplotlib.use('Agg')

class addNoise(torch.nn.Module):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, noise_levels):
        super().__init__()
        self.noise_levels = noise_levels
        
    def add_noise(self, image, req_snr):
        mean_i = (0.4914, 0.4822, 0.4465)
        std_i = (0.2023, 0.1994, 0.2010)
        #ipdb.set_trace(context=6)

        for t, m, s in zip(image, mean_i, std_i):
            t.mul_(s).add_(m)
        
        """
        %matplotlib inline

        # Generate the noise as you did
        im = image[0].detach().cpu().numpy()
        #ipdb.set_trace(context=6)
        im = np.transpose(im,(1,2,0))
        plt.imshow(im) 
        plt.show()
        #ipdb.set_trace(context=6)
        """

        ch, row, col = image.shape
        mean_n = 0
        var_n = 0.1
        sigma = var_n**0.5
        noise = torch.normal(mean_n,sigma,(ch, row, col))
        snr = 10.0 ** (req_snr / 10.0)
        #ipdb.set_trace(context=6)
        # work out the current SNR
        current_snr = torch.mean(image) / torch.std(noise)

        # scale the noise by the snr ratios (smaller noise <=> larger snr)
        noise = noise * (current_snr / snr)
        img_noise = image + noise
        
        """
        im = img_noise[0].detach().cpu().numpy()
        im = np.transpose(im,(1,2,0))
        plt.imshow(im) 
        plt.show()
        ipdb.set_trace(context=6)
        """

        #utils.save_image(args, img_noise, 'image1')
        img_noise = tf.normalize(img_noise, mean_i, std_i)
        return img_noise

    #def __call__(self, sample):
    def forward(self, img):
        #image, landmarks = sample['data'], sample['landmarks']

        noise_level_index = np.random.randint(0, len(self.noise_levels))
        img = self.add_noise(img, self.noise_levels[noise_level_index])

        #landmarks = landmarks - [left, top]

        #return {'image': image, 'landmarks': landmarks}
        return img, noise_level_index

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def load_custom_dataset(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.device == 'cuda' else {}
    noise_levels = [0]
    if args.add_noise:
        noise_levels = np.arange(args.min_noise_snr, args.max_noise_snr,
                         (args.max_noise_snr - args.min_noise_snr)/args.num_noise_levels)
        print('noise levels: ', noise_levels)

    print('shuffle train: ', args.shuffle_train)
    print('shuffle test: ', args.shuffle_test)
    print('validate: ', args.validate)

    if args.dataset == 'cifar10':
        root = 'data/cifar10'
        trainset = CIFAR10_noisy(root=root, train=True, validate=False, download=True, add_noise=args.add_noise, noise_levels=noise_levels,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))#,
                    #addNoise(noise_levels)]))

        testset = CIFAR10_noisy(root=root, train=False, validate=args.validate, download=True, add_noise=args.add_noise, noise_levels=noise_levels,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
    elif args.dataset == 'imagenet32':
        root = 'data/imagenet32'
        trainset = imagenet32(root=root, train=True, validate=False, download=False,\
            transform=transforms.Compose([\
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.472, 0.450, 0.401], std=[0.257, 0.251, 0.265])]))
        
        testset = imagenet32(root=root, train=False, validate=False, download=False,\
            transform=transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.450, 0.401], std=[0.257, 0.251, 0.265])]))
        
        
        if 0:
            imgs = [item[0] for item in trainset] # item[0] and item[1] are image and its label
            ipdb.set_trace(context=6)
        
            imgs = np.stack(imgs, axis=0)#.numpy()

            # calculate mean over each channel (r,g,b)
            mean_r = imgs[:,0,:,:].mean()/255
            mean_g = imgs[:,1,:,:].mean()/255
            mean_b = imgs[:,2,:,:].mean()/255
            print(mean_r,mean_g,mean_b)

            # calculate std over each channel (r,g,b)
            std_r = imgs[:,0,:,:].std()
            std_g = imgs[:,1,:,:].std()
            std_b = imgs[:,2,:,:].std()
            print(std_r,std_g,std_b)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                    shuffle=args.shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch,\
                    shuffle=args.shuffle_test, **kwargs)
    
    exit_tags = []
    return train_loader, test_loader, exit_tags, trainset, testset


def load_dataset(args):
    """dataset loader.

    This loads the dataset.
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.device == 'cuda' else {}
    
    #noise = AddNoise(noise_levels)
    if args.dataset == 'mnist':
        root = 'data/mnist'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    elif args.dataset == 'cifar10':
        root = 'data/cifar10'
        trainset = datasets.CIFAR10(root=root, train=True, download=True,\
             transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))

        testset = datasets.CIFAR10(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))

    elif args.dataset == 'cifar100':
        root = 'data/cifar100'
        trainset = datasets.CIFAR100(root=root, train=True, download=True,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]))

        testset = datasets.CIFAR100(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]))

    elif args.dataset == 'svhn':
        #def target_transform(target):
        #    return target[0]-1

        root = 'data/svhn'
        transform = transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.SVHN(root=root, split='train', download=True, transform=transform)
                    #,target_transform=target_transform)
        testset = datasets.SVHN(root=root, split='test', download=True, transform=transform)
                    #,target_transform=target_transform)

    elif args.dataset == 'imagenet':
        root = 'data/imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        testset = datasets.ImageFolder(root=root+'/val', transform=transforms.Compose([\
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    
    elif args.dataset == 'imagenet32':
        root = '../data/imagenet32'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        testset = datasets.ImageFolder(root=root+'/val', transform=transforms.Compose([\
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    elif args.dataset == 'tiny-imagenet':
        create_val_img_folder()
        root = '../data/tiny-imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

        testset = datasets.ImageFolder(root=root+'/val/images', transform=transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                    shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch,\
                    shuffle=False, **kwargs)
    """
    exit_tags = []
    if args.loss_func == "v4":
        for data, target in train_loader:
            exit_tags.append(torch.randint(0, args.num_ee+1, (len(target), 1)))
    """
    return train_loader, test_loader#, trainset, testset

def create_val_img_folder():
    """
    This method is responsible for separating validation images into separate sub folders.
    """
    val_dir = '../data/tiny-imagenet/val'
    img_dir = os.path.join(val_dir, 'images')

    file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    file.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def plot_histogram(args, vals, title, n_bins, filename, mean=None, std=None, xlim=None, ylim=None, saveplot=False):
    test_list = vals#np.squeeze(np.ndarray.flatten(vals))
    his = plt.hist(test_list, bins=n_bins, density=False)
    legend_label = None
    """
    if mean is None and std is None:
        mean = np.mean(test_list)
        std = np.std(test_list)
        legend_label = None
    else:
        legend_label = 1
    x_axis = np.arange(mean-5*std, mean+5*std, std/100)
    plt.plot(x_axis, norm.pdf(x_axis,mean,std))
    """
    plt.title(title)
    plt.xlabel('confidence value')
    plt.ylabel('probability density')
    if xlim is not None:
        plt.xlim(0, xlim)
    else:
        a = np.amin(test_list)
        b =  np.amax(test_list)
        plt.xlim(a, b)
    if ylim is not None:
        plt.ylim(0, ylim)
    if legend_label is None:
        plt.legend(('Fitted Gaussian', 'histogram'))
    else:
        plt.legend(('normal data distribution','test data distribution'))
    if saveplot:
        filenm = args.results_dir+'/' + filename + '.png'
        plt.savefig(filenm, bbox_inches='tight')
        #plt.show()
    plt.clf()
    threshold_25 = np.percentile(test_list, 25)
    threshold_75 = np.percentile(test_list, 75)
    threshold_1 = np.percentile(test_list, 99)
    #idx_large_error = np.squeeze(np.argwhere(test_anomaly_metric > threshold_1))
    #print(his[0][-20:])
    #print(his[1][-20:])
    #print("25% percentile: {}".format(threshold_25))
    #print("75% percentile: {}".format(threshold_75))
    print('confs in ' + filename + ': ' + str(len(test_list)))
    #print("Median: {}".format(np.median(test_anomaly_list)))
    #print("Mean: {}".format(np.mean(test_anomaly_list)))
    #print("Std dev: {}".format(np.std(test_anomaly_list)))
    #print("These windows scored the top 1% of anomaly metric ({}): \n{}".format(threshold_1, idx_large_error))
    return mean, std


def plot_history(args):
    """plot figures

    Argument is
    * history:  history to be plotted.

    This plots the history in a chart.
    """
    data = pd.read_csv(args.results_dir+'/history.csv')
    data = data.drop_duplicates(subset='epoch', keep="last")

    
    data = data.sort_values(by='epoch')
    
    title = 'loss of '+args.model+' on '+args.dataset
    xticks = data[['epoch']]
    yticks = data[['train_loss', 'train_loss_sem', 'val_loss', 'val_loss_sem',
                   'pred_loss', 'pred_loss_sem', 'cost_loss', 'cost_loss_sem']]
    labels = ('epochs', 'loss')
    filename = args.results_dir+'/loss_figure.png'
    plot_chart(title, xticks, yticks, labels, filename, args.plot_history)
    
    title = 'val. accuracy and cost rate of '+args.model+' on '+args.dataset
    xticks = data[['epoch']]
    yticks = data[['acc', 'acc_sem', 'cost', 'cost_sem']]
    labels = ('epochs', 'percent')
    filename = args.results_dir+'/acc_cost_figure.png'
    plot_chart(title, xticks, yticks, labels, filename, args.plot_history)

    title = 'std of exit samples in branches using of '+args.model+' on '+args.dataset
    xticks = data[['epoch']]
    yticks = data[['exit_points_std', 'exit_points_std_sem']]
    labels = ('epochs', 'std')
    filename = args.results_dir+'/exit_points_std.png'
    plot_chart(title, xticks, yticks, labels, filename, args.plot_history)

    title = 'cost rate of vs relative val. accuracy '+args.model+' on '+args.dataset
    xticks = data[['acc']]/90
    yticks = data[['cost', 'cost_sem']]
    labels = ('rel accu', 'cost')
    filename = args.results_dir+'/rel_acc_vs_cost_figure.png'
    plot_chart(title, xticks, yticks, labels, filename, args.plot_history)

    data = data.sort_values(by='flop')
    title = 'val. accuracy vs flops of '+args.model+' on '+args.dataset
    xticks = data[['flop', 'flop_sem']]
    yticks = data[['acc', 'acc_sem']]
    labels = ('flops', 'accuracy')
    filename = args.results_dir+'/acc_vs_flop_figure.png'
    plot_chart(title, xticks, yticks, labels, filename, args.plot_history)


def plot_chart(title, xticks, yticks, labels, filename, show):
    """draw chart

    Arguments are
    * title:     title of the chart.
    * xtick:     array that includes the xtick values.
    * yticks:    array that includes the ytick values.
    * labels:    labels of x and y axises.
    * filename:  filename of the chart.

    This plots the history in a chart.
    """
    _, axis = plt.subplots()
    axis.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))

    xerr = None
    for key, value in xticks.items():
        if key.endswith('_sem'):
            xerr = value
        else: xtick = value

    if all(float(x).is_integer() for x in xtick):
        axis.xaxis.set_major_locator(tick.MaxNLocator(integer=True))

    xlabel, ylabel = labels
    min_x = np.mean(xtick)
    if min_x // 10**9 > 0:
        xlabel += ' (GMac)'
    elif min_x // 10**6 > 0:
        xlabel += ' (MMac)'
    elif min_x // 10**3 > 0:
        xlabel += ' (KMac)'

    legend = []
    for key, value in yticks.items():
        if not key.endswith('_sem'):
            legend.append(key)
            ytick = value
            yerr = yticks[key+'_sem']
            plt.errorbar(xtick, ytick, xerr=xerr, yerr=yerr, capsize=3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='best')
    plt.savefig(filename)
    if show:
        plt.show()
    plt.clf()
    print('The figure is plotted under \'{}\''.format(filename))


def display_examples(args, model, dataset):
    """display examples

    Arguments are
    * model:    model object.
    * dataset:  dataset loader object.

    This method shows the correctly predicted sample of images from dataset.
    Produced table shows the early exit block which classifies that samples.
    """
    images = [[[] for j in range(10)] for i in range(args.num_ee+1)]
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data = data.view(-1, *args.input_shape)
            data, target = data.to(args.device), target.to(args.device).item()
            output, idx, _ = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1].item()
            if pred == target:
                if len(images[idx][target]) < 10:
                    images[idx][target].append(idx)

        for idx in range(args.num_ee+1):
            fig, axarr = plt.subplots(10, 10)
            for class_id in range(10):
                for example in range(10):
                    axarr[class_id, example].axis('off')
                for example in range(len(images[idx][class_id])):
                    axarr[class_id, example].imshow(
                        dataset[images[idx][class_id][example]][0].view(args.input_shape[1:]))
            fig.savefig(args.results_dir+'/exitblock'+str(idx)+'.png')


def save_model(args, model, epoch, best=False):
    """save model

    Arguments are
    * model:    model object.
    * best:     version number of the best model object.

    This method saves the trained model in pt file.
    """
    # create the folder if it does not exist
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    
    filename = args.models_dir+'/model'
    if best is False:
        torch.save(model, filename+'.v'+str(epoch)+'.pt')
    else:
        train_files = os.listdir(args.models_dir)
        for train_file in train_files:
            if not train_file.endswith('.v'+str(epoch)+'.pt'):
                os.remove(os.path.join(args.models_dir, train_file))
        os.rename(filename+'.v'+str(epoch)+'.pt', filename+'.pt')
    print('saving model: ', filename+'.v'+str(epoch)+'.pt')


def get_active_exit(args, epoch):
    quantize = min(args.num_ee, ((epoch-1) // 7))
    if args.ee_disable:
        sequential = (epoch-1) % (args.num_ee+1) #XX
    else:
        sequential = (epoch-1) % (args.num_ee) + 1
    return args.num_ee - sequential


def adaptive_learning_rate(args, optimizer, epoch):
    """adaptive learning rate

    Arguments are
    * optimizer:   optimizer object.
    * epoch:       the current epoch number when the function is called.

    This method adjusts the learning rate of training. The same configurations as the ResNet paper.
    """
    learning_rate = args.lr

    # Default SGD: lr=? momentum=0, dampening=0, weight_decay=0, nesterov=False
    if args.optimizer == "SGD":
        # Assummed batch-size is 128. Converge until 165 epochs

        if args.dataset == "cifar10":        
            
            # EENet-110 model
            if args.model[-3:] == "110": 
                if epoch % 60 == 0:
                    factor = epoch/60
                    learning_rate *= (0.1**factor)
            
            #learning_rate = 0.01
            
            # divide by 10 per 100 epochs
            #if epoch % 100 == 0:
            #    learning_rate = 0.1 / (10**(epoch // 100))

        # Assumed batch-size is 256. Converge until 150-160 epochs
        elif args.dataset == "imagenet":
            learning_rate = 0.05 * 0.1**((epoch-1) // 30 + 1)

    # Default Adam: lr=0.001 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    # elif args.optimizer == "Adam":

    print('learning rate: ', learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def print_validation(args, batch, exit_points=None):
    """print validation results

    Arguments are
    * batch:         validation batch results.
    * exit_points:   the number of samples exiting from the specified exit blocks.

    This method prints the results of validation.
    """
    # print the validation results of epoch
    print('     Test avg time: {:.4f}msec; avg val_loss: {:.4f}; avg val_acc: {:.2f}%'
          .format(np.mean(batch['time']),
                  np.mean(batch['val_loss']),
                  np.mean(batch['acc'])))

    # detail print for EENet based models
    if exit_points is not None:
        print('\tavg val_cost: {:.2f}%; exits: <'.format(np.mean(batch['cost'])), end='')
        for i in range(args.num_ee+1):
            print('{:d},'.format(exit_points[i]), end='')
        print('>')


def save_history(args, record):
    """save a record to the history file"""
    args.recorder.writerow([str(record[key]) for key in sorted(record)])
    args.hist_file.flush()


def close_history(args):
    """print the best record in the history file"""
    args.hist_file.close()
    args.hist_file = open(args.results_dir+'/history.csv', 'r', newline='')
    reader = csv.DictReader(args.hist_file)
    best_epoch = 0
    best = {}
    for epoch, record in enumerate(reader):
        if not best or record['val_loss'] < best['val_loss']:
            best = record
            best_epoch = epoch+1

    print('\nThe best avg val_loss: {}, avg val_cost: {}%, avg val_acc: {}%\n'
           .format(best['val_loss'], best['cost'], best['acc']))

    return best_epoch


def save_image(args, image, filename):
    image = utils.make_grid(image)
    npimg = image.numpy()   # convert from tensor
    npimg = np.clip(npimg, 0, 1)
    plt.imsave(args.results_dir + '/' + filename + '.jpg', np.transpose(npimg, (1, 2, 0)))


def x_fmt(x_value, _):
    """x axis formatter"""
    if x_value // 10**9 > 0:
        return '{:.1f}'.format(x_value / 10.**9)
    if x_value // 10**6 > 0:
        return '{:.1f}'.format(x_value / 10.**6)
    if x_value // 10**3 > 0:
        return '{:.1f}'.format(x_value / 10.**3)
    return str(x_value)
