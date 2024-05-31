import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick


def plot_history(model, dataset, results_dir):
    """plot figures

    Argument is
    * history:  history to be plotted.

    This plots the history in a chart.
    """
    data = pd.read_csv(results_dir+'/history.csv')
    data = data.drop_duplicates(subset='epoch', keep="last")

    data = data.sort_values(by='epoch')
    title = 'loss of '+model+' on '+dataset
    xticks = data[['epoch']]
    yticks = data[['train_loss', 'train_loss_sem', 'val_loss', 'val_loss_sem',
                   'pred_loss', 'pred_loss_sem', 'cost_loss', 'cost_loss_sem']]
    labels = ('epochs', 'loss')
    #filename = args.results_dir+'/loss_figure.png'
    plot_chart(title, xticks, yticks, labels)#, filename, args.plot_history)

    title = 'val. accuracy and cost rate of '+model+' on '+dataset
    xticks = data[['epoch']]
    yticks = data[['acc', 'acc_sem', 'cost', 'cost_sem']]
    labels = ('epochs', 'percent')
    #filename = args.results_dir+'/acc_cost_figure.png'
    plot_chart(title, xticks, yticks, labels)#, filename, args.plot_history)

    data = data.sort_values(by='flop')
    title = 'val. accuracy vs flops of '+model+' on '+dataset
    xticks = data[['flop', 'flop_sem']]
    yticks = data[['acc', 'acc_sem']]
    labels = ('flops', 'accuracy')
    #filename = args.results_dir+'/acc_vs_flop_figure.png'
    plot_chart(title, xticks, yticks, labels)#, filename, args.plot_history)


def plot_chart(title, xticks, yticks, labels):#, filename, show):
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
    #plt.savefig(filename)
    #if show:
    plt.show()
    #plt.clf()
    #print('The figure is plotted under \'{}\''.format(filename))


def x_fmt(x_value, _):
    """x axis formatter"""
    if x_value // 10**9 > 0:
        return '{:.1f}'.format(x_value / 10.**9)
    if x_value // 10**6 > 0:
        return '{:.1f}'.format(x_value / 10.**6)
    if x_value // 10**3 > 0:
        return '{:.1f}'.format(x_value / 10.**3)
    return str(x_value)

model = 'eenet20'
dataset = 'svhn'
results_dir = '../results/svhn/eenet20/ee3_fine_conv_lambda_1.7'

plot_history(model, dataset, results_dir)