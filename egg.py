'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from vgg import VGG
from flops_counter import get_model_complexity_info

__all__ = [
    'EGG', 'egg11', 'egg11_bn', 'egg13', 'egg13_bn', 'egg16', 'egg16_bn',
    'egg19_bn', 'egg19',
]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ExitBlock(nn.Module):
    """Exit Block defition.

    This allows the model to terminate early when it is confident for classification.
    """
    def __init__(self, inplanes, num_classes, input_shape, exit_type):
        super(ExitBlock, self).__init__()
        _, width, height = input_shape
        self.expansion = width * height if exit_type == 'plain' else 1
        self.dim = inplanes * self.expansion

        self.layers = nn.ModuleList()
        #self.etype = exit_type
        if exit_type == 'bnpool':
            bn = nn.Sequential(
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
            )
            self.layers.append(bn)
        elif exit_type == 'conv':
            #dim = inplanes * self.expansion
            conv = nn.Sequential(
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
            )
            self.layers.append(conv)
        elif exit_type == 'conv2':
            #dim = inplanes * self.expansion
            conv2 = nn.Sequential(
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
            )
            self.layers.append(conv2)
        elif exit_type == 'conv3':
            #dim = inplanes * self.expansion
            conv3 = nn.Sequential(
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
                conv3x3(inplanes * self.expansion, inplanes * self.expansion, 1),
                nn.BatchNorm2d(inplanes * self.expansion),
                nn.ReLU(inplace=True),
            )
            self.layers.append(conv3)
        if exit_type != 'plain':
            self.layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.confidence = nn.Sequential(
            nn.Linear(inplanes * self.expansion, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(inplanes * self.expansion, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        for layer in self.layers:
           x = layer(x)
        
        x = x.view(x.size(0), -1)
        conf = self.confidence(x)
        pred = self.classifier(x)
        return pred, conf

class EGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, cfg, total_layers, num_ee, distribution, num_classes, input_shape, exit_type, batch_norm=False, 
                disable_ee_forward=False, **kwargs):
        super(EGG, self).__init__()
        #self.features = features
        self.disable_ee_forward = disable_ee_forward
        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.cost = []
        self.complexity = []
        self.layers = nn.ModuleList()
        self.stage_id = 0
        self.num_ee = num_ee
        self.total_layers = total_layers
        self.exit_type = exit_type
        self.distribution = distribution
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.inplanes = 0

        counterpart_model = VGG(cfg, batch_norm)
        total_flops, total_params = self.get_complexity(counterpart_model)
        self.set_thresholds(distribution, total_flops)

        self.make_layers(cfg, total_flops, batch_norm)

        self.neck = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

        self.confidence = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        
        self.stages.append(nn.Sequential(*self.layers))
        self.stages.append(self.neck)
               
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer()
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        """

    def parameter_initializer(self):
        """
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros,
        and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def set_ee_disable(self, disable):
        self.disable_ee_forward = disable

    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(model, self.input_shape,\
                        print_per_layer_stat=False, as_strings=False)
        return flops, params

    """
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        return pred
    """

    def forward(self, x):
        preds, confs, costs = [], [], []

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            if not self.disable_ee_forward:
                pred, conf = exitblock(x)
                #if not self.training and not self.full_flow: 
                #    if conf.item() > self.exit_threshold:
                #        costs.append(self.cost[idx])
                #        return pred, conf, costs[idx], idx #self.cost[idx]
            else:
                pred = 0
                conf = 0
            preds.append(pred)
            confs.append(conf)
            #costs.append(self.cost[idx])

        x = self.stages[idx+1](x) #layer
        x = x.view(x.size(0), -1)
        x = self.stages[-1](x) #neck
        
        #x = self.neck(x)
        pred = self.classifier(x)
        conf = self.confidence(x)
        #if not self.training:# and not self.full_flow:
        #    return pred, conf, 1.0#, len(self.exits)
        preds.append(pred)
        confs.append(conf)
        #costs.append(1.0)
        
        # costs = self.cost
        return preds, confs#, costs#, 0

    def make_layers(self, cfg, total_flops, batch_norm=False):
        #layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.is_suitable_for_exit():
                    self.add_exit_block(self.exit_type, total_flops)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    self.layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                self.inplanes = in_channels
        #return nn.Sequential(*layers)

    def add_exit_block(self, exit_type, total_flops):
        """add early-exit blocks to the model

        Argument is
        * total_flops:   the total FLOPs of the counterpart model.

        This add exit blocks to suitable intermediate position in the model,
        and calculates the FLOPs and parameters until that exit block.
        These complexity values are saved in the self.cost and self.complexity.
        """
        
        #if self.stage_id > 4:
        #    exit_type = 'conv'            
        self.stages.append(nn.Sequential(*self.layers))
        self.exits.append(ExitBlock(self.inplanes, self.num_classes, self.input_shape, exit_type))
        intermediate_model = nn.Sequential(*(list(self.stages)+list(self.exits)[-1:]))
        flops, params = self.get_complexity(intermediate_model)
        self.cost.append(flops / total_flops)
        self.complexity.append((flops, params))
        self.layers = nn.ModuleList()
        self.stage_id += 1

    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee+1)
        self.threshold = []
        for i in range(self.num_ee):
            if distribution == 'pareto':
                self.threshold.append(total_flops * (1 - (0.8**(i+1))))
            elif distribution == 'fine':
                self.threshold.append(total_flops * (1 - (0.95**(i+1))))
            elif distribution == 'linear':
                self.threshold.append(total_flops * flop_margin * (i+1))
            else:
                self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))

    def is_suitable_for_exit(self):
        """is the position suitable to locate an early-exit block"""
        intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
        flops, _ = self.get_complexity(intermediate_model)
        return self.stage_id < self.num_ee and flops >= self.threshold[self.stage_id]


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def egg11(ee_disable=False, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    return EGG(cfg['A'], total_layers=11, batch_norm=False, disable_ee_forward=ee_disable, **kwargs)


def egg11_bn(ee_disable=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return EGG(cfg['A'], total_layers=11, batch_norm=True, disable_ee_forward=ee_disable, **kwargs)


def egg13():
    """VGG 13-layer model (configuration "B")"""
    return EGG(make_layers(cfg['B']))


def egg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return EGG(make_layers(cfg['B'], batch_norm=True))


def egg16():
    """VGG 16-layer model (configuration "D")"""
    return EGG(make_layers(cfg['D']))


def egg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return EGG(make_layers(cfg['D'], batch_norm=True))


def egg19():
    """VGG 19-layer model (configuration "E")"""
    return EGG(make_layers(cfg['E']))


def egg19_bn(ee_disable=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return EGG(cfg['E'], total_layers=19, batch_norm=True, disable_ee_forward=ee_disable, **kwargs)
