'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from densenet import DenseNet
from flops_counter import get_model_complexity_info

__all__ = [
    'DenseEENet', 'denseeenet121', 'denseeenet169', 'denseeenet201', 'denseeenet161',
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

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseEENet(nn.Module):
    def __init__(self, block, nblocks, num_ee, distribution, num_classes, input_shape, exit_type, 
                growth_rate=12, reduction=0.5, disable_ee_forward=False, **kwargs):
        super(DenseEENet, self).__init__()
        self.growth_rate = growth_rate

        self.disable_ee_forward = disable_ee_forward
        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.cost = []
        self.complexity = []
        self.layers = nn.ModuleList()
        self.stage_id = 0
        self.num_ee = num_ee
        #self.total_layers = total_layers
        self.exit_type = exit_type
        self.distribution = distribution
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.inplanes = 0

        counterpart_model = DenseNet(block, nblocks, growth_rate)
        total_flops, total_params = self.get_complexity(counterpart_model)
        self.set_thresholds(distribution, total_flops)

        num_planes = 2*growth_rate
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False),
        ))

        for j in range(4):
            self.inplanes = num_planes
            for i in range(nblocks[j]):
                self.layers.append(block(self.inplanes, self.growth_rate))
                self.inplanes += self.growth_rate
                if self.is_suitable_for_exit():
                    self.add_exit_block(exit_type, total_flops)
                
            num_planes += nblocks[j]*growth_rate
            if j < 3:
                out_planes = int(math.floor(num_planes*reduction))
                self.layers.append(Transition(num_planes, out_planes))
                num_planes = out_planes
            else:
                self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(num_planes),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                ))

        self.classifier = nn.Sequential(
            nn.Linear(num_planes, num_classes),
            nn.Softmax(dim=1),
        )

        self.confidence = nn.Sequential(
            nn.Linear(num_planes, 1),
            nn.Sigmoid(),
        )
        
        self.stages.append(nn.Sequential(*self.layers))
               
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer()

        """
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
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

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

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

        #x = self.stages[idx+1](x) #layer
        num_of_stages = len(self.stages)
        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        #x = self.stages[-1](x) #neck
        
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

    """
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    """

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

def denseeenet121(ee_disable=False, **kwargs):
    return DenseEENet(Bottleneck, [6,12,24,16], growth_rate=32, disable_ee_forward=ee_disable, **kwargs)

def denseeenet169(ee_disable=False, **kwargs):
    return DenseEENet(Bottleneck, [6,12,32,32], growth_rate=32, disable_ee_forward=ee_disable, **kwargs)

def denseeenet201(ee_disable=False, **kwargs):
    return DenseEENet(Bottleneck, [6,12,48,32], growth_rate=32, disable_ee_forward=ee_disable, **kwargs)

def denseeenet161(ee_disable=False, **kwargs):
    return DenseEENet(Bottleneck, [6,12,36,24], growth_rate=48, disable_ee_forward=ee_disable, **kwargs)
