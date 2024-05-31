"""
EENet models
"""
from torch import nn
from flops_counter import get_model_complexity_info
from resnet import ResNet, ResNet6n2
#import ipdb

__all__ = ['EENet',
           'eenet18', 'eenet34', 'eenet50', 'eenet101', 'eenet152',
           'eenet20', 'eenet32', 'eenet44', 'eenet56', 'eenet110',]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic Block defition.

    Basic 3X3 convolution blocks for use on ResNets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck Block defition.

    Bottleneck architecture for > 34 layer ResNets.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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

class EENet(nn.Module):
    """Builds a EENet like architecture.

    Arguments are
    * is_6n2model:        Whether the architecture of the model is 6n+2 layered ResNet.
    * block:              Block function of the architecture either 'BasicBlock' or 'Bottleneck'.
    * total_layers:       The total number of layers.
    * repetitions:        Number of repetitions of various block units.
    * num_ee:             The number of early exit blocks.
    * distribution:       Distribution method of the early exit blocks.
    * num_classes:        The number of classes in the dataset.
    * zero_init_residual: Zero-initialize the last BN in each residual branch,
                          so that the residual branch starts with zeros,
                          and each residual block behaves like an identity. This improves the model
                          by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    * input_shape:        Input shape of the model according to dataset.

    Returns:
        The nn.Module.
    """
    def __init__(self, is_6n2model, block, total_layers, num_ee, distribution, num_classes,
                 input_shape, exit_type, loss_func, repetitions=None, zero_init_residual=False, disable_ee_forward=False,
                 full_flow=False, **kwargs):
        super(EENet, self).__init__()
        self.disable_ee_forward = disable_ee_forward
        self.full_flow = full_flow
        #print('full_flow ' + str(self.full_flow))
        if is_6n2model:
            self.inplanes = 16
            repetitions = [(total_layers-2) // 6]*3
            counterpart_model = ResNet6n2(block, total_layers, num_classes, input_shape)
        else:
            self.inplanes = 64
            counterpart_model = ResNet(block, repetitions, num_classes, input_shape)

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
        self.exit_threshold = 0.5
        
        if loss_func == 'v3':
            self.exit_threshold = 1./self.num_ee

        channel, _, _ = input_shape
        total_flops, total_params = self.get_complexity(counterpart_model)
        self.set_thresholds(distribution, total_flops)

        if is_6n2model:
            self.layers.append(nn.Sequential(
                nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ))
        else:
            self.layers.append(nn.Sequential(
                nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))

        planes = self.inplanes
        stride = 1
        for repetition in repetitions:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            self.layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            if self.is_suitable_for_exit():
                self.add_exit_block(exit_type, total_flops)

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))
                if self.is_suitable_for_exit():
                    self.add_exit_block(exit_type, total_flops)

            planes *= 2
            stride = 2

        assert len(self.exits) == num_ee, \
            'The desired number of exit blocks is too much for the model capacity.'

        planes = 64 if is_6n2model else 512
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        #self.fully_connected = nn.Linear(planes * block.expansion, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(planes * block.expansion, num_classes),
            nn.Softmax(dim=1),
        )
        self.confidence = nn.Sequential(
            nn.Linear(planes * block.expansion, 1),
            nn.Sigmoid(),
        )
        self.stages.append(nn.Sequential(*self.layers))
        self.softmax = nn.Softmax(dim=1)
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer(zero_init_residual)
        
    def set_ee_disable(self, disable):
        self.disable_ee_forward = disable

    def set_full_flow(self, enable):
        self.full_flow = enable

    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(model, self.input_shape,\
                        print_per_layer_stat=False, as_strings=False)
        return flops, params


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


    def parameter_initializer(self, zero_init_residual):
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
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    
    def set_multiple_gpus(self):
        for idx, stage in enumerate(self.stages):
            self.stages[idx] = nn.DataParallel(stage)
        for idx, exitblock in enumerate(self.exits):
            self.exits[idx] = nn.DataParallel(exitblock)


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

        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        conf = self.confidence(x)
        #if not self.training:# and not self.full_flow:
        #    return pred, conf, 1.0#, len(self.exits)
        preds.append(pred)
        confs.append(conf)
        #costs.append(1.0)
        
        # costs = self.cost
        return preds, confs#, costs#, 0


def eenet18(full_flow=False, ee_disable=False, **kwargs):
    """EENet-18 model"""
    model = EENet(is_6n2model=False, block=BasicBlock, total_layers=18,
                  repetitions=[2, 2, 2, 2], disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model


def eenet34(full_flow=False, ee_disable=False, **kwargs):
    """EENet-34 model"""
    model = EENet(is_6n2model=False, block=BasicBlock, total_layers=34,
                  repetitions=[3, 4, 6, 3], disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet50(full_flow=False, ee_disable=False, **kwargs):
    """EENet-50 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=50,
                  repetitions=[3, 4, 6, 3], disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet101(full_flow=False, ee_disable=False, **kwargs):
    """EENet-101 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=101,
                  repetitions=[3, 4, 23, 3], disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet152(full_flow=False, ee_disable=False, **kwargs):
    """EENet-152 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=152,
                  repetitions=[3, 8, 36, 3], disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet20(full_flow=False, ee_disable=False, **kwargs):
    """EENet-20 model"""
    model = EENet(True, BasicBlock, 20, disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet32(full_flow=False, ee_disable=False, **kwargs):
    """EENet-32 model"""
    model = EENet(True, BasicBlock, 32, disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet44(full_flow=False, ee_disable=False, **kwargs):
    """EENet-44 model"""
    model = EENet(True, BasicBlock, 44, disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet56(full_flow=False, ee_disable=False, **kwargs):
    """EENet-56 model"""
    model = EENet(True, BasicBlock, 56, disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model

def eenet110(full_flow=False, ee_disable=False, **kwargs):
    """EENet-110 model"""
    model = EENet(True, BasicBlock, 110, disable_ee_forward=ee_disable, full_flow=full_flow, **kwargs)
    return model
