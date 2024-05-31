"""
Gated Net model
"""
from torch import nn
import torch.nn.functional as F
import torch
import ipdb

__all__ = ['CustomEENet', 'eenet8']

class GatedNetV(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(GatedNetV, self).__init__()
        channel, _, _ = input_shape
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.do = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        #self.fc = nn.Linear(128*4*4, 512)
        #self.fc_final = nn.Linear(512, num_classes)
        self.fc = nn.Linear(64*8*8, 512)
        self.fc_final = nn.Linear(512, num_classes)
        self.sm = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)
        #x = self.relu(self.conv3_1(x))
        #x = self.relu(self.conv3_2(x))
        #x = self.pool(x)
        #ipdb.set_trace(context=6) 
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.do(x)
        x = F.relu(self.fc_final(x))
        x = self.sm(x)
        return x

class GatedNetL(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GatedNetL, self).__init__()
        self.do = nn.Dropout(p=0.2)
        self.fc_00 = nn.Linear(input_size, input_size)
        #torch.nn.init.normal_(self.fc_00.weight, mean=0.0, std=1.0)        
        self.b_00 = nn.BatchNorm1d(input_size)
        self.fc_01 = nn.Linear(input_size, 512)  
        #torch.nn.init.normal_(self.fc_01.weight, mean=0.0, std=1.0)              
        self.b_01 = nn.BatchNorm1d(512)
        #self.fc1 = nn.Linear(input_size*2, 1024)
        #self.b1 = nn.BatchNorm1d(1024)
        #self.fc2 = nn.Linear(1024, 512)
        #self.b2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        #self.b3 = nn.BatchNorm1d(128)
        self.fc_final = nn.Linear(128, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        #x = torch.flatten(x, 1)
        x = F.relu(self.fc_00(x))
        x = self.b_00(x)
        x = F.relu(self.fc_01(x))
        x = self.b_01(x)        
        #x = self.do(x)
        
        #x = F.relu(self.fc1(x))
        #x = self.b1(x)
        #x = self.do(x)
        #x = F.relu(self.fc2(x))
        #x = self.b2(x)
        #x = self.do(x)
        x = F.relu(self.fc3(x))
        #x = self.b3(x)
        #x = self.do(x)
        #x = F.relu(self.fc_final(x))
        x = self.fc_final(x)
        x = self.sm(x)
        return x

class GatedNetS(nn.Module):
    def __init__(self, input_shape, num_classes, starting_filter):
        super(GatedNetS, self).__init__()
        channel, _, _ = input_shape
        self.filter = 6#starting_filter
        self.initblock = nn.Sequential(
            nn.Conv2d(channel, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.filter, 16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        #self.conv1 = nn.Conv2d(channel, self.filter, kernel_size=3, stride=1, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(self.filter, 16, kernel_size=5, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.bn = nn.BatchNorm2d(self.filter)
        #self.do = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * 15 * 15, num_classes)
        #self.fc2 = nn.Linear(512, 120)
        #self.fc3 = nn.Linear(120, 84)
        #self.fc4 = nn.Linear(84, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sm = nn.Softmax(dim=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        #ipdb.set_trace(context=6) 
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = self.initblock(x)
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        #x = self.do(x)
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        x = self.sm(x)
        return x

class GatedNet(nn.Module):
    """Custom EENet-8 model.

    This model (EENet-8) consists of constant two early-exit blocks.
    and it is a very small CNN having 2-8 filters in its layers.
    """
    def __init__(self, input_shape, num_classes, starting_filter):
        super(GatedNet, self).__init__()
        channel, _, _ = input_shape
        self.filter = starting_filter
        self.initblock = nn.Sequential(
            nn.Conv2d(channel, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
        )
        
        self.basicblock1 = nn.Sequential(
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
        )
        self.basicblock2 = self.get_basic_block(1)
        self.basicblock3 = self.get_basic_block(2)
        self.finalblock = nn.Sequential(
            nn.BatchNorm2d(self.filter*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.conv2d_6 = nn.Conv2d(self.filter, self.filter*2, kernel_size=1, stride=2, padding=0)
        self.conv2d_9 = nn.Conv2d(self.filter*2, self.filter*4, kernel_size=1, stride=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        #self.exit0_classifier = self.get_classifier(num_classes, 1)
        #self.exit1_classifier = self.get_classifier(num_classes, 2)
        #self.exit0_confidence = self.get_confidence(1)
        #self.exit1_confidence = self.get_confidence(2)
        self.classifier = self.get_classifier(num_classes, 4)
        self.complexity = [(546, 137), (1844, 407), (6982, 1490)]
        if self.filter == 4:
            self.complexity = [(1792, 407), (6608, 1395), (25814, 5332)]
    
    def get_basic_block(self, expansion):
        """get basic block as nn.Sequential"""
        filter_in = self.filter * expansion
        return nn.Sequential(
            nn.BatchNorm2d(filter_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_in, filter_in*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_in*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_in*2, filter_in*2, kernel_size=3, stride=1, padding=1))
    
    def get_classifier(self, num_classes, expansion):
        """get classifier as nn.Sequential"""
        filter_in = self.filter * expansion
        return nn.Sequential(
            nn.Linear(filter_in, num_classes),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        cost_0, cost_1 = 0.08, 0.26
        x = self.initblock(x)
        residual = self.basicblock1(x)
        x = residual + x
        """
        if not self.disable_ee_forward:
            e_x = self.pool(x).view(-1, self.filter)
            pred_0 = self.exit0_classifier(e_x)
            conf_0 = self.exit0_confidence(e_x)
            if (not self.training and conf_0.item() > 0.5):
                return pred_0, 0, cost_0
        else:
            pred_0 = 0
            conf_0 = 0
        """
        residual = self.basicblock2(x)
        x = self.conv2d_6(x)
        x = residual + x
        """
        if not self.disable_ee_forward:
            e_x = self.pool(x).view(-1, self.filter*2)
            pred_1 = self.exit1_classifier(e_x)
            conf_1 = self.exit1_confidence(e_x)
            if (not self.training and conf_1.item() > 0.5):
                return pred_1, 1, cost_1
        else:
            pred_1 = 0
            conf_1 = 0
        """
        residual = self.basicblock3(x)
        x = self.conv2d_9(x)
        x = residual + x

        x = self.finalblock(x)
        x = x.view(-1, self.filter*4)
        pred_2 = self.classifier(x)
        #if not self.training:
        #    return pred_2, 2, 1.0

        #return (pred_0, pred_1, pred_2), (conf_0, conf_1), (cost_0, cost_1)
        return pred_2


def gnet(input_shape, num_classes, filters=2):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    print('Building Gated Net')
    return GatedNet(input_shape, num_classes, filters)

def gnet_s(input_shape, num_classes, filters=2):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    print('Building S Gated Net')
    return GatedNetS(input_shape, num_classes, filters)
    #return GatedNetS()

def gnet_v(input_shape, num_classes):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    print('Building V Gated Net')
    return GatedNetV(input_shape, num_classes)
    
def gnet_l(input_shape, num_classes):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    print('Building L Gated Net')
    return GatedNetL(input_shape, num_classes)
