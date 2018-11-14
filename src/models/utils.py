#from TSN lr policy 
import torch
def get_optim_config(net):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    conv_cnt = 0
    bn_cnt = 0
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
            ps = list(m.parameters())
            '''
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
            '''
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm1d):
            bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            '''
            # later BN's are frozen
            if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            '''
            bn.extend( list(m.parameters()) )
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    return [
        #{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1, 'name': "first_conv_weight"},
        #{'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'name': "first_conv_bias"},
        {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "normal_weight"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': "BN scale/shift"},
    ]

