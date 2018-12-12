#from TSN lr policy 
import torch
def get_optim_config(net,lr_for_parts = [1]):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = [ [] for i in range(len(lr_for_parts)) ]
    normal_bias = [ [] for i in range(len(lr_for_parts)) ]
    bn_weight = [ [] for i in range(len(lr_for_parts)) ]
    bn_bias = [ [] for i in range(len(lr_for_parts)) ]

    conv_cnt = 0
    bn_cnt = 0
    for idx , m in enumerate(net.modules()):
        pass
    total_cnt = idx + 1

    print( 'len of net.modules {}'.format(total_cnt) )
    print( 'len lr_for_parts'.format( len(lr_for_parts) ) )
    part_idx = 0
    for idx , (name,m) in enumerate(net.named_modules()):
        if 'layer3' in name:
            part_idx = 1
        elif 'classifier' in name:
            part_idx = 2 

        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance( m , torch.nn.ConvTranspose2d ):
            ps = list(m.parameters())
            '''
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
            '''
            normal_weight[part_idx].append(ps[0])
            if len(ps) == 2:
                normal_bias[part_idx].append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight[part_idx].append(ps[0])
            if len(ps) == 2:
                normal_bias[part_idx].append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm1d):
            #bn[part_idx].extend(list(m.parameters()))
            ps = list(m.parameters())
            bn_weight[part_idx].append(ps[0])
            if len(ps) == 2:
                bn_bias[part_idx].append(ps[1])
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            '''
            # later BN's are frozen
            if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            '''
            ps = list(m.parameters())
            bn_weight[part_idx].append(ps[0])
            if len(ps) == 2:
                bn_bias[part_idx].append(ps[1])
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    ret= [] 
    for part_idx , x in enumerate(zip(normal_weight,lr_for_parts)):
        p , lr = x 
        ret.append( {'params':p , 'lr_mult':1*lr , 'decay_mult':1,'name':'normal weight {}'.format(part_idx)} )
    for part_idx , x in enumerate(zip(normal_bias,lr_for_parts)):
        p , lr = x 
        ret.append( {'params':p , 'lr_mult':1*lr , 'decay_mult':0,'name':'normal bias {}'.format(part_idx)} )
    for part_idx , x in enumerate(zip(bn_weight,lr_for_parts)):
        p , lr =  x
        ret.append( { 'params':p , 'lr_mult':1*lr , 'decay_mult':1 , 'name':"BN weight {}".format(part_idx) } )
    for part_idx , x in enumerate(zip(bn_bias,lr_for_parts)):
        p , lr =  x
        ret.append( { 'params':p , 'lr_mult':1*lr , 'decay_mult':0 , 'name':"BN bias {}".format(part_idx) } )
    return ret
    
    '''
    return [
        #{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1, 'name': "first_conv_weight"},
        #{'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'name': "first_conv_bias"},
        {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "normal_weight"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': "BN scale/shift"},
    ]
    '''

