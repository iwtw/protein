from torch._six import container_abcs
from torch._six import string_classes, int_classes, FileNotFoundError
from torch.utils.data.dataloader import default_collate
import re


a= [  [[1,2,3],[2],[3]] , [[1],[2]]  ] 

print( default_collate(  a ) )
