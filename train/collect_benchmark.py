import itertools
import os
import sys
import time
import numpy as np
import json


def get_real_arch(arch, stages=[2, 3, 3]):
  arch = list(arch)
  result = ''
  for stage in stages:
    id_num = 0
    for idx in range(stage):
      op = arch.pop(0)
      if idx == 0:
        result += op
        continue
      if op != '0':
        result += op
      else:
        id_num += 1
    result += '0' * id_num
  return result


'''read all files'''
data = {}
for f in os.listdir('bench-cifar10'):
    if not f.endswith('txt'):
        continue
    arch = json.load(open(os.path.join('bench-cifar10', f), 'r'))
    data[arch['arch']] = {'test_acc': arch['test_acc'], 'mean_acc': np.mean(arch['test_acc']), 'std': np.std(arch['test_acc']), 'params': arch['params'], 'flops': arch['flops']}

choices = ['0', '1', '2']
layers = 8
for idx, arch in enumerate(itertools.product(*[choices]*layers)):
    arch = ''.join(arch)
    data[arch] = data[get_real_arch(arch)]

print('data length: ', len(data))
json.dump(data, open('nas-bench-macro_cifar10.json', 'w'))