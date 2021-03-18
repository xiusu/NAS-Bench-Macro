import itertools
import os
import sys
import time

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

choices = ['0', '1', '2']
layers = 8

space_size = len(list(itertools.product(*[choices]*layers)))

evaluating_archs = {}
for idx, arch in enumerate(itertools.product(*[choices]*layers)):
    arch = ''.join(arch)
    if get_real_arch(arch) in evaluating_archs:
        print('Already evaluated.')
        continue
    evaluating_archs[get_real_arch(arch)] = 1
    if os.path.exists(os.path.join('bench-cifar10', '{}.txt'.format(get_real_arch(arch)))):
        print('Already evaluated.')
        continue
    print('Evaluating ({}/{}): {}'.format(idx, space_size, arch))
    os.system('python train.py --arch {}'.format(get_real_arch(arch)))

print('Evaluation done.')
