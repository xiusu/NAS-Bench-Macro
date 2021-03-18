import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def _data_transforms_cifar10():
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())


def count_FLOPs(model, input_shape=[3, 32, 32]):
  def conv2d_flops(m: nn.Conv2d, input_shape):
    c, w, h = input_shape
    c = m.out_channels
    w = (w + m.padding[0] * 2 - m.kernel_size[0]) // m.stride[0] + 1
    h = (h + m.padding[1] * 2 - m.kernel_size[1]) // m.stride[1] + 1
    flops = m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1]
    return flops, (c, w, h)

  total_flops = []
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
        flops, input_shape = conv2d_flops(m, input_shape)
        total_flops.append(flops)
    elif isinstance(m, nn.Linear):
        flops = m.in_features * m.out_features
        total_flops.append(flops)
    else:
        pass
  return sum(total_flops)


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def decode_arch(arch):
  return [int(x) for x in arch]


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

