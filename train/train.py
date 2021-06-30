import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network
import json

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--save', type=str, default='bench-cifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='11111111', help='which architecture to use')
args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %H:%M:%S')
if not exist os.path.join(args.save, 'log'):
    os.mkdir(os.path.join(args.save, 'log'))
fh = logging.FileHandler(os.path.join(args.save, 'log', '{}.log'.format(utils.get_real_arch(args.arch))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  args.seed += 1
  logging.info("args = %s", args)

  arch = utils.decode_arch(args.arch)
  model = Network(arch, CIFAR_CLASSES)
  model = model.cuda()

  params = utils.count_parameters_in_MB(model)
  logging.info("param size = %fMB", params / 1e6)
  flops = utils.count_FLOPs(model)
  logging.info("FLOPs = %d", flops)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10()
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=True)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=False)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_acc = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
      best_acc = valid_acc
    logging.info('valid_acc %f', valid_acc)
  
  return params, flops, best_acc


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda()

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d/%03d %.3f %.3f %.3f', step, len(train_queue), objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()
  
      logits = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)
  
      if step % args.report_freq == 0:
        logging.info('valid %03d/%03d %.3f %.3f %.3f', step, len(valid_queue), objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  accs = []
  arch = utils.get_real_arch(args.arch)
  result_fp = os.path.join(args.save, arch + '.txt')
  if os.path.exists(result_fp):
    exit(0)
  result = {'arch': arch, 'test_acc': []}
  for _ in range(3):   # train 3 times
    params, flops, test_acc = main() 
    result['params'] = int(params)
    result['flops'] = flops
    result['test_acc'].append(test_acc.cpu().tolist())
  json.dump(result, open(result_fp, 'w'))
