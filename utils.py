import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from main import CircleParams
from dataload import InputPipeline
from setting import Setting
from model import Network

def iou(a: CircleParams, b: CircleParams) -> float:
  """Calculate the intersection over union of two circles"""
  r1, r2 = a.radius, b.radius
  d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
  if d > r1 + r2:
    return 0
  if d <= abs(r1 - r2):
    return 1
  r1_sq, r2_sq = r1 ** 2, r2 ** 2
  d1 = (r1_sq - r2_sq + d ** 2) / (2 * d)
  d2 = d - d1
  h1 = r1_sq * np.arccos(d1 / r1)
  h2 = d1 * np.sqrt(r1_sq - d1 ** 2)
  h3 = r2_sq * np.arccos(d2 / r2)
  h4 = d2 * np.sqrt(r2_sq - d2 ** 2)
  intersection = h1 + h2 + h3 + h4
  union = np.pi * (r1_sq + r2_sq) - intersection
  return intersection / union


def np_to_circleparams(labels: torch.tensor):
    ''' Converts numpy array to CircleParams class'''
    labels = np.transpose(labels)
    x, y, r = labels
    output = []
    for curr_row, curr_col, curr_radius in zip(x, y, r):
        output.append(CircleParams(curr_row, curr_col, curr_radius))
    return output


def batch_sum_iou(targets, outputs):
    ''' Returns the iou score of one batch'''
    arr = []
    for curr_target, curr_output in zip(targets, outputs):
        arr.append(iou(curr_target, curr_output))
    return np.sum(arr)

def get_dataloader(setting, train):
    ''' Get train/valid dataloader'''
    dataset = InputPipeline(setting, setting.DATA_TRAIN if train else setting.DATA_VAL)
    dataloader = DataLoader(
        dataset,
        batch_size=setting.batch_size,
    )
    return dataloader


def get_low_dataloader(setting, train):
    '''	Get train/valid dataloader for lower noise images'''
    dataset = InputPipeline(setting, setting.LOW_TRAIN if train else setting.LOW_VAL)
    dataloader = DataLoader(
        dataset,
        batch_size=setting.batch_size,
    )
    return dataloader


def get_optimizer(hyp, network):
  '''
      Get optimizer with custom weight decay
    '''

  g0, g1, g2 = [], [], []  # optimizer parameter groups
  for v in network.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
      g2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
      g0.append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
      g1.append(v.weight)

  if hyp['adam']:
    optimizer = torch.optim.Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
  else:
    optimizer = torch.optim.SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

  optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
  optimizer.add_param_group({'params': g2})  # add g2 (biases)

  del g0, g1, g2

  return optimizer

def mae_loss(pred, target):
    ''' returns the mae loss of row, col, radius'''
    x_pred, y_pred, r_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_target, y_target, r_target = target[:, 0], target[:, 1], target[:, 2]
    x_error = torch.mean(torch.abs(x_pred - x_target))
    y_error = torch.mean(torch.abs(y_pred - y_target))
    r_error = torch.mean(torch.abs(r_pred - r_target))
    return x_error + y_error + r_error


def get_network(setting, device):
    ''' returns the network, and loads the weights if available'''
    network = Network()
    if setting.pretrained != '':
        pretrained_weights = torch.load(setting.pretrained)
        network.load_state_dict(pretrained_weights)
    return network.to(device)

def show_circle(img: np.ndarray):
    '''plots the circle'''
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Circle')
    plt.show()
