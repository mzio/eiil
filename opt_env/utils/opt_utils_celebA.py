"""
Version of EIIL for CelebA

Main idea: the main function is split_data_opt
- We just need to feed in the CelebA dataset and a trained model
- Run inference to compute losses
- Then use these losses to update the model
"""
import os

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

def nll(logits, y, reduction='mean'):
  return nn.functional.cross_entropy(logits, y, reduction=reduction)
#   print(logits[:10])
#   return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def mean_accuracy(logits, y):
#   try:
#     _, preds = torch.max(logits.data, 1)
#     return ((preds - y).abs() < 1e-2).float().mean()
#   except:
  preds = (logits > 0.).float()
  return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
  scale = torch.tensor(1.).cuda().requires_grad_()
  loss = nll(logits * scale, y)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)


def split_data_opt(train_loader, model, 
                   n_steps=10000, n_samples=-1, lr=0.001,
                   batch_size=None, join=True, no_tqdm=False,
                   flags=None):
  """Learn soft environment assignment."""

  if flags.no_cuda is True:
    scale = torch.tensor(1.).requires_grad_()
  else:
    scale = torch.tensor(1.).cuda().requires_grad_()

  if flags.logits_path is not None:
    fname = flags.logits_path
    
    if os.path.exists(fname):
      print(f'> Loading logits from {fname}')
      with open(fname, 'rb') as f:
        logits = torch.tensor(np.load(f))

      labels = train_loader.dataset.targets
      if flags.no_cuda is False:
        logits = logits.cuda()
        labels = labels.cuda()
      loss = nll(logits * scale, labels, reduction='none')
  else:
    model.to(torch.device('cpu'))
    
    model_name = flags.pretrained_model_path.split('/')[-1].split('.pt')[0]
    fname = f'logits-d={flags.dataset}-m={model_name}-o=Adam_nsteps={n_steps}-nsamples={n_samples}-lr={lr}.npy'
    fname = os.path.join(flags.results_dir,fname)
    if os.path.exists(fname):
      print(f'> Loading logits from {fname}')
      with open(fname, 'rb') as f:
        logits = torch.tensor(np.load(f))

      labels = train_loader.dataset.targets
      if flags.no_cuda is False:
        logits = logits.cuda()
        labels = labels.cuda()
      loss = nll(logits * scale, labels, reduction='none')
  
    else:
      all_logits = []
      all_labels = []
      all_loss = []
      num_correct = 0
      num_total = 0
      for ix, data in enumerate(tqdm(train_loader, 
                                     desc='Saving logits')):
        inputs, labels, data_ix = data

        if flags.no_cuda:
            pass
        else:
            model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        logits = model(inputs)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).detach().cpu()
        num_correct += correct.sum()
        num_total += len(labels)

        if flags.no_cuda:
          logits = logits.to(torch.device('cpu'))
          labels = labels.to(torch.device('cpu'))

        loss = nll(logits * scale, labels, reduction='none')
        # loss = loss.detach().cpu()

        all_loss.append(loss)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        labels = labels.detach().cpu()
        inputs = inputs.detach().cpu()
        del inputs; del labels; del logits

        model.to(torch.device('cpu'))
    
      print(f'Average accuracy: {num_correct / num_total * 100:<.2f}%')

      logits = torch.cat(all_logits)
      labels = torch.cat(all_labels)
      loss = torch.cat(all_loss)

      model_name = flags.pretrained_model_path.split('/')[-1].split('.pt')[0]
      fname = f'logits-d={flags.dataset}-m={model_name}-o=Adam_nsteps={n_steps}-nsamples={n_samples}-lr={lr}.npy'

      fname = os.path.join(flags.results_dir,fname)
      with open(fname, 'wb') as f:
        np.save(f, logits.numpy())
        print(f'Saved logits to {fname}!')

  

  if flags.no_cuda is True:
    env_w = torch.randn(len(logits)).requires_grad_()
  else:
    env_w = torch.randn(len(logits)).cuda().requires_grad_()
    loss = loss.cuda()
  
  optimizer = optim.Adam([env_w], lr=lr)

  with tqdm(total=n_steps, position=1, bar_format='{desc}', desc='AED Loss: ', disable=no_tqdm) as desc:
    for i in tqdm(range(n_steps), disable=no_tqdm):
      # penalty for env a
      lossa = (loss.squeeze() * env_w.sigmoid()).mean()
      grada = autograd.grad(lossa, [scale], create_graph=True)[0]
      penaltya = torch.sum(grada**2)
      # penalty for env b
      lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
      gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
      penaltyb = torch.sum(gradb**2)
      # negate
      npenalty = - torch.stack([penaltya, penaltyb]).mean()     
        
      # step
      optimizer.zero_grad()
      npenalty.backward(retain_graph=True)
      optimizer.step()
      desc.set_description('AED Loss: %.8f' % npenalty.cpu().item())

  print('Final AED Loss: %.8f' % npenalty.cpu().item())

  # split envs based on env_w threshold
  new_envs = []
  idx0 = (env_w.sigmoid()>.5)
  idx1 = (env_w.sigmoid()<=.5)

  print(idx0.numpy())
  print(idx1.numpy())
  print(np.where(idx0.numpy())[0])
  print(np.where(idx1.numpy())[0])

  env_preds = np.zeros(len(env_w))
  env_preds[np.where(idx1.numpy())[0]] += 1

#   env_indices = np.stack([np.where(idx0.numpy())[0],
#                           np.where(idx1.numpy())[0]])
#   print(env_indices.shape)
#   try:
#     model_name = flags.pretrained_model_path.split('/')[-1].split('.pt')[0]
  if flags.logits_path is not None:
    model_name = flags.logits_path.split('/')[-1].split('.npy')[0]
  fname = f'env_preds-d={flags.dataset}-m={model_name}-o=Adam-nsteps={n_steps}-nsamples={n_samples}-lr={flags.lr}-wd={flags.l2_regularizer_weight}.npy'
    
  fname = os.path.join(flags.results_dir, fname)
  with open(fname, 'wb') as f:
    np.save(f, env_preds)
    print(f'Saved EIIL predictions to {fname}!')
    
    return env_preds
  # train envs
  # NOTE: envs include original data indices for qualitative investigation
  for _idx in env_indices:
    new_env = dict()
    for k, v in joined_train_envs.items():
      if k == 'paths':  # paths is formatted as a list of str, not ndarray or tensor
        v_ = np.array(v)
        v_ = v_[_idx.cpu().numpy()]
        v_ = list(v_)
        new_env[k] = v_
      else:
        new_env[k] = v[_idx]
    new_envs.append(new_env)
  print('size of env0: %d' % len(new_envs[0]['labels']))
  print('size of env1: %d' % len(new_envs[1]['labels']))

  if join:  #NOTE: assume the user includes test set as part of arguments only if join=True
    new_envs.append(envs[-1])
    print('size of env2: %d' % len(new_envs[-1]['labels']))
  return new_envs


def train_irm_batch(model, envs, flags):
  """Batch version of the IRM algo for CMNIST expers."""
  def _pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".os.path.join(str_values))

  if flags.color_based_eval:  # skip IRM and evaluate color-based model
    from opt_env.utils.model_utils import ColorBasedClassifier
    model = ColorBasedClassifier()
  if not flags.color_based_eval:
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)
  for step in range(flags.steps):
    for env in envs:
      # My code here
      all_logits = []
      all_labels = []
      for ix, data in enumerate(tqdm(env['dataloader'])):
        inputs, labels, data_ix = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        logits = model(inputs)
        all_logits.append(logits)
        all_labels.append(labels)
      logits = torch.cat(all_logits)
      labels = torch.cat(all_labels)
      print(logits.shape, env['labels'].shape)
      env['nll'] = nll(logits, labels)
      env['acc'] = mean_accuracy(logits, labels)
      env['penalty'] = penalty(logits, labels)
      # End my code here
#       logits = model(env['images'])
#       env['nll'] = nll(logits, env['labels'])
#       env['acc'] = mean_accuracy(logits, env['labels'])
#       env['penalty'] = penalty(logits, env['labels'])

#     train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
#     train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
#     train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
    train_nll = torch.stack([env['nll'] for env in envs[:-2]]).mean()
    train_acc = torch.stack([env['acc'] for env in envs[:-2]]).mean()
    train_penalty = torch.stack([env['penalty'] for env in envs[:-2]]).mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in model.parameters():
      weight_norm += w.norm().pow(2)
    loss = train_nll.clone()
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    if not flags.color_based_eval:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    val_acc = envs[-2]['acc']
    test_acc = envs[-1]['acc']
    if step % 100 == 0:
      _pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        val_acc.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_acc = train_acc.detach().cpu().numpy()
  final_test_acc = test_acc.detach().cpu().numpy()
  return model, final_train_acc, final_test_acc

