import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from tqdm import tqdm

import copy

from torch.utils.data import Dataset, DataLoader

def nll(logits, y, reduction='mean'):
  return nn.functional.cross_entropy(logits, y, reduction=reduction)
#   return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def mean_accuracy(logits, y):
  try:
    _, preds = torch.max(logits.data, 1)
    return ((preds - y).abs() < 1e-2).float().mean()
  except:
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
  scale = torch.tensor(1.).cuda().requires_grad_()
  loss = nll(logits * scale, y)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)

def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain spurious dataset resampled_set
    Args:
    - dataset (torch.utils.data.Dataset): Spurious correlations dataset
    - resampled_set_indices (int[]): List-like of indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:  # Waterbirds things
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.y_array
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        # print(e)
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except:
            resampled_set_indices = np.concatenate(resampled_set_indices)
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        try:
            resampled_set.df = resampled_set.df.iloc[resampled_set_indices]
        except AttributeError:
            pass
            # resampled_set.data = resampled_set.data[resampled_set_indices]
            
        try:
            resampled_set.data = resampled_set.data[resampled_set_indices]
        except AttributeError:
            pass
    
    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]
    return resampled_set

def split_data_opt(envs, model, n_steps=10000, n_samples=-1, lr=0.001,
                   batch_size=None, join=True, no_tqdm=False,
                   flags=None, train_loader=None):
  """Learn soft environment assignment."""
  model.cuda()
    
  # Hack - only for CMNIST
  num_envs = 5   

  if join and train_loader is None:  # assumes first two entries in envs list are the train sets to joined
    print('pooling envs')
    # pool all training envs (defined as each env in envs[:-1])
    joined_train_envs = dict()
    for k in envs[0].keys():
      if 'dataloader' not in k:
        if envs[0][k].numel() > 1:  # omit scalars previously stored during training
          joined_values = torch.cat([env[k][:n_samples] for env in envs[:-2]],
                                  0)
          joined_train_envs[k] = joined_values
        # Dict? torch.cat of indices
    print('size of pooled envs: %d' % len(joined_train_envs['images']))
    # Need to get the right dataloader using get_resampled_set
    new_dataset = get_resampled_set(
        envs[0]['source_dataloader'].dataset,
        joined_train_envs['indices'],
        copy_dataset=True)
    joined_train_envs['dataloader'] = DataLoader(new_dataset,
                                                 flags.bs_trn,
                                                 shuffle=True,
                                                 num_workers=flags.num_workers)
  elif train_loader is not None:
    joined_train_envs = dict()
    joined_train_envs['dataloader'] = train_loader
    
  else:
    if not isinstance(envs, dict):
      raise ValueError(('When join=False, first argument should be a dict'
                        ' corresponding to the only environment.'
                       ))
    print('splitting data from single env of size %d' % len(envs['images']))
    joined_train_envs = envs

  scale = torch.tensor(1.).cuda().requires_grad_()

  all_logits = []
  all_labels = []
  for ix, data in enumerate(tqdm(joined_train_envs['dataloader'], desc='Collecting logits')):
    inputs, labels, data_ix = data
    inputs = inputs.cuda()
    labels = labels.cuda()

    logits = model(inputs)
    all_logits.append(logits.detach())
    all_labels.append(labels)
  logits = torch.cat(all_logits)
  labels = torch.cat(all_labels)

#   if batch_size:
#     logits = []
#     i = 0
#     num_examples = len(joined_train_envs['labels'])
#     while i < num_examples:
#       images = joined_train_envs['images'][i:i+64]
#       images = images.cuda()
#       logits.append(model(images).detach())
#       i += 64
#     logits = torch.cat(logits)
#   else:
#     logits = model(joined_train_envs['images'])
#     logits = logits.detach()

  loss = nll(logits * scale, labels.cuda(), reduction='none')

  env_w = torch.randn((len(logits), num_envs)).cuda().requires_grad_()
  optimizer = optim.Adam([env_w], lr=lr)

  with tqdm(total=n_steps, position=1, bar_format='{desc}', desc='AED Loss: ', disable=no_tqdm) as desc:
    for i in tqdm(range(n_steps), disable=no_tqdm):
#       # penalty for env a
#       lossa = (loss.squeeze() * env_w.sigmoid()).mean()
#       grada = autograd.grad(lossa, [scale], create_graph=True)[0]
#       penaltya = torch.sum(grada**2)
#       # penalty for env b
#       lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
#       gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
#       penaltyb = torch.sum(gradb**2)
#       # negate
#       npenalty = - torch.stack([penaltya, penaltyb]).mean()
        
      # 5 class colored MNIST - hack
      
      losses = []
      penalties = []
      for i in range(num_envs):
          loss_env = (loss.squeeze() * env_w.softmax(dim=1)[:, i]).mean()
          grad_env = autograd.grad(loss_env, [scale], create_graph=True)[0]
          penalty_env = torch.sum(grad_env**2)
          penalties.append(penalty_env)
      npenalty = - torch.stack(penalties).mean()


      # penalty for env a
#       lossa = (loss.squeeze() * env_w.softmax(dim=1)).mean()
#       grada = autograd.grad(lossa, [scale], create_graph=True)[0]
#       penaltya = torch.sum(grada**2)
#       # penalty for env b
#       lossb = (loss.squeeze() * (1-env_w.softmax(dim=1))).mean()
#       gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
#       penaltyb = torch.sum(gradb**2)
#       # negate
#       npenalty = - torch.stack([penaltya, penaltyb]).mean()
        
        
      # step
      optimizer.zero_grad()
      npenalty.backward(retain_graph=True)
      optimizer.step()
      desc.set_description('AED Loss: %.8f' % npenalty.cpu().item())

  print('Final AED Loss: %.8f' % npenalty.cpu().item())

  # split envs based on env_w threshold
#   new_envs = []
#   env_indices = []
#   for _ in range(len(envs[:-2])):  # train, val, test?
#     env_ix = (env_w.softmax()[]
        
  _, pred_envs = torch.max(env_w.softmax(dim=1), 1)
             
  new_envs = []
  for e in range(num_envs):
    env_ix = np.where(pred_envs.cpu().numpy() == e)[0]
    np.random.shuffle(env_ix)
    env = {}
    dataset = get_resampled_set(train_loader.dataset, 
                                resampled_set_indices=env_ix, 
                                copy_dataset=True)
    env['dataloader'] = DataLoader(dataset,
                                   batch_size=flags.bs_trn,
                                   shuffle=True,
                                   num_workers=flags.num_workers)
    env['labels'] = dataset.targets
    print(f"size of env{e}: {len(env['labels'])}")
    new_envs.append(env)
  return new_envs
              
              
#   env_w.softmax()
#   idx0 = (env_w.sigmoid()>.5)
#   idx1 = (env_w.sigmoid()<=.5)
    
 
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

  model.cuda()
  def _pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  if flags.color_based_eval:  # skip IRM and evaluate color-based model
    from opt_env.utils.model_utils import ColorBasedClassifier
    model = ColorBasedClassifier()
  if not flags.color_based_eval:
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)
    
  tqdm_object = tqdm(range(flags.steps), total=flags.steps, desc='Training')

  max_robust_test_acc = -1
  max_avg_test_acc = -1
  max_avg_test_acc = -1
  early_stopping_counter = 0
  step_ix = 0
    
  all_train_acc = []
  all_robust_test_acc = []
  all_avg_test_acc = []
  for step in tqdm_object:
    for env_, env in enumerate(envs):
      # My code here
      all_logits = []
      all_labels = []
      for ix, data in enumerate(tqdm(env['dataloader'], leave=False, desc=f'> Env {env_}')):
        inputs, labels, data_ix = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        logits = model(inputs)
        all_logits.append(logits)
        all_labels.append(labels)
      logits = torch.cat(all_logits)
      labels = torch.cat(all_labels)
      # print(logits.shape, env['labels'].shape)
      env['nll'] = nll(logits, labels)
      env['acc'] = mean_accuracy(logits, labels).detach().cpu()
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

    val_acc = envs[-flags.n_test_groups - 1]['acc']
#     test_acc = envs[-1]['acc']
    
    test_accs = [e['acc'].numpy() for e in envs[-flags.n_test_groups:]]
    test_accs = np.array(test_accs)
    avg_test_acc = np.mean(test_accs)
    rob_test_acc = np.min(test_accs)
    
    all_train_acc.append(train_acc.numpy())
    all_robust_test_acc.append(rob_test_acc)
    all_avg_test_acc.append(avg_test_acc)
    if step % 100 == 0:
      _pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.numpy(),
        train_penalty.detach().cpu().numpy(),
        val_acc.numpy(),
        avg_test_acc,
        rob_test_acc
      )
    
    tqdm_object.set_postfix(epoch=step,
                            train_nll=train_nll.detach().cpu().numpy(),
                            train_acc=train_acc.numpy(),
                            train_penalty=train_penalty.detach().cpu().numpy(),
                            val_acc=val_acc.numpy(),
                            avg_test_acc=avg_test_acc,
                            rob_test_acc=rob_test_acc)
    
    step_ix += 1
#     if avg_test_acc > max_avg_test_acc:
    if rob_test_acc > max_robust_test_acc:
      max_robust_test_acc = rob_test_acc
      max_avg_test_acc = avg_test_acc
      early_stopping_counter = 0
    else:
      early_stopping_counter += 1

    if early_stopping_counter > 20 and max_robust_test_acc > 0:
      final_train_acc = train_acc.numpy()
      final_avg_test_acc = max_avg_test_acc
      final_rob_test_acc = max_robust_test_acc
      print(f'> Early stopping at epoch: {step}')
        
      all_train_acc = np.array(all_train_acc)
      all_robust_test_acc = np.array(all_robust_test_acc)
      all_avg_test_acc = np.array(all_avg_test_acc)
      
      results = np.stack((all_train_acc, all_avg_test_acc, all_robust_test_acc))
      results_name = f'cmnist_results/r-{flags.exp_name}-rs={flags.restart_ix}.npy'
      with open(results_name, 'wb') as f:
        np.save(f, results)
        
      
      return model, final_train_acc, final_avg_test_acc, final_rob_test_acc, test_accs

  final_train_acc = train_acc.numpy()
  final_avg_test_acc = avg_test_acc
  final_rob_test_acc = rob_test_acc

  all_train_acc = np.array(all_train_acc)
  all_robust_test_acc = np.array(all_robust_test_acc)
  all_avg_test_acc = np.array(all_avg_test_acc)

  results = np.stack((all_train_acc, all_avg_test_acc, all_robust_test_acc))
  results_name = f'cmnist_results/r-{flags.exp_name}-rs={flags.restart_ix}.npy'
  with open(results_name, 'wb') as f:
    np.save(f, results)
  return model, final_train_acc, final_avg_test_acc, final_rob_test_acc, test_accs

