import argparse
import os
import pdb
import pickle
import sys

import numpy as np
import torch
import torchvision
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
from tqdm import tqdm

from opt_env.utils.env_utils import get_envs
from opt_env.utils.opt_utils_celebA import split_data_opt
from opt_env.utils.opt_utils import train_irm_batch
from opt_env.utils.model_utils import load_mlp
from opt_env.utils.model_utils import ColorBasedClassifier

from opt_env.utils.colored_mnist import get_envs as get_cmnist_envs
from opt_env.utils.celebA import get_envs as get_celebA_envs, load_dataloaders

from collections import OrderedDict


def get_net(flags, pretrained=None):
    """
    Return model architecture
    """
    if 'cnn' in flags.arch:
        net = CNN(num_classes=flags.num_classes)
    elif 'resnet' in flags.arch:
        if 'resnet50' in flags.arch:
            pretrained = True if '_pt' in flags.arch else False
            net = torchvision.models.resnet50(pretrained=pretrained)
            d = net.fc.in_features
            net.fc = nn.Linear(d, flags.num_classes)
        elif flags.arch == 'resnet34':
            pretrained = True if '_pt' in flags.arch else False
            net = torchvision.models.resnet34(pretrained=pretrained)
            d = net.fc.in_features
            net.fc = nn.Linear(d, flags.num_classes)
        net.activation_layer = 'avgpool'
    else:
        raise NotImplementedError
    return net


def load_pretrained_model(path, flags):
    checkpoint = torch.load(path)
    net = get_net(flags)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    return net


# Added for evaluation on CMNIST
class CNN(nn.Module):
  def __init__(self, num_classes):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5
    self.fc2 = nn.Linear(120, 84)  # Activations layer
    self.fc = nn.Linear(84, num_classes)
    self.relu_1 = nn.ReLU()
    self.relu_2 = nn.ReLU()

    self.activation_layer = torch.nn.ReLU

  def forward(self, x):
    # Doing this way because only want to save activations
    # for fc linear layers - see later
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = self.relu_1(self.fc1(x))
    x = self.relu_2(self.fc2(x))
    x = self.fc(x)
    return x


def main(flags):
    
  torch.manual_seed(flags.seed)
    
  if flags.dataset == 'colored_mnist':
    get_envs = get_cmnist_envs
    flags.num_classes = 5
  elif flags.dataset == 'celebA':
    get_envs = get_celebA_envs
    flags.num_classes = 2
  
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
    
  flags.results_dir += f'/d={flags.dataset}_o={flags.optim}_lr={flags.lr}_wd={flags.weight_decay}_m={flags.momentum}_s={flags.seed}'
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  
  # save this file and command for reproducibility
  if flags.results_dir != '.':
    with open(__file__, 'r') as f:
      this_file = f.readlines()
      with open(os.path.join(flags.results_dir, 'irm_cmnist.py'), 'w') as f: 
        f.write(''.join(this_file))
    cmd = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(flags.results_dir, 'command.sh'), 'w') as f:
      f.write(cmd)
  # save params for later
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  pickle.dump(flags, open(os.path.join(flags.results_dir, 'flags.p'), 'wb'))
  for f in sys.stdout, open(os.path.join(flags.results_dir, 'flags.txt'), 'w'):
    print('Flags:', file=f)
    for k,v in sorted(vars(flags).items()):
      print("\t{}: {}".format(k, v), file=f)
      
  print('results will be found here:')
  print(flags.results_dir)
    
  if flags.pretrained_model_path != '':
    print('> Loading pretrained model')
    erm_model = load_pretrained_model(flags.pretrained_model_path,
                                      flags)
    erm_model.eval()
    
    dataloaders = load_dataloaders(flags, train_shuffle=False,
                                   transform=None)
    
    env_indices = split_data_opt(dataloaders[0], erm_model,
                                 flags=flags)
    return

  elif flags.logits_path is not None:
    print('> Loading logits')
#     erm_model = load_pretrained_model(flags.pretrained_model_path,
#                                       flags)
#     erm_model.eval()
    
    dataloaders = load_dataloaders(flags, train_shuffle=False,
                                   transform=None)
    
    env_indices = split_data_opt(dataloaders[0], None, lr=flags.lr,
                                 flags=flags)
    return
  

  final_train_accs = []
  final_test_accs = []
  for restart in range(flags.n_restarts):
    print("Restart", restart)

    rng_state = np.random.get_state()
    np.random.set_state(rng_state)

    # Build environments
    envs = get_envs(flags=flags)
    for env in envs:
      print(env.keys())

    # Define and instantiate the model
    if flags.color_based:  # use color-based reference classifier without trainable params
      mlp_pre = ColorBasedClassifier()
    elif flags.dataset in ['colored_mnist', 'celebA']:
      mlp_pre = get_net(flags).cuda()
    else:
      mlp_pre = load_mlp(results_dir=None, flags=flags).cuda()  # reference classifier
    if flags.dataset in ['colored_mnist', 'celebA']:
      mlp = get_net(flags)
    else:
      mlp = load_mlp(results_dir=None, 
                     flags=flags).cuda()  # invariant representation learner
    mlp_pre.train()
    mlp.train()

    # Define loss function helpers

    def nll(logits, y, reduction='mean'):
      # return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)
      return nn.functional.cross_entropy(logits, y, reduction=reduction)

    def mean_accuracy(logits, y):
      try:
        _, preds = torch.max(logits.data, 1)
        return ((preds - y).abs() < 1e-2).float().mean()
      except Exception as e:
        print(e)
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
      scale = torch.tensor(1.).cuda().requires_grad_()
      loss = nll(logits * scale, y)
      grad = autograd.grad(loss, [scale], create_graph=True)[0]
      return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
      col_width = 13
      def format_val(v):
        if not isinstance(v, str):
          v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
      str_values = [format_val(v) for v in values]
      print("   ".join(str_values))


    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    if flags.eiil:
      if flags.color_based:
        print('Color-based refernece classifier was specified, to skipping pre-training.') 
      else:
        if flags.optim == 'Adam':
          optimizer_pre = optim.Adam(mlp_pre.parameters(), lr=flags.lr)
        else:
          optimizer_pre = optim.SGD(mlp_pre.parameters(), lr=flags.lr,
                                    momentum=flags.momentum,
                                    weight_decay=flags.weight_decay)
        for step in range(flags.steps):
          # 2 envs
          accs_0 = []
          accs_1 = []
          penalties_0 = []
          penalties_1 = []
          nlls_0 = []
          nlls_1 = []
        
          all_train_nll = []
          all_train_acc = []
          all_train_penalty = []
          
          dataloader_iterator = iter(envs[0]['dataloader'])
            
          for i, data1 in enumerate(envs[1]['dataloader']):
            try:
              data0 = next(dataloader_iterator)
            except StopIteration:
              dataloader_iterator = iter(envs[0]['dataloader'])
              data0 = next(dataloader_iterator)
                
            inputs0, labels0, data_ix0 = data0
            inputs1, labels1, data_ix1 = data1
            
            inputs0 = inputs0.cuda()
            labels0 = labels0.cuda()
            
            logits0 = mlp_pre(inputs0)
            logits1 = mlp_pre(inputs1)
            
            nll0 = nll(logits0, labels0)
            nll1 = nll(logits1, labels1)
            
            penalty0 = penalty(logits0,
                               labels0).detach().cpu()
            penalties_0.append(penalty0)
            penalty1 = penalty(logits1,
                               labels1).detach().cpu()
            penalties_1.append(penalty1)
            
            acc0 = mean_accuracy(logits0, labels0)
            acc0 = acc.detach().cpu()
            accs_0.append(acc0)
            
            acc1 = mean_accuracy(logits1, labels1)
            acc1 = acc.detach().cpu()
            accs_1.append(acc1)
            
            train_nll = torch.stack([nll0, nll1]).mean()
            train_acc = torch.stack([acc0, acc1]).mean()
            train_penalty = torch.stack([penalty0, 
                                         penalty1]).mean()

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp_pre.parameters():
              weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm

            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()
            
            train_nll = train_nll.detach().cpu()
            
            nll0 = nll0.detach().cpu()
            nll1 = nll1.detach().cpu()
            nlls_0.append(nll0)
            nlls_1.append(nll1)
            
            inputs0 = inputs0.detach().cpu()
            labels0 = labels0.detach().cpu()
            inputs1 = inputs1.detach().cpu()
            labels1 = labels1.detach().cpu()
            
            all_train_nll.append(train_nll)
            all_train_acc.append(train_acc)
            all_train_penalty.append(train_penalty)
          
          env['nll'] = torch.mean([nlls_0, nlls_1])
          env['acc'] = torch.mean([acc_0, acc_1])
          env['penalty'] = torch.mean([penalties_0, penalties_1])
          
          val_acc = envs[-2]['acc']
          test_acc = envs[-1]['acc']
          if step % 100 == 0:
            pretty_print(
              np.int32(step),
              torch.cat(all_train_nll).numpy(),
              torch.cat(all_train_acc).numpy(),
              torch.cat(all_train_penalty).numpy(),
              val_acc.detach().cpu().numpy(),
              test_acc.detach().cpu().numpy()
            )
      torch.save(mlp_pre.state_dict(), 
                 os.path.join(flags.results_dir, 'mlp_pre.%d.p' % restart))
      envs = split_data_opt(envs, mlp_pre, flags)
        
    mlp_pre = mlp_pre.to(torch.device('cpu'))
    mlp = mlp.cuda()
    mlp.train()
    mlp, final_train_acc, final_test_acc = train_irm_batch(mlp, envs, flags)
    final_train_accs.append(final_train_acc)
    final_test_accs.append(final_test_acc)
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('done with restart %d' % restart)
    torch.save(mlp.state_dict(), 
               os.path.join(flags.results_dir, 'mlp.%s.p' % restart))

  print('done with all restarts')
  final_train_accs = [t.item() for t in final_train_accs]
  final_test_accs = [t.item() for t in final_test_accs]
  metrics = {'Train accs': final_train_accs,
             'Test accs': final_test_accs}
  with open(os.path.join(flags.results_dir, 'metrics.p'), 'wb') as f:
    pickle.dump(metrics, f)

  print('results are here:')
  print(flags.results_dir)

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IRM Colored MNIST')
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--n_restarts', type=int, default=1)
  parser.add_argument('--penalty_anneal_iters', type=int, default=100)
  parser.add_argument('--penalty_weight', type=float, default=10000.0)
  parser.add_argument('--steps', type=int, default=5001)
  parser.add_argument('--grayscale_model', action='store_true')
  parser.add_argument('--eiil', action='store_true')
  parser.add_argument('--results_dir', type=str,
                      default='/tmp/opt_env/irm_cmnist', 
                      help='Directory where results should be saved.')
  parser.add_argument('--label_noise', type=float, default=0.25)
  parser.add_argument('--train_env_1__color_noise', type=float, default=0.2)
  parser.add_argument('--train_env_2__color_noise', type=float, default=0.1)
  parser.add_argument('--test_env__color_noise', type=float, default=0.9)
  parser.add_argument('--color_based', action='store_true')  # use color-based reference classifier without trainable params
  parser.add_argument('--color_based_eval', action='store_true')  # skip IRM phase and evaluate color-based classifier

  # MY ADDITIONS
  parser.add_argument('--arch', type=str, default='cnn')
  parser.add_argument('--no_cuda', default=False,
                      action='store_true')
  parser.add_argument('--pretrained_model_path', type=str,
                      default='')
  parser.add_argument('--logits_path', type=str,
                      default=None)
  parser.add_argument('--dataset', type=str,
                      default='colored_mnist')
  parser.add_argument('--bs_trn', type=int, default=32)
  parser.add_argument('--bs_val', type=int, default=32)
  parser.add_argument('--val_split', type=float, default=0.2)
  parser.add_argument('--seed', type=float, default=42)
  parser.add_argument('--num_workers', type=int, default=2)
  parser.add_argument('--optim', type=str, default='SGD')
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  # Colored MNIST specific
  # - Ignored if flags.dataset != 'colored_mnist'
  parser.add_argument('--data_cmap', type=str, default='hsv',
                        help="Color map for digits. If solid, color all digits the same color")
  parser.add_argument('--test_cmap', type=str, default='',
                        help="Color map for digits. Solid colors applies same color to all digits. Only applies if specified, and automatically changes test_shift to 'generalize'")
  parser.add_argument('-pc', '--p_correlation', type=float, default=0.995,
                    help="Ratio of majority group size to total size")
  parser.add_argument('-pcc', '--p_corr_by_class', type=float, nflags='+', action='append',
                        help="If specified, p_corr for each group, e.g. -pcc 0.9 -pcc 0.9 -pcc 0.9 -pcc 0.9 -pcc 0.9 is the same as -pc 0.9")
  parser.add_argument('-tc', '--train_classes', type=int, nflags='+', action='append',
                        help="How to set up the classification problem, e.g. -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9")
  parser.add_argument('-tcr', '--train_class_ratios', type=float, nflags='+', action='append',
                        help="If specified, introduce class imbalance by only including the specified ratio of datapoints per class, e.g. for original ratios: -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 ")
  parser.add_argument('--test_shift', type=str, default='random',
                        help="How to shift the colors encountered in the test set - choices=['random', 'unseen', 'iid', 'shift_n' 'generalize']")
  parser.add_argument('--flipped', default=False, action='store_true',
                      help="If true, color background and leave digit white")

  flags = parser.parse_flags()

  if flags.dataset == 'colored_mnist':
    flags.root_dir = './datasets/data/'
    flags.data_path = './datasets/data/'
    flags.target_name = 'digit'
    flags.confounder_names = ['color']
    flags.image_mean = 0.5
    flags.image_std = 0.5
    flags.augment_data = False
  elif flags.dataset == 'celebA':
    flags.root_dir = '/dfs/scratch0/nims/CelebA/celeba/'
    # IMPORTANT - dataloader assumes that we have directory structure
    # in ./datasets/data/CelebA/ :
    # |-- list_attr_celeba.csv
    # |-- list_eval_partition.csv
    # |-- img_align_celeba/
    #     |-- image1.png
    #     |-- ...
    #     |-- imageN.png
    flags.target_name = 'Blond_Hair'
    flags.confounder_names = ['Male']
    flags.image_mean = np.mean([0.485, 0.456, 0.406])
    flags.image_std = np.mean([0.229, 0.224, 0.225])
    flags.augment_data = False
    flags.image_path = './images/celebA/'
    flags.train_classes = ['blond', 'nonblond']
    flags.val_split = 0.2
    
  elif flags.dataset == 'waterbirds':
    flags.root_dir = '/dfs/scratch1/mzhang/projects/slice-and-dice-smol/datasets/data/Waterbirds'
    flags.target_name = 'waterbird_complete95'
    flags.confounder_names = ['forest2water2']
    flags.image_mean = np.mean([0.485, 0.456, 0.406])
    flags.image_std = np.mean([0.229, 0.224, 0.225])
    flags.augment_data = False
    flags.train_classes = ['landbirds', 'waterbirds']
  torch.cuda.set_device(0)
  main(flags)
