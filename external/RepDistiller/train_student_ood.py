"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill_ood as train_ood, validate_ood
from helper.pretrain import init

import json
import sys
import collections
import numpy as np
import random
import pathlib

sys.path.append('/srv/anisio/does_kd/external/OoD-Bench/external/DomainBed')
from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import model_selection
from domainbed.lib import misc, reporting
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.query import Q


# code adapted from OoD-Bench
def _get_domainbed_dataloaders(args, dataset, hparams):
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    
    loaders = (train_loaders, uda_loaders, eval_loaders)
    others = (eval_weights, eval_loader_names, in_splits)
    return loaders, others


def _get_ood_validation_dataloaders(args, dataset, hparams):
    # A separate set of validation environments (specified by "--val_envs") is used for
    # model selection. The data from training environments are all used for training,
    # and the data from test environments are all used for testing. The holdout data
    # of which the size is specified by "--holdout_fraction" are now a sample of
    # the training data and are used to compute training accuracy, equivalent to the
    # training-environemnt in-split accuarcy in other model selection methods.
    if args.task == 'domain_adaptation' or args.uda_holdout_fraction > 0:
        raise NotImplementedError

    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        if hparams['class_balanced']:
            weights = misc.make_weights_for_balanced_classes(env)
        else:
            weights = None
        if env_i in args.val_envs:
            in_splits.append((None, None))  # dummy placeholder
            out_splits.append((env, weights))
        elif env_i in args.test_envs:
            in_splits.append((None, None))  # dummy placeholder
            out_splits.append((env, weights))
        else:
            out, _ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            in_splits.append((env, weights))
            # add a small sample to check training accuracy
            if hparams['class_balanced']:
                out_weights = misc.make_weights_for_balanced_classes(out)
            else:
                out_weights = None
            out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs + args.val_envs]

    uda_loaders = []

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in out_splits]   # in splits are removed to save computation time
    eval_weights = [None for _, weights in out_splits]

    eval_loader_names = ['env{}_out'.format(i) for i in range(len(dataset))]

    loaders = (train_loaders, uda_loaders, eval_loaders)
    others = (eval_weights, eval_loader_names, in_splits)
    return loaders, others

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet18', 
                                 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    # parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # ood parameters
    parser.add_argument('--teacher_algorithm', type=str, default="ERM")
    parser.add_argument('--student_algorithm', type=str, default="ERM")

    parser.add_argument('--teacher_arch', type=str, choices=['resnet18', 'resnet50'])
    parser.add_argument('--student_arch', type=str, choices=['resnet18', 'resnet50'])

    parser.add_argument('--ood_dir_models', type=str, default=None, help='ood teacher models dir')

    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--val_envs', type=int, nargs='+', default=[],
                        help='Environments for OOD-validation model selection.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--save_model_every_checkpoint', default=True, action='store_true')
    parser.add_argument('--output_dir', type=str, default="train_output")

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    # opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                # opt.gamma, opt.alpha, opt.beta, opt.trial)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, algorithm):
    print('==> loading teacher model')
    checkpoint = torch.load(model_path)
    input_shape = checkpoint['model_input_shape']
    num_classes = checkpoint['model_num_classes']
    num_domains = checkpoint['model_num_domains']
    teacher_arch = checkpoint['model_teacher_arch']
    hparams = checkpoint['model_hparams']
    algorithm_class = algorithms.get_algorithm_class(algorithm)
    algorithm = algorithm_class(input_shape, num_classes, num_domains, teacher_arch, hparams)
    model_dict = checkpoint['model_dict']
    algorithm.load_state_dict(model_dict)
    print('==> done')
    return algorithm
    # model_t = get_teacher_name(model_path)
    # model = model_dict[model_t](num_classes=n_cls)
    # model.load_state_dict(torch.load(model_path)['model'])
    # print('==> done')
    # return model

def _best_model(results_dir, dataset):
    records = reporting.load_records(results_dir)
    # if records and records[0]["args"].get("val_envs", False):
    #     selection_method = model_selection.OODValidationSelectionMethod
    # elif dataset in ['ColoredMNIST_IRM']:
    #     selection_method = model_selection.IIDAccuracySelectionMethod
    # else:
    #     raise NotImplementedError
    
    if dataset in ['PACS', 'OfficeHome', 'TerraIncognita']:
        selection_method = model_selection.IIDAccuracySelectionMethod
    elif dataset in [ 'WILDSCamelyon', 'NICO_Mixed']:
        selection_method = model_selection.OODValidationSelectionMethod
    elif dataset in ['ColoredMNIST_IRM', 'CelebA_Blond']:
        selection_method = model_selection.OracleSelectionMethod
    else:
        raise NotImplementedError

    best_record = selection_method.hparams_accs(records)
    result_path = os.path.join(best_record[0][0]['output_dir'],
                                        'model_step%s.pkl' % (best_record[0][0]['step']))
    print(f'best model path: {result_path}')
    return result_path

def get_best_teacher_model(ood_dir_models, algorithm, dataset, teacher_arch):
    results_dir = os.path.join(ood_dir_models, dataset, teacher_arch)
    best_model_file = _best_model(results_dir, dataset)
    model = load_teacher(best_model_file, algorithm)
    return model

def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(opt.student_algorithm, opt.dataset)
    else:
        hparams = hparams_registry.random_hparams(opt.student_algorithm, opt.dataset,
            misc.seed_hash(opt.hparams_seed, opt.trial_seed))
    if opt.hparams:
        hparams.update(json.loads(opt.hparams))

    hparams['student_arch'] = opt.student_arch
    hparams['teacher_arch'] = opt.teacher_arch

    print('Args:')
    for k, v in sorted(vars(opt).items()):
        print('\t{}: {}'.format(k, v))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
    
    # ---
    os.environ['TORCH_HOME'] = '/tmp/torch/'

    # ---
    os.makedirs(opt.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(opt.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(opt.output_dir, 'err.txt'))
    def save_checkpoint(filename):
        if opt.skip_model_save:
            return
        save_dict = {
            "args": vars(opt),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(opt.test_envs) - len(opt.val_envs),
            "model_hparams": hparams,
            "model_dict": model_s.state_dict(),
            "model_student_algorithm": opt.student_algorithm,
            "model_arch": opt.student_arch
        }
        torch.save(save_dict, os.path.join(opt.output_dir, filename))
    # --

    # dataloader
    if opt.dataset in vars(datasets):
        no_aug_envs = opt.test_envs + opt.val_envs  # no data augmentations
        dataset = vars(datasets)[opt.dataset](opt.data_dir, no_aug_envs, hparams)
        if opt.val_envs:
            loaders, others = _get_ood_validation_dataloaders(opt, dataset, hparams)
        else:
            loaders, others = _get_domainbed_dataloaders(opt, dataset, hparams)
        train_loaders, uda_loaders, eval_loaders = loaders
        eval_weights, eval_loader_names, in_splits = others
    else:
        raise NotImplementedError(opt.dataset)

    # model
    # model_t = load_teacher(opt.path_t, opt.teacher_algorithm)
    model_t = get_best_teacher_model(opt.ood_dir_models, 
                                     opt.teacher_algorithm, 
                                     opt.dataset, 
                                     opt.teacher_arch)

    model_s_alg_class = algorithms.get_algorithm_class(opt.student_algorithm)
    model_s = model_s_alg_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(opt.test_envs) - len(opt.val_envs),
                                opt.student_arch, hparams)

    data = torch.randn(2, 3, 224, 224)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        device = 'cuda'
    else:
        device = 'cpu'

    # validate teacher accuracy
    # evals = zip(eval_loader_names, eval_loaders, eval_weights)
    # for name, loader, weights in evals:
    #     teacher_acc1, teacher_acc5, avglosses = validate_ood(loader, weights, model_t, 
    #                                                          criterion_cls, opt, device)
    #     print('teacher %s acc@1 %.5f acc@5 %.5f losses %.5f' % (name, 
    #                                                             teacher_acc1.item(), 
    #                                                             teacher_acc5.item(), 
    #                                                             avglosses))

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    checkpoint_freq = opt.checkpoint_freq or dataset.CHECKPOINT_FREQ
    steps_per_epoch = min([len(env)/hparams['batch_size'] 
                           for env, _ in in_splits if env is not None])

    # results = {}
    start_step = 0
    last_results_keys = None
    n_steps = opt.steps or dataset.N_STEPS
    for step in range(start_step, n_steps):
        step_start_time = time.time()

        adjust_learning_rate(step, opt, optimizer) # check param_groups
        minibatches_train = [(x.to(device), y.to(device)) 
                              for x,y in next(train_minibatches_iterator)]
        if opt.task == "domain_adaptation":
            minibatches_uda = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            minibatches_uda = None

        train_acc, train_loss = train_ood(step, minibatches_train, minibatches_uda,
                                          module_list, criterion_list, optimizer, opt, device)
        
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        checkpoint_vals['train_loss'].append(train_loss)
        checkpoint_vals['train_acc'].append(train_acc.cpu().numpy().tolist())

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                test_acc1, test_acc5, test_loss = validate_ood(loader, weights, model_s, 
                                                               criterion_cls, opt, device)
                results[name+'_acc'] = test_acc1.cpu().numpy().tolist()
                results[name+'_acc5'] = test_acc5.cpu().numpy().tolist()
                results[name+'_loss'] = test_loss
            
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(opt)
            })

            epochs_path = os.path.join(opt.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = model_s.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if opt.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')


        checkpoint_vals['step_time'].append(time.time() - step_start_time)


    save_checkpoint(f'model_last.pkl')

if __name__ == '__main__':
    main()
