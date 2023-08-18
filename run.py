#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import argparse
from copy import deepcopy
import time
import typing

import pandas as pd
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import higher
from tqdm import tqdm
from han_nom_dataset import HanNomDatasetNShot
from mobilenetv2 import MobileNetV2
import os
from torch.utils.tensorboard import SummaryWriter
from pickle import dump, load

def main():
    # region arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='Root save dataset', default='demo_ds')
    argparser.add_argument('--exp', type=str, help='Root experiment dataset', default='exp')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--epoch_test', type=int, help='epoch test number', default= 100)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--val_iter', type=int, help='validation iteration', default=10)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--cuda', action='store_true', help='enables cuda')
    argparser.add_argument('--seed', type=int, help='seet for reproduce', default=2103)

    args = argparser.parse_args()
    print(f'args: {args}')
    # endregion

    # region Experiment placeholder
    exp_path = args.exp
    os.makedirs(exp_path, exist_ok = True)
    # endregion

    # region writer tensorboard
    writer = SummaryWriter(
        log_dir=exp_path
    )
    # endregion

    # region set seed to reproduce
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # endregion

    # region Set up the Han-Nom loader.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f'batch_size: {args.task_num}')
    db_train = HanNomDatasetNShot(
        mode = 'train',
        root= args.root,
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=args.imgsz
    )

    db_val = HanNomDatasetNShot(
        mode = 'val',
        root= args.root,
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=args.imgsz
    )

    db_test = HanNomDatasetNShot(
        mode = 'test',
        root= args.root,
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=args.imgsz
    )

    # endregion

    # region get model and optimizer parameters
    net = MobileNetV2(num_classes=3964)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)
    # endregion

    # region Training
    best_model_path = os.path.join(exp_path, 'best_model.pth')
    train(
        db_train=db_train,
        db_val= db_val,
        net = net,
        device = device,
        meta_opt= meta_opt,
        epoch= args.epoch,
        update_step=args.update_step,
        update_step_test= args.update_step_test,
        writer= writer,
        best_model_path=best_model_path,
        epoch_test= args.epoch_test,
        )
    # endregion
    
    # region Testing
    net.load_state_dict(
        torch.load(best_model_path)
    )
    acc, loss = evaluate(
        db = db_test,
        net = net,
        device= device,
        test_epoch= args.epoch_test,
        update_step_test= args.update_step_test
    )
    print(f'Test acc: {acc} - loss: {loss}')
    # endregion

def train(
    db_train: HanNomDatasetNShot,
    db_val: HanNomDatasetNShot,
    net: MobileNetV2,
    device: str,
    meta_opt: optim.Adam,
    epoch: int,
    writer: SummaryWriter,
    best_model_path : str,
    update_step: int = 10,
    update_step_test: int = 10,
    epoch_test: int = 1
    ):
    net = net.to(device)
    net.train()
    best_val_acc = -1

    for step in range(epoch):
        print(f'Step: {step + 1}')
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        # n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(update_step): # Train on support set
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i]) # Test on query set
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward() # Update meta model following task 

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        end_time = time.time()
        print(f'Training step: {step + 1}: {(end_time-start_time)} seconds')
        if step % 5 == 0: # Validation on valid and save model
            print(
                f'[Validation on step {step + 1}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
            )
            writer.add_scalar('Train loss',
                    qry_losses,
                    step)
            
            writer.add_scalar('Train Acc',
                    qry_accs,
                    step)
            
            val_start_time = time.time()
            val_acc, val_loss = evaluate(
                db = db_val,
                net = net,
                test_epoch= epoch_test,
                update_step_test= update_step_test,
                device= device
            )
            val_end_time = time.time()
            print(f'Time to evaluate: {(val_end_time - val_start_time)} seconds')
            postfix = ' (Best)' if val_acc >= best_val_acc else ' (Best: {})'.format(
              best_val_acc)
            
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
              val_loss, val_acc, postfix))
            
            writer.add_scalar('Val loss',
                    val_loss,
                    step)
            
            writer.add_scalar('Val acc',
                    val_acc,
                    step)


            # region save mode
            if val_acc > best_val_acc:
              torch.save(net.state_dict(), best_model_path)
              best_val_acc = val_acc
            # endregion

def evaluate(
    db: HanNomDatasetNShot,
    net: MobileNetV2,
    device: str,
    test_epoch: int, # Run on `test_epoch` and calculate mean and standard deviation accurately
    update_step_test: int
    )-> [float, float]:
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    copy_net = deepcopy(net).to(device)
    qry_accs = []
    qry_losses = []
    # num_task = 0
    for step in range(test_epoch):
        x_spt, y_spt, x_qry, y_qry = db.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        task_num, setsz, c_, h, w = x_spt.size()
        # num_task+=task_num
        querysz = x_qry.size(1)


        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
        for i in range(task_num):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (copy_net, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(update_step_test): # Train on support set
                    spt_logits = copy_net(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = copy_net(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i]) # Test on query set
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                # qry_loss.backward() # Update meta model following task 
    del copy_net
    qry_losses = sum(qry_losses) / len(qry_losses)
    qry_accs = 100. * sum(qry_accs) / len(qry_accs)

    # region calculate mean on valid set
    # endregion
    return qry_accs, qry_losses
    

# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()