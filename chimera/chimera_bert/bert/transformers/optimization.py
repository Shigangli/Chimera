# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
from transformers.utils import is_main_process

import torch.distributed as distrib

multi_tensor_l2norm = amp_C.multi_tensor_l2norm
lamb_compute_update = amp_C.multi_tensor_lamb_stage1_cuda
lamb_apply_update = amp_C.multi_tensor_lamb_stage2_cuda
scale = amp_C.multi_tensor_scale

import time

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0, flush_group=None, flush_group_size=None, stage_id=None, num_stages=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        self.flush_group = flush_group
        self.scale = 1.0/flush_group_size
        self.flush_group_size = flush_group_size

        self.stage_id = stage_id

        self.grad_shapes = []
        self.grad_sizes = []
        self.counter = 0
        self.boundaries = []
        self.grad_stages = []
        self.handles = []


        #16d 48layers
        if num_stages == 16:
            if stage_id == 0:
                self.boundaries = [1, 9, 15]
            elif stage_id == num_stages - 1:
                self.boundaries = [6, 12, 20]
            else:
                self.boundaries = [6, 12]


        #8d 48layers
        if num_stages == 8:
            if stage_id == 0:
                self.boundaries = [1, 15, 27]
            elif stage_id == num_stages - 1:
                self.boundaries = [12, 24, 38]
            else:
                self.boundaries = [12, 24]

        #4d 48layers
        if num_stages == 4:
            if stage_id == 0:
                self.boundaries = [1, 15, 27, 39, 51, 63]
            elif stage_id == num_stages - 1:
                self.boundaries = [12, 24, 36, 48, 60, 74]
            else:
                self.boundaries = [12, 24, 36, 48, 60]

        #2d 48layers
        if num_stages == 2:
            if stage_id == 0:
                self.boundaries = [1, 27, 51, 75, 99, 123]
            elif stage_id == num_stages - 1:
                self.boundaries = [24, 48, 72, 96, 120, 144]
            else:
                self.boundaries = [24, 48, 72, 96, 120]

        self.stages = len(self.boundaries) + 1

        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def grads_sync(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        #start_time = time.time()
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                if self.counter == 0:
                    self.grad_shapes.append(p.grad.data.shape)
                    self.grad_sizes.append(torch.numel(p.grad.data))
                grads.append(p.grad.data.view(-1))
                
        self.handles = [] 
        self.grad_stages = []
        

        for i in range(len(self.boundaries)):
            if i == 0: 
                self.grad_stages.append(torch.cat(grads[:self.boundaries[i]]) * self.scale)
                self.handles.append(distrib.all_reduce(self.grad_stages[i], op=distrib.ReduceOp.SUM, group=self.flush_group, async_op=True))
                self.grad_stages.append(torch.cat(grads[self.boundaries[i]:self.boundaries[i+1]]) * self.scale)
                self.handles.append(distrib.all_reduce(self.grad_stages[i+1], op=distrib.ReduceOp.SUM, group=self.flush_group, async_op=True))
            elif i == len(self.boundaries)-1:
                self.grad_stages.append(torch.cat(grads[self.boundaries[i]:]) * self.scale)
                self.handles.append(distrib.all_reduce(self.grad_stages[i+1], op=distrib.ReduceOp.SUM, group=self.flush_group, async_op=True))
            else:
                self.grad_stages.append(torch.cat(grads[self.boundaries[i]:self.boundaries[i+1]]) * self.scale)
                self.handles.append(distrib.all_reduce(self.grad_stages[i+1], op=distrib.ReduceOp.SUM, group=self.flush_group, async_op=True))

        self.counter += 1

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        #start_time = time.time()

        self.handles[0].wait()
        split_stage_grads = torch.split(self.grad_stages[0], self.grad_sizes[:self.boundaries[0]])

        indx = 0
        cstage = 0
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                #grad = split_stage_grads[indx-offset].view(self.grad_shapes[indx])/self.flush_group_size
                grad = split_stage_grads[indx-offset].view(self.grad_shapes[indx])
                assert grad.shape == p.grad.data.shape

                state = self.state[p]
                # State initialization.
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values.
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values.
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping.
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient.
                # In-place operations to update the averages at the same time.
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                indx += 1
                if cstage < self.stages-1:
                    if indx == self.boundaries[cstage]:
                        offset = self.boundaries[cstage]
                        cstage += 1
                        self.handles[cstage].wait()
                        if cstage < self.stages-1:
                            split_stage_grads = torch.split(self.grad_stages[cstage], self.grad_sizes[self.boundaries[cstage-1]:self.boundaries[cstage]])
                        else:
                            split_stage_grads = torch.split(self.grad_stages[cstage], self.grad_sizes[self.boundaries[cstage-1]:])

        self.grad_stages = []
        self.handles = [] 
        #print("stage ", self.stage_id, "step time: ", (time.time()-start_time))

        return loss
