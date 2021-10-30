# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)
import torch.distributed as distrib


def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True, flush_group=None, flush_group_size=None, stage_id=None, num_stages=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
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

        #64 layers 64d
        if num_stages == 64:
            if stage_id == 0:
                self.boundaries = [1, 5, 7]
            elif stage_id == num_stages - 1:
                self.boundaries = [3, 5, 7]
            else:
                self.boundaries = [3, 5]

        #64 layers 32d
        if num_stages == 32:
            if stage_id == 0:
                self.boundaries = [1, 7, 12]
            elif stage_id == num_stages - 1:
                self.boundaries = [5, 10, 13]
            else:
                self.boundaries = [5, 10]

        #64 layers 16d
        if num_stages == 16:
            if stage_id == 0:
                self.boundaries = [1, 11, 19]
            elif stage_id == num_stages - 1:
                self.boundaries = [9, 17, 25]
            else:
                self.boundaries = [9, 17]

        #64 layers 8d
        if num_stages == 8:
            if stage_id == 0:
                self.boundaries = [1, 19, 36]
            elif stage_id == num_stages - 1:
                self.boundaries = [17, 34, 49]
            else:
                self.boundaries = [17, 34]

        #64 layers 4d
        if num_stages == 4:
            if stage_id == 0:
                self.boundaries = [1, 19, 36, 51, 67, 84]
            elif stage_id == num_stages - 1:
                self.boundaries = [17, 34, 49, 65, 82, 97]
            else:
                self.boundaries = [17, 34, 49, 65, 82]

        self.stages = len(self.boundaries) + 1

        super().__init__(params, defaults)

    def grads_sync(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                #if p.grad.data.is_sparse:
                #    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                #if p.grad.is_sparse:
                #    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                grad = split_stage_grads[indx-offset].view(self.grad_shapes[indx])
                assert grad.shape == p.grad.data.shape

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

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
