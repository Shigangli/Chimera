# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import sys
sys.path.append("../..")
from torch.optim.optimizer import required
from optimizer_with_stashing_and_aggregation import OptimizerWithStashingAndAggregation

class SGDWithStashingAndAggregation(OptimizerWithStashingAndAggregation):
    """
    SGD optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, update_interval,
                 verbose_freq=0, lr=required, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        super(SGDWithStashingAndAggregation, self).__init__(
            optim_name='SGD',
            modules=modules,
            master_parameters=master_parameters,
            update_interval=update_interval,
            verbose_freq=verbose_freq,
            lr=lr, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay,
            nesterov=nesterov,
        )

def test(num_warmup_minibatches):
    print()
    print("Starting test with num_warmup_minibatches: %d" % num_warmup_minibatches)

    # D_in is input dimension;
    # D_out is output dimension.
    D_in, D_H, D_out = 4, 4, 4

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_H),
        torch.nn.ReLU(),
        torch.nn.Linear(D_H, D_out)
    ).cuda()
    optimizer = SGDWithStashingAndAggregation(
        modules=[model],
        master_parameters=model.parameters(),
        update_interval=4,
        verbose_freq=1,
        lr=1e-1)

    for i in range(num_warmup_minibatches):
        optimizer.load_forward_params()

    for i in range(20 - num_warmup_minibatches):
        optimizer.load_forward_params()
        optimizer.load_backward_params()
        optimizer.step()
        optimizer.zero_grad()

    for i in range(num_warmup_minibatches):
        optimizer.load_backward_params()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    test(0)
    test(1)
    test(2)
    test(3)
