# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import sys
sys.path.append("../..")
from torch.optim.optimizer import required
from optimizer_with_stashing import OptimizerWithStashing

class SGDWithStashing(OptimizerWithStashing):
    """
    SGD optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False, verbose_freq=0,
                 macrobatch=False):
        super(SGDWithStashing, self).__init__(
            optim_name='SGD',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay,
            nesterov=nesterov, verbose_freq=verbose_freq,
            macrobatch=macrobatch,
        )

def test(num_versions, assertion_ground_truth):
    # N is batch size; D_in is input dimension;
    # D_out is output dimension.
    N, D_in, D_H, D_out = 4, 4, 4, 4

    # Create random input and output tensors.
    x1 = torch.randn(N, D_in).cuda().requires_grad_(True)
    y1 = torch.randn(N, D_out).cuda()

    # Create copies of these tensors.
    x2 = x1.clone().detach().requires_grad_(True)
    y2 = y1.clone()

    x3 = x1.clone().detach().requires_grad_(True)
    y3 = y1.clone()

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_H),
        torch.nn.ReLU(),
        torch.nn.Linear(D_H, D_out)
    ).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = SGDWithStashing([model], model.parameters(),
                                model_parameters=None,
                                loss_scale=1.0,
                                num_versions=num_versions,
                                lr=1e-1)

    inputs = [x1, x2, x3]

    # Compute the prediction and loss function using the same weights
    # and inputs.
    y1_pred = model(x1)
    y2_pred = model(x2)
    y3_pred = model(x3)
    assert torch.equal(y1_pred, y2_pred)
    assert torch.equal(y1_pred, y3_pred)

    losses = [loss_fn(y1_pred, y1), loss_fn(y2_pred, y2),
              loss_fn(y3_pred, y3)]
    x_grads = []

    for loss, x in zip(losses, inputs):
        optimizer.zero_grad()
        optimizer.load_backward_params()
        loss.backward()
        x_grads.append(x.grad.clone().detach())
        optimizer.load_forward_params()
        optimizer.step()

    # Assert that the right weight versions are used to compute the
    # gradients.
    assert (torch.equal(x_grads[0], x_grads[1]) ==
            assertion_ground_truth[0])
    assert (torch.equal(x_grads[0], x_grads[2]) ==
            assertion_ground_truth[1])

    assert not torch.equal(y1_pred, model(x1))


if __name__ == '__main__':
    test(1, [False, False])
    test(2, [True, False])
    test(3, [True, True])
