# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.optim
import time

from collections import deque  # Efficient ring buffer implementation.

class Version:
    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version+1)

class OptimizerWithStashingAndAggregation(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - update_interval: the total number of steps that need to be aggregated
                           together. In practice, for the number of versions to be 2,
                           this needs to be the number of workers in the stage.
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, modules, master_parameters,
                 num_stages, update_interval, verbose_freq=0,
                 base_optimizer_cls=None, **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters

        self.num_versions = 2
        self.num_stages = num_stages
        self.update_interval = update_interval
        if self.num_stages == 1:
            self.num_versions = 0
            self.update_interval = 1

        if base_optimizer_cls is not None:
            self.base_optimizer = base_optimizer_cls(
                master_parameters, **optimizer_args)
        else:
            self.base_optimizer = getattr(torch.optim, optim_name)(
                master_parameters, **optimizer_args)
        self.latest_version = Version()
        self.current_version = Version()

        self.forward_counter = 0
        self.backward_counter = 0

        self.verbose_freq = verbose_freq

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        self.queue = deque(maxlen=self.num_versions)
        for i in range(self.num_versions):
            self.queue.append(self.get_params(clone=True))
        if len(self.queue) > 0:
            self.buffered_state_dicts = self.queue[0][0]

    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # Don't update running_mean and running_var; these should
                # accumulate normally.
                # mask might have a different shape, so don't copy it to
                # the module this way.
                if "running_" in key or "mask" in key:
                    state_dict[key] = cur_state_dict[key]
            #module.load_state_dict(state_dict)
            for key in state_dict:
                module.state_dict()[key].data.copy_(state_dict[key].data)

            # Load the mask.
            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    # NOTE: Do we need to clone here?
                    attribute = state_dict[key]
        self.current_version = version

    def load_forward_params(self):
        if self.update_interval == 1: return
        # Compute the desired version; load version at the tail of the queue if
        # it matches the desired version.
        desired_version = self.forward_counter // self.update_interval
        desired_version = max(desired_version-1, 0)
        for (state_dicts, version) in self.queue:
            if desired_version == version.version:
                if self.current_version.version != version.version:
                    self.set_params(state_dicts, version)
                break
        if self.verbose_freq > 0:
            print(self.forward_counter, "load_forward_params", self.current_version,
                  self.latest_version)
        self.forward_counter += 1

    def load_backward_params(self):
        if self.update_interval == 1: return
        # Compute the desired version; load version at the head of the queue if
        # it matches the desired version.
        desired_version = self.backward_counter // self.update_interval
        desired_version = max(desired_version-1, 0)
        for (state_dicts, version) in self.queue:
            if desired_version == version.version:
                if self.current_version.version != version.version:
                    self.set_params(state_dicts, version)
                break
        if self.verbose_freq > 0:
            print(self.backward_counter, "load_backward_params", self.current_version,
                  self.latest_version)
        self.backward_counter += 1

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _load_step_params(self):
        if self.update_interval == 1: return
        (state_dicts, version) = self.queue[-1]
        self.set_params(state_dicts, version)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # Update the gradient every `update_interval` steps.
        if self.verbose_freq > 0:
            print("step", self.current_version, self.latest_version)

        # Load latest parameters before applying latest gradient.
        self._load_step_params()

        # TODO: Put back fp16 and timing logic as needed.
        loss = self.base_optimizer.step()
        self.latest_version = self.latest_version.incr()
        if self.update_interval > 1:
            self.buffered_state_dicts = self.queue[0][0]
            self.queue.append(self.get_params(clone=False))

        return loss
