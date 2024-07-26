import torch
import torch.optim as optim
import copy
from stuned.utility.utils import (
    add_custom_properties,
    error_or_print,
    raise_unknown,
    get_with_assert
)
from stuned.utility.imports import (
    FROM_CLASS_KEY,
    make_from_class_ctor
)
import pytorch_warmup as warmup


class NanCheckingOptimizer(torch.nn.Module):

    def __init__(self, optimizer, set_to_none=True, logger=None):
        super().__init__()
        self.optimizer = optimizer
        self.set_to_none = set_to_none
        add_custom_properties(self.optimizer, self)
        self.logger = logger

    def step(self):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                assert p.requires_grad
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any():
                    error_or_print("Encountered NaN in grads", self.logger)
                    return
        self.optimizer.step()

    def zero_grad(self, set_to_none=None):
        # currently this method is never used
        # because model.zero_grad() is called in wilds algorithm
        if set_to_none is None:
            set_to_none = self.set_to_none
        self.optimizer.zero_grad(set_to_none=set_to_none)


def make_nan_checking_optimizer(optimizer, logger=None):
    return NanCheckingOptimizer(optimizer, logger)


def make_optimizer(
    optimizer_config,
    param_container,
    wrapping_function=None,
    **make_args
):

    if isinstance(param_container, torch.nn.Module):
        param_container = param_container.parameters()
    else:
        for param in param_container:
            param.requires_grad = True

    trainable_params = filter(lambda p: p.requires_grad, param_container)
    optimizer_type = optimizer_config["type"]
    optimizer_params_config = optimizer_config.get(optimizer_type, {})
    start_lr = optimizer_config["start_lr"]
    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            trainable_params,
            lr=start_lr,
            momentum=optimizer_params_config["momentum"],
            weight_decay=optimizer_params_config["weight_decay"],
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=start_lr,
            **optimizer_params_config
        )
    elif optimizer_type == "adamW":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=start_lr,
            **optimizer_params_config
        )
    else:
        raise_unknown("optimizer", optimizer_type, "optimizer config")

    if wrapping_function is not None:
        optimizer = wrapping_function(optimizer)
    return optimizer


def make_lr_scheduler(scheduler_config, optimizer, **make_args):
    scheduler_type = scheduler_config["type"]

    specific_config = scheduler_config.get(scheduler_type, {})
    if isinstance(optimizer, torch.nn.Module):
        optimizer = optimizer.optimizer
    if scheduler_type == "reduce_on_plateau":
        specific_config = copy.deepcopy(specific_config)
        loss_stat_name = specific_config.pop("loss_stat_name", None)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **specific_config
        )
        lr_scheduler.loss_stat_name = loss_stat_name
    elif scheduler_type.startswith(FROM_CLASS_KEY):
        if (
                specific_config["class"]
            ==
                "torch.optim.lr_scheduler.CosineAnnealingLR"
        ):
            specific_kwargs = get_with_assert(specific_config, "kwargs")
            assert "T_max" in specific_kwargs
            if specific_kwargs["T_max"] is None:
                assert "max_epochs" in make_args, \
                    (
                        "max_epochs should be passed to make_lr_scheduler"
                        " when T_max is None"
                    )
                specific_kwargs["T_max"] = make_args["max_epochs"]
        lr_scheduler = make_from_class_ctor(specific_config, [optimizer])
    else:
        raise_unknown("lr_scheduler", scheduler_type, "scheduler config")

    warmup_config = scheduler_config.get("warmup", {})

    if warmup_config:
        # in case of optimizer wrappers
        if isinstance(optimizer, torch.nn.Module):
            optimizer = optimizer.optimizer
        lr_scheduler.warmup = warmup.LinearWarmup(
            optimizer,
            warmup_period=get_with_assert(warmup_config, "duration")
        )
    else:
        lr_scheduler.warmup = None
    return lr_scheduler
