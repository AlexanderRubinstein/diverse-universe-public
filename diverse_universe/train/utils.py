import torch
# import sys
# import os
# import torch.optim as optim
# import copy
# import pytorch_warmup as warmup


# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from utility.imports import (
#     FROM_CLASS_KEY,
#     make_from_class_ctor
# )
# from utility.utils import (
#     raise_unknown,
#     get_with_assert,
#     aggregate_tensors_by_func,
#     func_for_dim
# )
# sys.path.pop(0)


BASE_ESTIMATOR_LOG_SUFFIX = "base_estimator"


# def make_optimizer(
#     optimizer_config,
#     param_container,
#     wrapping_function=None,
#     **make_args
# ):

#     if isinstance(param_container, torch.nn.Module):
#         param_container = param_container.parameters()
#     else:
#         for param in param_container:
#             param.requires_grad = True

#     trainable_params = filter(lambda p: p.requires_grad, param_container)
#     optimizer_type = optimizer_config["type"]
#     optimizer_params_config = optimizer_config.get(optimizer_type, {})
#     start_lr = optimizer_config["start_lr"]
#     if optimizer_type == "sgd":
#         optimizer = optim.SGD(
#             trainable_params,
#             lr=start_lr,
#             momentum=optimizer_params_config["momentum"],
#             weight_decay=optimizer_params_config["weight_decay"],
#         )
#     elif optimizer_type == "adam":
#         optimizer = optim.Adam(
#             trainable_params,
#             lr=start_lr,
#             **optimizer_params_config
#         )
#     elif optimizer_type == "adamW":
#         optimizer = torch.optim.AdamW(
#             trainable_params,
#             lr=start_lr,
#             **optimizer_params_config
#         )
#     else:
#         raise_unknown("optimizer", optimizer_type, "optimizer config")

#     if wrapping_function is not None:
#         optimizer = wrapping_function(optimizer)
#     return optimizer


# def make_lr_scheduler(scheduler_config, optimizer, **make_args):
#     scheduler_type = scheduler_config["type"]

#     specific_config = scheduler_config.get(scheduler_type, {})
#     if isinstance(optimizer, torch.nn.Module):
#         optimizer = optimizer.optimizer
#     if scheduler_type == "reduce_on_plateau":
#         specific_config = copy.deepcopy(specific_config)
#         loss_stat_name = specific_config.pop("loss_stat_name", None)
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             **specific_config
#         )
#         lr_scheduler.loss_stat_name = loss_stat_name
#     elif scheduler_type.startswith(FROM_CLASS_KEY):
#         if (
#                 specific_config["class"]
#             ==
#                 "torch.optim.lr_scheduler.CosineAnnealingLR"
#         ):
#             specific_kwargs = get_with_assert(specific_config, "kwargs")
#             assert "T_max" in specific_kwargs
#             if specific_kwargs["T_max"] is None:
#                 assert "max_epochs" in make_args, \
#                     (
#                         "max_epochs should be passed to make_lr_scheduler"
#                         " when T_max is None"
#                     )
#                 specific_kwargs["T_max"] = make_args["max_epochs"]
#         lr_scheduler = make_from_class_ctor(specific_config, [optimizer])
#     else:
#         raise_unknown("lr_scheduler", scheduler_type, "scheduler config")

#     warmup_config = scheduler_config.get("warmup", {})

#     if warmup_config:
#         # in case of optimizer wrappers
#         if isinstance(optimizer, torch.nn.Module):
#             optimizer = optimizer.optimizer
#         lr_scheduler.warmup = warmup.LinearWarmup(
#             optimizer,
#             warmup_period=get_with_assert(warmup_config, "duration")
#         )
#     else:
#         lr_scheduler.warmup = None
#     return lr_scheduler


def take_from_2d_tensor(tensor, indices, dim=-1):
    # dim = dimension along which indices are applied
    # i.e. when dim = -1: res = tensor(tensor[0, indices[0]], ..., tensor[i, indices[i]])

    assert len(tensor.size()) == 2

    dim_for_aranged = abs(dim) - 1

    aranged = torch.arange(tensor.size(dim_for_aranged))

    if dim == -1:
        return tensor[
            aranged,
            indices
        ]
    else:
        assert dim == 0
        return tensor[
            indices,
            aranged
        ]


# def compute_ensemble_output(
#     outputs,
#     weights=None,
#     process_logits=None
# ):

#     if process_logits is None:
#         process_logits = lambda x: x

#     if weights is None:
#         weights = [1.0] * len(outputs)

#     if stores_input(outputs):
#         extractor = lambda x: x[1]
#     else:
#         extractor = lambda x: x

#     return aggregate_tensors_by_func(
#         [
#             weight * process_logits(extractor(submodel_output).unsqueeze(0))
#                 for weight, submodel_output
#                     in zip(weights, outputs)
#         ],
#         func=func_for_dim(torch.mean, dim=0)
#     ).squeeze(0)


# def stores_input(outputs):
#     assert len(outputs) > 0
#     output_0 = outputs[0]
#     return isinstance(output_0, (list, tuple)) and len(output_0) == 2


# def bootstrap_ensemble_outputs(outputs, assert_len=True):
#     if_stores_input = stores_input(outputs)
#     if assert_len:
#         assert if_stores_input
#     if if_stores_input:
#         return [output[1] for output in outputs]
#     else:
#         return outputs


# # TODO(Alex | 25.07.2024): re-balance if conditions
# def record_diversity(
#     res,
#     outputs,
#     stacked_outputs,
#     metrics_mappings,
#     labels=None,
#     name_prefix="",
#     detailed_results=None
# ):

#     # metrics_mappings is a tuple of tuples:
#     # ((name_1, func_1), ... (name_k, func_k))
#     for metric_tuple in metrics_mappings:

#         metric_name = metric_tuple[0]
#         metric_key = name_prefix + metric_name
#         compute_metric = metric_tuple[1]
#         if metric_name not in res:
#             res[metric_key] = 0
#         if metric_name in PER_SAMPLE_METRIC_NAMES:
#             value = compute_metric(stacked_outputs)
#         elif metric_name == "div_ortega":
#             assert labels is not None
#             value = compute_metric(stacked_outputs, labels).item()
#         elif metric_name in [
#             "var",
#             "std",
#             "dis",
#             "max_var",
#             "div_different_preds",
#             "div_mean_logits",
#             "div_max_logit",
#             "div_entropy",
#             "div_max_prob",
#             "div_mean_prob",
#             "div_different_preds_per_model",
#             "div_continous_unique"
#         ]:
#             value = compute_metric(stacked_outputs).item()
#         else:
#             value = aggregate_tensors_by_func(
#                 apply_pairwise(outputs, compute_metric)
#             ).item()

#         if not torch.is_tensor(value):
#             res[metric_key] += value

#         if detailed_results is not None:
#             if metric_key in PER_SAMPLE_METRIC_NAMES:
#                 if metric_key not in detailed_results:
#                     detailed_results[metric_key] = []

#                 if metric_key in detailed_results:
#                     for subvalue in value:
#                         detailed_results[metric_key].append(subvalue.item())


def make_base_estimator_name(base_estimator_id):
    return "{} {}".format(
        BASE_ESTIMATOR_LOG_SUFFIX,
        base_estimator_id
    )
