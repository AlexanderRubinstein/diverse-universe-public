import torch
# import sys
# import os
# import numpy as np
# import re
from stuned.utility.utils import (
    NAME_SEP,
    raise_unknown
    # get_project_root_path,
    # log_or_print,
    # get_device,
    # update_dict_by_nested_key,
    # get_with_assert,
)


# # local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from utility.utils import (
#     NAME_SEP,
#     raise_unknown,
#     is_number,
#     parse_float_or_int_from_string
# )
# from local_models.diverse_vit import (
#     is_diverse_vit_output
# )
# sys.path.pop(0)


ALT_SEP = "-"
assert ALT_SEP != NAME_SEP
# MEAN_ORACLE_KEY = "mean_oracle"


class MetricsWrapper:

    def __init__(
        self,
        compute_metrics,
        metrics_base_name,
        # final_aggregation=None,
        aggregatable_stages=None,
        report_best_metric=False
    ):
        self.compute_metrics = compute_metrics
        # self.final_aggregation = final_aggregation
        # self.aggregatable_stages = aggregatable_stages
        # self.report_best_metric = report_best_metric

        # if self.aggregatable_stages is not None:
        #     assert self.final_aggregation is not None, \
        #         f"final_aggregation must be specified " \
        #         f"if aggregatable_stages is specified in metrics config"

        self.metrics_base_name = metrics_base_name

        # if self.final_aggregation is not None:
        #     self.aggregate_metrics, self.regex_for_keys_to_aggregate \
        #         = make_aggregation(final_aggregation, self.metrics_base_name)
        # else:
        #     assert not self.report_best_metric, "report_best_metric is not " \
        #         "supported without final_aggregation in metrics config"
        #     self.aggregate_metrics = None
        #     self.regex_for_keys_to_aggregate = None

        # assert not self.report_best_metric, "report_best_metric is not " \
        #         "supported without final_aggregation in metrics config"
        # self.aggregate_metrics = None
        # self.regex_for_keys_to_aggregate = None

    def aggregate(self, all_stats):
        assert self.aggregate_metrics is not None
        return self.aggregate_metrics(
            all_stats,
            self.regex_for_keys_to_aggregate
        )

    def __call__(self, output, target):
        # if is_diverse_vit_output(output):
        #     output = output[0]
        if isinstance(output, list):
            assert output
            assert len(output[0]) == 2
            return [
                self.compute_metrics(single_output[1], target)
                    for single_output
                        in output
            ]
        elif torch.is_tensor(output):
            return self.compute_metrics(output, target)
        else:
            raise_unknown("output type", type(output), "metrics __call__()")


def make_metric(config, device="cpu"):

    metric_config = config["metric"]
    metric_type = metric_config["type"]
    specific_metric_config = metric_config.get(metric_type, {})
    if metric_type == "accuracy":
        compute_metrics = AccuracyTopK(1)
    elif metric_type == "accuracy_top_k":
        compute_metrics = AccuracyTopK(specific_metric_config["top_k"])
    else:
        raise_unknown("metric", metric_type, "metric config")

    compute_metrics = MetricsWrapper(
        compute_metrics.to(device),
        metric_type.replace(NAME_SEP, ALT_SEP),
        # final_aggregation=metric_config.get("final_aggregation"),
        aggregatable_stages=metric_config.get("aggregatable_stages"),
        report_best_metric=metric_config.get("report_best_metric", False)
    )
    return compute_metrics


# def make_aggregation(final_aggregation, metrics_base_name):
#     if final_aggregation == MEAN_ORACLE_KEY:
#         regex_for_oracle = re.compile(fr"{metrics_base_name}.*max")
#         return aggregate_mean_oracle, regex_for_oracle
#     else:
#         raise_unknown(
#             "final_aggregation",
#             final_aggregation,
#             "make_aggregation"
#         )


# def aggregate_mean_oracle(all_stats, regex_for_keys_to_aggregate):
#     oracle_preds = []
#     for stage_stats in all_stats.values():
#         for stat_name, stat_value in stage_stats.items():
#             if regex_for_keys_to_aggregate.match(stat_name):

#                 assert isinstance(stat_value, str)
#                 assert is_number(stat_value)
#                 oracle_preds.append(parse_float_or_int_from_string(stat_value))

#     assert len(oracle_preds) > 0, f"Mean oracle aggregation is requested " \
#         f"but there is no stats matching " \
#         f"pattern: {regex_for_keys_to_aggregate.pattern}"
#     oracle_preds = np.array(oracle_preds)
#     return {
#         MEAN_ORACLE_KEY: oracle_preds.mean(),
#         "max_oracle": oracle_preds.max()
#     }


class AccuracyTopK(torch.nn.Module):

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.accuracy_impl = accuracy_topk

    def __call__(self, output, target):
        if output.shape[1] == 1:
            output = torch.cat([1 - output, output], dim=1)
        list_topk_accs = self.accuracy_impl(output, target, topk=(self.topk,))
        return list_topk_accs[0]


# taken from: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662283#gistcomment-3662283
def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]
