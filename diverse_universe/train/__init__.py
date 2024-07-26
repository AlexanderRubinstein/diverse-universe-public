# from .losses import (
#     kl_divergence,
#     div_var,
#     div_std,
#     max_prob_var,
#     dis
# )

# OOM_SAFE_METRICS = tuple([
#     tuple(["kl", kl_divergence]),
#     # tuple("js", js_divergence),
#     # tuple("cos", cosine_similarity),
#     # tuple("orth", orthogonality_loss),
#     # tuple("euc", euclidean_distance),
#     # tuple("jac", jaccard_similarity),
#     # tuple("wass", wasserstein_distance),
#     # tuple("spear", spearman_rank),
#     # tuple("rev", reverse_cross_entropy),
#     # tuple("sink", sinkhorn_distance),
#     # tuple(["dis", dis]),
#     tuple(["var", div_var]),
#     tuple(["std", div_std]),
#     tuple(["max_var", max_prob_var])
# ])
# METRICS = tuple(
#     [
#         tuple(["dis", dis])
#     ] + list(OOM_SAFE_METRICS)
# )
