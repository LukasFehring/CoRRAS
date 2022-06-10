import numpy as np
from copy import deepcopy
from sklearn.preprocessing import minmax_scale
from scipy.stats import kendalltau, rankdata


def ndcg_metric(gt_scores: np.ndarray, predicted_scores: np.ndarray):
	predicted_scores = np.array(deepcopy(predicted_scores))
	gt_scores = np.array(minmax_scale(gt_scores, feature_range=(0, 1)))

	predicted_ranking = rankdata(predicted_scores)
	gt_ranking = rankdata(gt_scores)

	predicted_scores_sum = np.sum((gt_scores - 1) / np.log2(1 + predicted_ranking))
	return predicted_scores_sum / np.sum((gt_scores - 1) / np.log2(1 + gt_ranking))
