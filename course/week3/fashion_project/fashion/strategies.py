import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = np.random.choice(len(pred_probs), size=budget, replace=False).tolist()
  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  uncertainty = 1 - pred_probs.max(dim=1)[0]  # Least confident = highest uncertainty
  indices = uncertainty.argsort(descending=True)[:budget].tolist()
  return indices

def margin_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  top_two_probs = torch.topk(pred_probs, 2, dim=1)[0]
  margin = top_two_probs[:, 0] - top_two_probs[:, 1]
  indices = margin.argsort()[:budget].tolist()
  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  epsilon = 1e-6
  entropy = -torch.sum(pred_probs * torch.log(pred_probs + epsilon), dim=1)
  indices = entropy.argsort(descending=True)[:budget].tolist()
  return indices