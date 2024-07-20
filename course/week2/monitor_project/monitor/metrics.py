import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def get_ks_score(tr_probs, te_probs):
  # Convert tensors to numpy arrays
  tr_probs_np = tr_probs.numpy()
  te_probs_np = te_probs.numpy()
  
  # Apply the Kolmogorov-Smirnov test
  _, p_value = ks_2samp(tr_probs_np, te_probs_np)
  
  # Return the p-value as the score
  return p_value


def get_hist_score(tr_probs, te_probs, bins=10):
  # Convert tensors to numpy arrays
  tr_probs_np = tr_probs.numpy()
  te_probs_np = te_probs.numpy()
  
  # Compute histograms
  tr_heights, bin_edges = np.histogram(tr_probs_np, bins=bins, density=True)
  te_heights, _ = np.histogram(te_probs_np, bins=bin_edges, density=True)
  
  # Compute the histogram intersection score
  score = 0.0
  for i in range(len(bin_edges) - 1):
      bin_diff = bin_edges[i+1] - bin_edges[i]
      tr_area = bin_diff * tr_heights[i]
      te_area = bin_diff * te_heights[i]
      intersect = min(tr_area, te_area)
      score += intersect
  
  return score


def get_vocab_outlier(tr_vocab, te_vocab):
  num_total = len(te_vocab)
  
  if num_total == 0:
    return 0
  
  num_seen = sum(1 for word in te_vocab if word in tr_vocab)
  
  # Compute the outlier score
  score = 1 - (num_seen / num_total)
  
  return score

class MonitoringSystem:

  def __init__(self, tr_vocab, tr_probs, tr_labels):
    self.tr_vocab = tr_vocab
    self.tr_probs = tr_probs
    self.tr_labels = tr_labels

  def calibrate(self, tr_probs, tr_labels, te_probs):
    # Convert tensors to numpy arrays
    tr_probs_np = tr_probs.numpy()
    tr_labels_np = tr_labels.numpy()
    te_probs_np = te_probs.numpy()
    
    # Fit the isotonic regression model
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(tr_probs_np, tr_labels_np)
    
    # Calibrate probabilities
    tr_probs_cal = torch.tensor(iso_reg.transform(tr_probs_np))
    te_probs_cal = torch.tensor(iso_reg.transform(te_probs_np))
    
    return tr_probs_cal, te_probs_cal

  def monitor(self, te_vocab, te_probs):
    tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)

    # Compute metrics
    ks_score = get_ks_score(tr_probs, te_probs)
    hist_score = get_hist_score(tr_probs, te_probs)
    outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)

    metrics = {
        'ks_score': ks_score,
        'hist_score': hist_score,
        'outlier_score': outlier_score,
    }
    return metrics