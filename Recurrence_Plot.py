import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

def recur_opt(em,eps):
  D = euclidean_distances(em,em)

  cinds = np.where(D>eps)
  c = 1 / np.amax(D[cinds])
  x_0 = np.amin(D[cinds])
  a = -0.99/np.log(c*x_0)

  r = np.where(D<=eps,1.0,-a*np.log(c*D))
  return r

def embed(data,dim,tau=1):
  m = len(data) - (dim)*tau
  return np.array([data[i:i + (dim - 1)*tau + 1: tau] for i in range(m)])

def get_recur_opt(sig):
  d = 12
  em = embed(sig,d)
  D = euclidean_distances(em,em)
  ul = np.mean(D)
  idx = np.where(D!=0.0)
  ll = np.amin(D[idx])

  epses = np.linspace(ll,ul+0.001,10)
  recurs = []
  ens = []

  for i in range(epses.shape[0]):
    eps = epses[i]
    rec = np.round(recur_opt(em,eps),2)
    recurs.append(rec)

    rand,prob = r_histograms(rec)
    en = cal_entropy(prob)
    ens.append(en)

  idx_max = np.where(ens==np.amax(ens))[0][0]
  r1 = recurs[idx_max]

  return r1

def cal_entropy(probs):
  en = -np.sum(probs*np.log(probs))
  sh = probs.shape[0]
  l = round(math.log(sh),2)
  return round(en/l,3)

def r_histograms(r):
  r_idxs = np.where(r!=1)
  r = r[r_idxs].flatten()
  rands,freqs = np.unique(r, return_counts=True)
  N = r.shape[0]
  probs = freqs/N
  return rands,probs

def get_opt_flow(sig):
  d = 12
  em = embed(sig, d)
  D = euclidean_distances(em, em)
  ul = np.mean(D)
  idx = np.where(D != 0.0)
  ll = np.amin(D[idx])

  epses = np.linspace(ll, ul + 0.001, 10)
  recurs = []
  ens = []

  for i in range(epses.shape[0]):
    eps = epses[i]
    rec = np.round(recur_opt(em, eps), 2)
    recurs.append(rec)

    rand, prob = r_histograms(rec)
    en = cal_entropy(prob)
    ens.append(en)
  return epses,np.array(ens)