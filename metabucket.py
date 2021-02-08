import numpy as np
import tensorflow as tf

from configurable import Configurable
from bucket import Bucket

#***************************************************************
class Metabucket(Configurable):
  """"""

  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""

    name = kwargs.get('name', 'Sents')
    self._n_bkts = kwargs.pop('n_bkts', None)
    super(Metabucket, self).__init__(*args, **kwargs)
    if self._n_bkts is None:
      self._n_bkts = super(Metabucket, self).n_bkts
    self._buckets = [Bucket(self._config, name='%s-%d' % (name, i)) for i in xrange(self.n_bkts)]
    self._sizes = None
    self._data = None
    self._len2bkt = None
    return

  #=============================================================
  def reset(self, sizes, pad=False):
    """"""

    if pad:
      self._data = [(0,0)]
    else:
      self._data = []
    self._sizes = sizes
    self._len2bkt = {}
    prev_size = -1
    for bkt_idx, size in enumerate(sizes):
      self._buckets[bkt_idx].reset(size, pad=pad)
      self._len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx]*(size-prev_size)))
      prev_size=size
    return

  #=============================================================
  def add(self, sent):
    """"""

    if isinstance(self._data, np.ndarray):
      raise TypeError("The buckets have already been finalized, you can't add more to them")

    bkt_idx = self._len2bkt[len(sent)]
    idx = self._buckets[bkt_idx].add(sent)
    self._data.append( (bkt_idx, idx) )
    return len(self._data)-1

  #=============================================================
  def _finalize(self):
    """"""

    for bucket in self:
      bucket._finalize()

    self._data = np.array(self._data)
    return

  #=============================================================
  @property
  def n_bkts(self):
    return self._n_bkts
  @property
  def data(self):
    return self._data
  @property
  def size(self):
    return self.data.shape[0]

  #=============================================================
  def __iter__(self):
    return (bucket for bucket in self._buckets)
  def __getitem__(self, key):
    return self._buckets[key]
  def __len__(self):
    return len(self._buckets)
