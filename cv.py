#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# In[ ]:


class NestedTimeSeriesCV:
    def __init__(self, n_splits_outer=5, n_splits_inner=5):
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.outer_cv = TimeSeriesSplit(n_splits=n_splits_outer)
        self.inner_cv = TimeSeriesSplit(n_splits=n_splits_inner)

    def split(self, X, y):
        for train_outer_idx, test_outer_idx in self.outer_cv.split(X):
            train_outer, test_outer = X.iloc[train_outer_idx], X.iloc[test_outer_idx]
            y_train_outer, y_test_outer = y.iloc[train_outer_idx], y.iloc[test_outer_idx]

            for train_inner_idx, val_inner_idx in self.inner_cv.split(train_outer):
                train_inner, val_inner = train_outer.iloc[train_inner_idx], train_outer.iloc[val_inner_idx]
                y_train_inner, y_val_inner = y_train_outer.iloc[train_inner_idx], y_train_outer.iloc[val_inner_idx]

                yield train_inner.index, val_inner.index, test_outer.index

