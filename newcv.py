#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn.model_selection import KFold
import pandas as pd
from types import GeneratorType

class NestedCV:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        outer_cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_idx, test_idx in outer_cv.split(X):
            yield train_idx, test_idx

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv(r"F:\Assessment IQGateway\New folder\Processed data\newtrain_df.csv")

    # Nested cv
    k = 3
    cv = NestedCV(k)
    splits = cv.split(data)

    # Check return type
    assert isinstance(splits, GeneratorType)

    # Check return types, shapes, and data leaks
    count = 0
    for train_idx, validate_idx in splits:
        train = data.iloc[train_idx]
        validate = data.iloc[validate_idx]

        # Convert train and validate to DataFrames
        train = pd.DataFrame(train)
        validate = pd.DataFrame(validate)

        # Types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(validate, pd.DataFrame)

        # Shape
        assert train.shape[1] == validate.shape[1]

        count += 1

    # Check number of splits returned
    assert count == k


# In[ ]:


get_ipython().system('jupyter nbconvert --to script newcv.ipynb')


# In[ ]:




