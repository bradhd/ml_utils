"""
Some utility functions for balancing classes in DataFrames.
The typical use case is that you have some DataFrame with a categorical target variable, and you want to either undersample or 
oversample the rows of the DataFrame so as to balance the classes for the purpose of training a classifier.
"""

__author__ = 'Brad Hannigan-Daley'

import pandas as pd
import numpy as np


def oversample(df, label_name):
	"""
	Assume that df has a column called label_name which is a categorical variable.
	Perform random oversampling of the rows of df to balance the classes defined by label_name and return the resulting DataFrame.

	:param df: DataFrame to oversample
	:param label_name: name of the column of df containing the class label
	"""
	v_counts = df[label_name].value_counts() # sorted in descending order by default
	max_count = max(v_counts)

	df_parts = [] # create one oversampled df for each label name, to concatenate at the end
	for v in v_counts.index:
		v_index = df[df[label_name]==v].index
		df_parts.append(df.ix[np.random.choice(v_index, max_count - v_counts[v])])
	return pd.concat([df] + df_parts)


def undersample(df, label_name):
	"""
	Assume that df has a column called label_name which is a categorical variable.
	Perform random undersampling of the rows of df to balance the classes defined by label_name and return the resulting DataFrame.

	:param df: DataFrame to undersample
	:param label_name: name of the column of df containing the class label
	"""

	v_counts = df[label_name].value_counts() # sorted in descending order by default
	min_count = min(v_counts)

	df_parts = [] # create one undersampled df for each label name, to concatenate at the end
	for v in v_counts.index:
		v_index = df[df[label_name]==v].index
		df_parts.append(df.ix[np.random.choice(v_index, min_count, replace=False)])
	return pd.concat(df_parts)
	