
# Amazon Recommendation System - Lab

## Introduction

Now that you've gotten an introduction to collaborative filtering and recommendation systems, it's time to put your skills to test and build a recommendation system for a real world dataset! For this lab, you'll be using a dataset regarding the book reviews on the Amazon marketplace. While the previous lesson focused on user-based recommendation systems, you'll apply a parallel process for an item-based recommendation system to recommend similar books at the bottom of the product page.

## Objectives

In this lab you will: 

- Use graph-based similarity metrics to create a collaborative filtering recommender system

## Load the Dataset


```python
import pandas as pd
import networkx as nx
G = nx.Graph()

df = pd.read_csv('books_data.edgelist', names=['source', 'target', 'weight'], delimiter=' ')
df.head()
```

## Load the Metadata 

Import the metadata available in the file `'books_meta.txt'` (note it is `'\t'` seperated). 


```python
# Your code here
```

## Select Books to Test Your Recommender On

Select a small subset of books that you are interested in generating recommendations for. 


```python
# Your code here
```

## Generate Recommendations for a Few Books of Choice

The `'books_data.edgelist'` has conveniently already calculated the distance between items for you. Given this preprocessed data, it's time to employ collaborative filtering to generate recommendations! Generate the top 10 recommendations for each book in the subset you chose. Be sure to print the book name that you are generating recommendations for as well as the name of the books being recommended. 


```python
# Your code here
```

## Summary

Well done! In this lab, you effectively created a recommendation system for a real world dataset!
