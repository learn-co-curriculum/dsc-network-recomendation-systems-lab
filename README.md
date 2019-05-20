
# Recommendation Systems - Lab

## Introduction

Now that you've gotten an introduction to collaborative filtering and recommendation systems, it's time to put your skills to test and attempt to build a recommendation system for a real world dataset! For this exercise, you'll be using a dataset regarding the book reviews on the Amazon marketplace. While the previous lesson focused on user-based recommendation systems, you'll apply a parallel process for an item-based recommendation system to recommend similar books at the bottom of the product page.

## Objectives

You will be able to:
* Implement a recommendation system on a real world dataset

## Load the Dataset

To start, load the dataset stored in the file `'books_data.edgelist'`.


```python
#Your code here
```

## Load the MetaData

Next, load the metadata associated with each of the books being reviewed. The metadata is stored in the file `'books_meta.txt'`.


```python
#Your code here
```

## Create an Item Matrix

This is essentially the same as the user based matrix you saw constructed in the previous lesson, but for items versus other items. From this, you'll then select the most similar items in order to produce a recommendation suitable for the bottom of a product page.


```python
#Your code here
```

## Select Books to Test Your Recommender On

Select a small subset of books that you are interested in generating recommendations for. 


```python
#Your code here
```

## Generate Recommendations for a Few Books of Choice

Now that you have the preprocessed and transformed the data, it's time to employ collaborative filtering to generate recommendations! Be sure to print the book name that you are generating recommendations for as well as the name of the books being recommended.


```python
#Your code here
```

## Summary

Well done! In this lab, you effectively created a recommendation system for a real world dataset!
