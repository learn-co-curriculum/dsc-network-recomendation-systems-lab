
# Recommendation Systems - Lab

## Introduction

Now that you've gotten an introduction to collaborative filtering and recommendation systems, it's time to put your skills to test and attempt to build a recommendation system for a real world dataset! For this exercise, you'll be using a dataset regarding the book reviews on the Amazon marketplace. While the previous lesson focused on user-based recommendation systems, you'll apply a parallel process for an item-based recommendation system to recommend similar books at the bottom of the product page.

## Objectives

You will be able to:
* Implement a recommendation system on a real world dataset

## Load the Dataset


```python
import pandas as pd
df = pd.read_csv('books_data.edgelist', names=['source', 'target', 'weight'], delimiter=' ')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>target</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0827229534</td>
      <td>0804215715</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0827229534</td>
      <td>156101074X</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0827229534</td>
      <td>0687023955</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0827229534</td>
      <td>0687074231</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0827229534</td>
      <td>082721619X</td>
      <td>0.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
import networkx as nx
G = nx.Graph()
```

## Load the MetaData


```python
meta = pd.read_csv('books_meta.txt', sep='\t')
meta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ASIN</th>
      <th>Title</th>
      <th>Categories</th>
      <th>Group</th>
      <th>SalesRank</th>
      <th>TotalReviews</th>
      <th>AvgRating</th>
      <th>DegreeCentrality</th>
      <th>ClusteringCoeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0827229534</td>
      <td>Patterns of Preaching: A Sermon Sampler</td>
      <td>clergi sermon subject religion preach spiritu ...</td>
      <td>Book</td>
      <td>396585</td>
      <td>2</td>
      <td>5.0</td>
      <td>8</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0738700797</td>
      <td>Candlemas: Feast of Flames</td>
      <td>subject witchcraft earth religion spiritu base...</td>
      <td>Book</td>
      <td>168596</td>
      <td>12</td>
      <td>4.5</td>
      <td>9</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0486287785</td>
      <td>World War II Allied Fighter Planes Trading Cards</td>
      <td>general hobbi subject craft home garden book</td>
      <td>Book</td>
      <td>1270652</td>
      <td>1</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0842328327</td>
      <td>Life Application Bible Commentary: 1 and 2 Tim...</td>
      <td>spiritu translat commentari christian book gui...</td>
      <td>Book</td>
      <td>631289</td>
      <td>1</td>
      <td>4.0</td>
      <td>6</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1577943082</td>
      <td>Prayers That Avail Much for Business: Executive</td>
      <td>subject religion spiritu busi christian live w...</td>
      <td>Book</td>
      <td>455160</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## Select Books to Test Your Recommender On

Select a small subset of books that you are interested in generating recommendations for. 


```python
#Lets rexamine our fascination with Game of Thrones...
GOT = meta[meta.Title.str.contains('Thrones')]
GOT
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ASIN</th>
      <th>Title</th>
      <th>Categories</th>
      <th>Group</th>
      <th>SalesRank</th>
      <th>TotalReviews</th>
      <th>AvgRating</th>
      <th>DegreeCentrality</th>
      <th>ClusteringCoeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59750</th>
      <td>83450</td>
      <td>0553103547</td>
      <td>A Game of Thrones (A Song of Ice and Fire, Boo...</td>
      <td>general subject martin author epic z seri fant...</td>
      <td>Book</td>
      <td>16330</td>
      <td>1191</td>
      <td>4.5</td>
      <td>4</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>130560</th>
      <td>182190</td>
      <td>1572701293</td>
      <td>Thrones, Dominations</td>
      <td>literatur seri tape book format fiction genera...</td>
      <td>Book</td>
      <td>395606</td>
      <td>61</td>
      <td>3.5</td>
      <td>4</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>183820</th>
      <td>256164</td>
      <td>0553573403</td>
      <td>A Game of Thrones (A Song of Ice and Fire, Boo...</td>
      <td>general subject martin author epic z seri fant...</td>
      <td>Book</td>
      <td>969</td>
      <td>1196</td>
      <td>4.5</td>
      <td>7</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>261763</th>
      <td>362549</td>
      <td>0553381687</td>
      <td>A Game of Thrones (A Song of Ice and Fire, Boo...</td>
      <td>general subject martin author epic z seri fant...</td>
      <td>Book</td>
      <td>11463</td>
      <td>1196</td>
      <td>4.5</td>
      <td>4</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>331188</th>
      <td>457079</td>
      <td>0312968302</td>
      <td>Thrones, Dominations (A Lord Wimsey Mystery)</td>
      <td>general subject british author sayer dorothi l...</td>
      <td>Book</td>
      <td>68918</td>
      <td>61</td>
      <td>3.5</td>
      <td>7</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>
</div>



## Generate Recommendations for a Few Books of Choice

The 'books_data.edgelist' has conveniently already calculated the distance between items for you. Given this preprocessed and data, it's time to employ collaborative filtering to generate recommendations! Generate the top 10 recommendations for each book in the subset you chose. Be sure to print the book name that you are generating recommendations for as well as the name of the books being recommended.


```python
#Well, got a couple or extraneous results in there, but perhaps good measure for comparion.
#What does our recommender return for these books?
rec_dict = {}
id_name_dict = dict(zip(meta.ASIN, meta.Title))
for row in GOT.index:
    book_id = GOT.ASIN[row]
    book_name = id_name_dict[book_id]
    most_similar = df[(df.source==book_id)
                      | (df.target==book_id)
                     ].sort_values(by='weight', ascending=False).head(10)
    most_similar['source_name'] = most_similar['source'].map(id_name_dict)
    most_similar['target_name'] = most_similar['target'].map(id_name_dict)
    recommendations = []
    for row in most_similar.index:
        if most_similar.source[row] == book_id:
            recommendations.append((most_similar.target_name[row], most_similar.weight[row]))
        else:
            recommendations.append((most_similar.source_name[row], most_similar.weight[row]))
    rec_dict[book_name] = recommendations
    print("Recommendations for:", book_name)
    for r in recommendations:
        print(r)
    print('\n\n')
```

    Recommendations for: A Game of Thrones (A Song of Ice and Fire, Book 1)
    ('A Clash of Kings (A Song of Ice and Fire, Book 2)', 1.0)
    ('A Feast for Crows (A Song of Ice and Fire, Book 4)', 0.92)
    ('A Storm of Swords (A Song of Ice and Fire, Book 3)', 0.85)
    ("Assassin's Apprentice (The Farseer Trilogy, Book 1)", 0.56)
    
    
    
    Recommendations for: Thrones, Dominations
    ('Have His Carcase', 0.59)
    ('The Nine Tailors', 0.58)
    ('Strong Poison', 0.55)
    ("Busman's Honeymoon", 0.55)
    
    
    
    Recommendations for: A Game of Thrones (A Song of Ice and Fire, Book 1)
    ('A Storm of Swords : Book Three of A Song of Ice and Fire (A Song of Ice and Fire, Book 3)', 1.0)
    ('A Storm of Swords (A Song of Ice and Fire, Book 3)', 1.0)
    ('A Clash of Kings (A Song of Ice and Fire, Book 2)', 1.0)
    ('A Feast for Crows (A Song of Ice and Fire, Book 4)', 0.92)
    ('A Storm of Swords (A Song of Ice and Fire, Book 3)', 0.85)
    ("Assassin's Apprentice (The Farseer Trilogy, Book 1)", 0.56)
    ('The Fourth Tower of Inverness', 0.24)
    
    
    
    Recommendations for: A Game of Thrones (A Song of Ice and Fire, Book 1)
    ('A Clash of Kings (A Song of Ice and Fire, Book 2)', 1.0)
    ('A Feast for Crows (A Song of Ice and Fire, Book 4)', 0.92)
    ('A Storm of Swords (A Song of Ice and Fire, Book 3)', 0.85)
    ("Assassin's Apprentice (The Farseer Trilogy, Book 1)", 0.56)
    
    
    
    Recommendations for: Thrones, Dominations (A Lord Wimsey Mystery)
    ('Have His Carcase', 0.93)
    ('Strong Poison', 0.86)
    ("Busman's Honeymoon", 0.86)
    ('A Presumption of Death (Mystery Masters Series)', 0.75)
    ('A Presumption of Death: A New Lord Peter Wimsey/Harriet Vane Mystery', 0.71)
    ('The Nine Tailors', 0.67)
    ('A Presumption of Death (Mystery Masters Series)', 0.63)
    
    
    


## Summary

Well done! In this lab, you effectively created a recommendation system for a real world dataset!
