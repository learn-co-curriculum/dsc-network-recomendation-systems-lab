
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



## Create an Item Matrix

This is essentially the same as the user based matrix you saw constructed in the previous lesson, but for items versus other items. From this, you'll then select the most similar items in order to produce a recommendation suitable for the bottom of a product page.


```python
#Your code here
```

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

Now that you have the preprocessed and transformed the data, it's time to employ collaborative filtering to generate recommendations! Be sure to print the book name that you are generating recommendations for as well as the name of the books being recommended.


```python
#Well, got a couple or extraneous results in there, but perhaps good measure for comparion.
#What does our recommender return for these books?
rec_dict = {}
# id_name_dict = dict(zip(meta.ASIN, meta.Title))
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
    
    
    



```python
Siddhartha = meta[meta.Title.str.contains('Siddhartha')]

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
      <th>17602</th>
      <td>24826</td>
      <td>0822012243</td>
      <td>Steppenwolf and Siddhartha Notes : Including L...</td>
      <td>NaN</td>
      <td>Book</td>
      <td>458237</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21583</th>
      <td>30377</td>
      <td>1569752303</td>
      <td>Before He Was Buddha: The Life of Siddhartha</td>
      <td>general biographi subject religion jack kornfi...</td>
      <td>Book</td>
      <td>603288</td>
      <td>2</td>
      <td>4.5</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>70504</th>
      <td>98334</td>
      <td>0141181230</td>
      <td>Siddhartha: An Indian Tale (Penguin Twentieth-...</td>
      <td>hermann general subject literatur hess german ...</td>
      <td>Book</td>
      <td>144102</td>
      <td>18</td>
      <td>4.5</td>
      <td>4</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>122695</th>
      <td>171003</td>
      <td>0486404374</td>
      <td>Siddhartha: A Dual-Language Book (Dual-Languag...</td>
      <td>hermann general foreign subject literatur hess...</td>
      <td>Book</td>
      <td>335833</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>158936</th>
      <td>221642</td>
      <td>1572700483</td>
      <td>Siddhartha (Mondo Folktales)</td>
      <td>hermann general folklor subject literatur hess...</td>
      <td>Book</td>
      <td>560858</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>165712</th>
      <td>231025</td>
      <td>1567310079</td>
      <td>Siddhartha</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>553190</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>190794</th>
      <td>265968</td>
      <td>0764191241</td>
      <td>Barron's Book Notes Hermann Hesse's Steppenwol...</td>
      <td>educ subject book refer note</td>
      <td>Book</td>
      <td>896189</td>
      <td>5</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>195990</th>
      <td>273198</td>
      <td>0142437182</td>
      <td>Siddhartha: An Indian Tale (Penguin Classics)</td>
      <td>hermann general subject literatur hess german ...</td>
      <td>Book</td>
      <td>368607</td>
      <td>18</td>
      <td>4.5</td>
      <td>4</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>208238</th>
      <td>290107</td>
      <td>1400001293</td>
      <td>Siddhartha (Spanish Edition)</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>809069</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>249846</th>
      <td>346370</td>
      <td>0553208845</td>
      <td>Siddhartha</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>366</td>
      <td>363</td>
      <td>4.5</td>
      <td>39</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>268227</th>
      <td>371400</td>
      <td>0811202925</td>
      <td>Siddhartha</td>
      <td>hermann general subject religion literatur hes...</td>
      <td>Book</td>
      <td>41999</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>302161</th>
      <td>417560</td>
      <td>1570629706</td>
      <td>Siddhartha</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>15848</td>
      <td>16</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>316455</th>
      <td>437021</td>
      <td>1570627215</td>
      <td>Siddhartha : Siddhartha (Shambhala Classics)</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>473700</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>340325</th>
      <td>469427</td>
      <td>0861711211</td>
      <td>Prince Siddhartha Coloring Book</td>
      <td>general subject religion color eastern activ b...</td>
      <td>Book</td>
      <td>500248</td>
      <td>1</td>
      <td>5.0</td>
      <td>5</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>344585</th>
      <td>475277</td>
      <td>1564559149</td>
      <td>Siddhartha: A New Translation</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>282728</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>352185</th>
      <td>485630</td>
      <td>081120068X</td>
      <td>Siddhartha</td>
      <td>hermann general subject religion literatur hes...</td>
      <td>Book</td>
      <td>75920</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>378969</th>
      <td>522397</td>
      <td>0486406539</td>
      <td>Siddhartha (Dover Thrift Editions)</td>
      <td>hermann general subject literatur hess author ...</td>
      <td>Book</td>
      <td>8135</td>
      <td>363</td>
      <td>4.5</td>
      <td>5</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>389139</th>
      <td>536979</td>
      <td>0861710169</td>
      <td>Prince Siddhartha: The Story of Buddha (Wisdom...</td>
      <td>general subject religion eastern book age chil...</td>
      <td>Book</td>
      <td>478216</td>
      <td>1</td>
      <td>5.0</td>
      <td>5</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

Well done! In this lab, you effectively created a recommendation system for a real world dataset!
