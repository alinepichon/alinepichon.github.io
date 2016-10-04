---
layout: post
title: "Next.Buy: recommend your user's next purchase"
---


## Insight data science project: recommend your user's next purchase

<center> <img src="{{ site.baseurl }}/images/good-better-best.jpg" alt="alt text" width="600px"> </center>

During my stay at Insight [Insight Data Science](http://insightdatascience.com/), I took a consulting project for a company who provides retailers with the opportunity to communicate with their customers after the sale is made. This way, the retailers can  make personalized offers to their customers. My goal as consultant was to offer some insight on the kind of offers that could be made, by generating a recommendation model predicting the next purchase a given user would be the most likely to make, based on her/his purchase histories, demographic specificities and a given retailer. More specifically, the aim of such a recommendation model is to present a ranked list of objects (e.g. products) given an input object (another product) or a user. I used a two-step strategy to tackle the high sparsity of my data. 

### Next.Buy Demo

<center> <iframe src="//www.slideshare.net/slideshow/embed_code/key/2gJ94ezFc59xfn" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/AlinePichon/aline-pichon-demonoaddslides" title="Aline pichon demo_no_addslides" target="_blank">Aline pichon demo_no_addslides</a> </strong> from <strong><a href="//www.slideshare.net/AlinePichon" target="_blank">Aline Pichon</a></strong> </div> </center>


### Next.Buy: the project

I was given access to the whole MySQL database [MySQL](https://en.wikipedia.org/wiki/MySQL) - located on a remote server - that contained information about more than 260M products, purchased by 9M users, in 6000 stores from 40 retailers across the USA and Canada. 

<center> <img src="{{ site.baseurl }}/images/USA_CANADA_MAP.png" alt="alt text" width="600px"> </center>

The Non-Disclosure Agreement (NDA) associated to this project prohibits me to name retailers and any other features that could identify them, so I will refer to them with different names, such as retailer A, B, C. Each product was characterized by several features, including the following: a description (name, e.g. "banana"), a category (e.g. "fruit"), the quantity bought for each purchase, the location of the purchase (store and store address), and the retailer name.

<center> <img src="{{ site.baseurl }}/images/DATA_ORGANIZATION.png" alt="alt text" width="600px"> </center>

The fact that the server was distant caused the SQL queries to be extremely slow, which in turn meant that I could only pull a limited amount of users and their purchase history. Since the deliverable called for a model that was both demographic and merchant specific, I started by characterizing "who buys where". The graph below (Fig. 1) represents the number of users for the top 10 biggest retailers. The 3 first retailers, A, B and C totalize most of the users (>94%). 

Figure 1.Number of users per retailer.
<center> <img src="{{ site.baseurl }}/images/USERSxRETAILER.png" alt="alt text" width="600px"> </center>

Another interesting data to look at is where the most purchases were made. I did not have access to the information about the user's personal location, so I took the store's location (state) as a proxy. About a third of the purchases for retailer A were made in Florida and Texas (Fig. 2)

Figure 2. Number of purchases made per state for retailer A
<center> <img src="{{ site.baseurl }}/images/PURCHASExSTATE.png" alt="alt text" width="600px"> </center>

I thus decided to focus on these subsets to estimate my model.

### Collaborative filtering

I wanted a model that was optimal for characterizing the large number of users/items in the data set (9M/260M), but that could still be estimated for the limited amount of users (between 5000 and 10000) and products (2000 for merchant C) that I managed to pull.

I chose collaborative filtering [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), which assesses users or item’s similarity to predict user’s preference. For example, if two users 1 and 2 tend to watch and like similar kinds of movies, the likelihood that user 1 will like a movie that user 2 likes too is high. The liking is usually assessed by a rating, for example from 1 (very bad movie) to 5 (very good movie).

A way to visualize the data is to generate a N rows x M columns, where N are users and M items, where item is the product that is being recommended(in Machine learning, N are called 'samples' and M, 'features'). The matrix is populated with user-item pairs, corresponding to an event (e.g. rating when a given user rated a given item, or an empty entry when a given user did not rate the given item, often modelized with a 0). Sparsity corresponds to the proportion of user-item pairs that actually have data. 

In our case two adaptations were made:

1. Since there was no information available on how the customers liked their purchase (no ratings), I used the quantity of items purchased as a proxy for users's preference to populate the user - item matrix, with a 0 entry when no purchase was made (Fig. 3 and 4).

2. the user - product matrix was too sparse (less than 0.1%) to come up with a direct product recommendation. Therefore, I used the category information provided in the dataset as an intermediate step to reduce dimensionality. Again, the NDA prevents me to reveal true information about the products so I replaced real categories with grocery categories and product names with grocery items. Since the number of categories in merchant B were limited, I only estimated the model in merchants A and C for this project.


Figure 3. Number of purchases made per category for retailer C.
<center> <img src="{{ site.baseurl }}/images/COUNT_QTY_CATEG_18_C.png" alt="alt text" width="600px"> </center>

Figure 4. Distribution of purchases made for category 'Fruit', retailer C as a function of the quantity of items bought
<center> <img src="{{ site.baseurl }}/images/PURCHASExSTATE.png" alt="alt text" width="600px"> </center>


## A two-step recommender system 


### train and test phase 

Leave k out:  a split percentage is chosen (e.g., 80% train, 20% test) and the test percentage is selected randomly from the user-item pairs with non-zero entries. 

### Category recommendation

I trained a model per location (state) and retailer (merchant A and C)
The way the model works is the following: Francis, a user from FL, will be recommended with various categories according to his purchase history and his similarity with other users from FL (User-based recommendation).These categories will be ranked from most likely (fruits) to least likely (drinks), as function of similarity scores.  For this example, Francis most recommended category was 'fruits'. 

<center> <img src="{{ site.baseurl }}/images/FRANCIS_CATEGORY.png" alt="alt text" width="600px"> </center>

For this example I chose a User-Based recommendation system that considers similarities between user consumption histories and item similarities. I used cosine similarity [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) as the distance metric to assess the user to user and the item to item similarity. This similarity is measured as the cosine of the angle between the two users' vectors.

### Product recommendation

Once a given category has been recommended, the user will be recommended with products within this category (or the next best category). In this case, Francis was recommended with bananas. 

<center> <img src="{{ site.baseurl }}/images/FRANCIS_ALL_REC.png" alt="alt text" width="600px"> </center>


### Repurchase score & recommendation

In this particular model, to avoid recommending something that was already purchased, items (categories or products) that were already bought by a given were assigned with a similarity score of zero, so thay they do not appear in top of the recommendation. Therefore, there is high a likelihood that the most recommended category for Francis was one he did not actually buy, which means we would not have data for Francis at the product level. In other terms, how could we recommend a given fruit (bananas, apples, strawberries...), if Francis did not buy any fruits at all? 

The solution is to APPROXIMATE Francis's behavior by that of his most similar user within Florida, Felipe, that bought within the 'Fruit' category, and look at Felipe's 1st recommendation at the product level.

<center> <img src="{{ site.baseurl }}/images/FRANCIS_FELIPE.png" alt="alt text" width="600px"> </center>

Nevertheless, the user behavior will vary as function of the categories of articles sold by a retailer. It is sometimes preferable for the model to keep recommending items that were already bought, if for example, we are designing a recommendation algorithm for a grocery store, because the likelihood to purchase the same items again and again is high (e.g. milk, bread). This is known as a high repurchase score. 

On the other hand, other retailers (e.g. furniture store) will sell items that are not bought frequently, i.e. with a low repurchase score (e.g. bed, sofa). In this case, the model could be adapted by suppressing the already bought items from the recommendation.

Finally, in the case of a a retailer that sells both high and low repurchase score items, a possible solution would be to weigh the recommendation of these items according to their repurchase score. 

The repurchase score can be intuitively understood as a combination of how popular an item is (number of times an item was purchased) and whether these items were purchased over several days. It can be estimated by assessing the frequency with which a given item was rated / bought by a given user. This implies having access to a long history of purchases (over a year for example), which was not the case with the subsample studied here. 


### Insights on local shopping behavior

We mentioned earlier that the recommender models described here are specic to each retailer and location. This offers the possibility to compare user behaviors geographically. For instance, if we examine the recommendations made for Ted, Francis's most similar user in TX, we can see that Ted also got recommended with the 'fruit' category', but instead of bananas, he was recommended with an apple. 

<center> <img src="{{ site.baseurl }}/images/FRANCIS_TED_ALL_REC.png" alt="alt text" width="600px"> </center>

With sufficient computational power, this algorithm could be run on all possible users from all locations. By cumulating the comparisons accross states, the model can eventually provide quantitative insight on local shopping behaviour, and thus infer suggestions to retailers on how to adjust their product offer according to location. 


## Validation metrics

Distinguishing between good and bad recommendations is part of what we want to achieve with recommender systems. A binarization threshold determines which data is 'good' (or labeled as 1) or bad (0) when dealing with non-binary data (such as ratings or purchased quantities in our case). For both category and product recommendation, this threshold was set to 1.1 as the distribution of items suggested a vast majority of unique purchases (see Fig. 4). 

Classification metrics (ranging from 0 to 1)
Classification metrics used in other machine learning algorithms (e.g. binary classification) are thus suitable here.
The Receiver Operator Characteristic (ROC) curve, a toll used in classification, plots the True Positive Rate against the False Positive rate and this enables to summarize the performance of the model. It can be summarized by its integral, or Area Under the Curve (AUC-ROC). The recall represents the ability of the classifier to fun all positive samples, while the precision epresents the ability of the classifier not label as positive a sample that is negative.
[Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
[ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)


Ranking metrics (ranging from 0 to 1)
Since we are producing a ranking of the best possible items to recommend for each user. A useful rank-based metric is the Normalized Discounted Cumulative Gain (NDCG) which emphasizes that items with high relevance should be placed early in the ranked list, without binarizing the data.
[NDCG](https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain)

Figure 5 shows the validation metrics for boths steps of the recommender system
On the left category choice, both classification metrics and ranking metrics suggest a good prediction and accurate ranking of relevant items. On the right, for the product choice, metrics are generally good with the exception of precision. This is possibly due to the small sample of users (5000) vs large amount products (400), resulting in important sparsity. The resuts could be optimized if the model was to be run on more users. 

Figure 5
<center> <img src="{{ site.baseurl }}/images/validation_metrics.png" alt="alt text" width="120px"> </center>


## Improvements


- Use Pearson coefficients / euclidean distances in lieu of cosine similarity
- Different clustering of products than by category, using Principal Components Analysis for example
- Weighing the recommendations by the repurchase score (or a probability for a given user to buy a given category / product again).


See also for more on recommender systems
[Data Piques, Intro to Recommender Systems](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/)
[Collaborative filter with Python](http://www.salemmarafi.com)
