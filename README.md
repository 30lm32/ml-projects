
### Introduction
In this repo, you can find my works related to a bunch of different self paced machine learning projects in which I was involved to get dirty my hands in order to turn my real passion into hands-on experiences.

Please, check out once the table to get briefly information about projects, below.

|__Problem__|__Data__|__Methods__|__Libs__|__Desc.__|__Link__|
|-|-|-|-|-|-|
|`NLP`|Text|`Naive Bayesian`, `SVM`, `Random Forest Classifier`, `Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|[Click](#which-one-does-it-catch-whole-spam-sms)|https://github.com/erdiolmezogullari/ml-spam-sms-classification|
|`NLP`|Text|`Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|[Click](#which-novel-do-i-belong-to)|https://github.com/erdiolmezogullari/ml-deep-learning-keras-novel|
|`Imbalanced Data`|Car Booking|`Random Forest Classifier`|`Sklearn`, `Pandas`, `Seaborn`|[Click](#why-do-customers-choose-and-book-specific-vehicles)|https://github.com/erdiolmezogullari/ml-imbalanced-car-booking-data|
|`Forecasting, Timeseries`|Sales|`Random Forest Regressor`|`statsmodels`, `pandas`, `sklearn`, `seaborn`|[Click](#forecasting-impact-of-promos-promo1-promo2-on-sales-in-germany-austria-and-france)|https://github.com/erdiolmezogullari/ml-time-series-analysis-on-sales-data|
|`ML Service`|Randomly Generated|`Random Forest Classifier`|`Flask`, `Docker`, `Redis`, `Sklearn`|[Click](#deploying-machine-learning-model-as-a-service-in-a-docker-container--mlass)|https://github.com/erdiolmezogullari/ml-dockerized-microservice|
|`PySpark`|Randomly Generated|`Random Forest Classifier`|`Spark (PySpark)`, `Sklearn`, `Pandas`, `Seaborn`|[Click](#random-forest-classification-pyspark)| https://github.com/erdiolmezogullari/ml-random-forest-pyspark|
|`Data Enrichment`|Spatial|`Kd-tree`|`cKDTree`|[Click](#spatial-data-enrichment-join-two-geolocation-datasets-by-using-kdtree)|https://github.com/erdiolmezogullari/ml-join-spatial-data|
|`Implementation`|Statistics of Countries|`K-Means`|`Java SDK`|[Click](#implementation-of-k-means-algorithm-from-scratch-in-java)|https://github.com/erdiolmezogullari/ml-k-means|


Please, scroll down to see the comprehensive details about projects or visit their repos.

### Which one does it catch whole* SPAM SMS?
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`NLP`|Text|`Naive Bayesian`, `SVM`, `Random Forest Classifier`, `Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|https://github.com/erdiolmezogullari/ml-spam-sms-classification|

In this project, We applied supervised learning (classification) algorithms and deep learning (LSTM).

We used a public [SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which is not purely clean dataset. The data consists of two different columns (features), such as context, and class. The column context is referring to SMS. The column class may take a value that can be either `spam` or `ham` corresponding to related SMS context.

Before applying any supervised learning methods, we applied a bunch of data cleansing operations to get rid of messy and dirty data since it has some broken and messy context.

After obtaining cleaned dataset, we created tokens and lemmas of SMS corpus seperately by using [Spacy](https://spacy.io/), and then, we generated [bag-of-word](https://en.wikipedia.org/wiki/Bag-of-words_model) and [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) of SMS corpus, respectively. In addition to these data transformations, we also performed [SVD](https://en.wikipedia.org/wiki/Singular-value_decomposition), [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce dimension of dataset.

To manage data transformation in training and testing phase effectively and avoid [data leakage](https://www.kaggle.com/wiki/Leakage), we used Sklearn's [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class. So, we added each data transformation step (e.g. `bag-of-word`, `TF-IDF`, `SVC`) and classifier (e.g. `Naive Bayesian`, `SVM`, `Random Forest Classifier`) into an instance of class `Pipeline`.

After applying those supervised learning methods, we also perfomed deep learning.
Our deep learning architecture we used is based on [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory). To perform LSTM approching in [Keras  (Tensorflow)](https://keras.io/), we needed to create an embedding matrix of our corpus. So, we used [Gensim's Word2Vec](https://radimrehurek.com/gensim/) approach to obtain embedding matrix, rather than TF-IDF.

At the end of each processing by different classifier, we plotted [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) to compare which one the best classifier for filtering SPAM SMS.

### Which novel do I belong To?
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`NLP`|Text|`Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|https://github.com/erdiolmezogullari/ml-deep-learning-keras-novel|

This project is related to text classification problem that we tackled with NLP and sufficient machine learning approaches. Dataset consists of different arbitrary passages which were collected over the different popular novels, below. We need to build a machine learning model that classifies those given arbitrary contexts as belonging to out of the following 12 novels:

    0. alice_in_wonderland
    1. dracula
    2. dubliners
    3. great_expectations
    4. hard_times
    5. huckleberry_finn
    6. les_miserable
    7. moby_dick
    8. oliver_twist
    9. peter_pan
    10. talw_of_two_cities
    11. tom_sawyer

In other words, those novels are our target variables as we listed above. To handle that problem, We need to focus on the semantical meaning of sentences amongst passages. If there is any semantical flow amongst sentences in corresponding passage, We think about similar passage were collected over same novels mostlikely. Therefore, `Deeplearing (LSTM)` is the most suitable apporach with `word2vec`.

`Deeplearing (LSTM)` were used on top of `Keras (Tensorflow)` after creating embedding matrix by `Gensim's word2vec`.

### Why do customers choose and book specific vehicles?
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Imbalanced Data`|Car Booking|`Random Forest Classifier`|`Sklearn`, `Pandas`, `Seaborn`|https://github.com/erdiolmezogullari/ml-imbalanced-car-booking-data|

We built a machine learning model that answers the question, -what is the customer preference- on car booking dataset.

We explored the dataset by using `Seaborn`, and transformed, derived new features necessary.

In addition, the shape of dataset is `imbalanced`. It means that the target variable's distribution is skewed. To overcome that challenge, there are already defined a few different techniques (e.g. `over/under resampling techniques`) and intuitive approaches. We try to solve that problem using resampling techniques, as well.

### Forecasting impact of promos (promo1, promo2) on sales in Germany, Austria, and France
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Forecasting - Timeseries`|Sales|`Random Forest Regressor`|`statsmodels`, `pandas`, `sklearn`, `seaborn`|https://github.com/erdiolmezogullari/ml-time-series-analysis-on-sales-data|

There are stores are giving two type of promos such as radio, TV corresponding to promo1 and promo2 so that they want to increase their sales across Germany, Austria, and France. However, they don't have any idea about which promo is sufficient to do it. So, the impact of promos on their sales are important roles on their preference.

To define well-defined promo strategy, we once need to analysis data in terms of impacts of promos. In that case, since data is based on time series, we once referred to use  `time series decomposition`. After we decomposed `observed` data into `trend`, `seasonal`, and `residual` components, We exposed the impact of promos clearly to make a decision which promo is better in each country.

In addition, we used `Random Forest Regression` in this forecasting problem to boost our decision. 

### Deploying Machine Learning model as a Service in a Docker container : MLasS
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`ML Service`|Randomly Generated|`Random Forest Classifier`|`Flask`, `Docker`, `Redis`, `Sklearn`|https://github.com/erdiolmezogullari/ml-dockerized-microservice|

In this project, a `ML based micro-service` was developed on top of `REST` and `Docker` after building a machine learning model by performing `Random Forest`

We used `docker-compose` to launch the micro services, below.

    1.Jupyter Notebook,
    2.Restful Comm. (Flask),
    3.Redis

After we created three different container, our MLasS would be ready.

### Random Forest Classification (PySpark)
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`PySpark`|Randomly Generated|`Random Forest Classifier`|`Spark (PySpark)`, `Sklearn`, `Pandas`, `Seaborn`| https://github.com/erdiolmezogullari/ml-random-forest-pyspark|

You can find a bunch of sample code related to how you can use PySpark. In this repo, Spark's MLlib (Random Forest Classifier), and Pipeline via PySpark.

### Spatial data enrichment: Join two geolocation datasets by using Kdtree
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Data Enrichment`|Spatial|`Kd-tree`|`cKDTree`|https://github.com/erdiolmezogullari/ml-join-spatial-data|

In this project, to build an efficient script that finds the closest airport to a given user based on their geolocation and the geolocation of the airport.

To make that data enrichment, we used `Kd-tree` algorithm.

### Implementation of K-Means Algorithm from scratch in Java

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Implementation`|Statistics of Countries|`K-Means`|`Java SDK`| https://github.com/erdiolmezogullari/ml-k-means|

Dataset: https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/K-Means#Input_data