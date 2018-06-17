

![Image](https://cdn-images-1.medium.com/max/1600/1*60gs-SFYyooZZBxatuoNJw.jpeg)

### Introduction
In this repository, you can find my works related to a bunch of different self paced machine learning projects. Since I have a real passion for ML, I got my hands dirty to get hands-on experiences.

Please, check out once the table to get briefly information about projects, below.

|__Problem__|__Data__|__Methods__|__Libs__|__Desc.__|__Link__|
|-|-|-|-|-|-|
|`NLP`|Text|`LightGBM`, `TF-IDF` |`Sklearn`, `LightGBM`, `Pandas`, `Seaborn`|[Click](#multi-language-and-multi-label-classification-problem-on-fashion-dataset)|https://github.com/erdiolmezogullari/multi-label-classification|
|`NLP`|Text|`Naive Bayesian`, `SVM`, `Random Forest Classifier`, `Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|[Click](#which-one-does-it-catch-whole-spam-sms)|https://github.com/erdiolmezogullari/ml-spam-sms-classification|
|`NLP`|Text|`Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|[Click](#which-novel-do-i-belong-to)|https://github.com/erdiolmezogullari/ml-deep-learning-keras-novel|
|`Imbalanced Data`|Car Booking|`Random Forest Classifier`|`Sklearn`, `Pandas`, `Seaborn`|[Click](#why-do-customers-choose-and-book-specific-vehicles)|https://github.com/erdiolmezogullari/ml-imbalanced-car-booking-data|
|`Forecasting, Timeseries`|Sales|`Random Forest Regressor`|`statsmodels`, `pandas`, `sklearn`, `seaborn`|[Click](#forecasting-impact-of-promos-promo1-promo2-on-sales-in-germany-austria-and-france)|https://github.com/erdiolmezogullari/ml-time-series-analysis-on-sales-data|
|`ML Service`|Randomly Generated|`Random Forest Classifier`|`Flask`, `Docker`, `Redis`, `Sklearn`|[Click](#deploying-machine-learning-model-as-a-service-in-a-docker-container--mlass)|https://github.com/erdiolmezogullari/ml-dockerized-microservice|
|`PySpark`|Randomly Generated|`Random Forest Classifier`|`Spark (PySpark)`, `Sklearn`, `Pandas`, `Seaborn`|[Click](#random-forest-classification-pyspark)| https://github.com/erdiolmezogullari/ml-random-forest-pyspark|
|`Data Enrichment`|Spatial|`Kd-tree`|`cKDTree`|[Click](#spatial-data-enrichment-join-two-geolocation-datasets-by-using-kdtree)|https://github.com/erdiolmezogullari/ml-join-spatial-data|
|`Implementation`|Statistics of Countries|`K-Means`|`Java SDK`|[Click](#implementation-of-k-means-algorithm-from-scratch-in-java)|https://github.com/erdiolmezogullari/ml-k-means|
|`Forecasting, Timeseries`|AWS EC2 Spot Price|`Adaboost Classifier`, `Decision Tree`|`Rapidminer`|[Click](#forecasting-aws-spot-price-by-using-adaboosting-on-rapidminer)|https://github.com/erdiolmezogullari/ml-forecasting-aws-spot-price|


Please, scroll down to see the comprehensive details about projects or visit their repository.

### Multi-language and Multi-label Classification Problem on Fashion Dataset

Dataset was collected over different fashion web sites. It consists of 7 fields like below.

* `id`: A unique product identifier
* `name`: The title of the product, as displayed on our website
* `description`: The description of the product
* `price`: The price of the product
* `shop`: The shop from which you can buy this product
* `brand`: The product brand
* `labels`: The category labels that apply to this product

The text features (name, description) are in different languages, such as English, German and Russian. The format of target feature is multilabels (60 categories) that were tagged according to corresponding to the category in fashion web sites differently.

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

This project is related to text classification problem that we tackled with `Deeplearing (LSTM)` model, which classifies given arbitrary paragraphes collected over 12 different novels randomly, above: 

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

In other words, you can think about those novels are our target classes of our dataset.
To distinguish actual class of paragraph, the semantic latent amongst paragraphes would play an important role. Therefore, We used `Deeplearing (LSTM)` on top of `Keras (Tensorflow)` after creating an embedding matrix by `Gensim's word2vec`.

If there is any semantic latent amongst sentences in corresponding paragraph, 
We think about similar paragraphes were collected from same resources (novels) most likely.

### Why do customers choose and book specific vehicles?
---

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Imbalanced Data`|Car Booking|`Random Forest Classifier`|`Sklearn`, `Pandas`, `Seaborn`|https://github.com/erdiolmezogullari/ml-imbalanced-car-booking-data|

We built a machine learning model that answers the question, -what is the customer preference- on car booking dataset.

We explored the dataset by using `Seaborn`, and transformed, derived new features necessary.

In addition, the shape of dataset is `imbalanced`. It means that the target variable's distribution is skewed. To overcome that challenge, there are already defined a few different techniques (e.g. `over/under re-sampling techniques`) and intuitive approaches. We try to solve that problem using resampling techniques, as well.

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

In this repository, you can find a bunch of sample code related to how you can use PySpark Spark's MLlib (Random Forest Classifier), and Pipeline via PySpark.

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

### Forecasting AWS Spot Price by using Adaboosting on Rapidminer

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`Forecasting, Timeseries Analysis`|AWS EC2 Spot Price|`Adaboost Classifier`, `Decision Tree`|`Rapidminer`|https://github.com/erdiolmezogullari/ml-forecasting-aws-spot-price|


In our project, we will use public data, which was collected by third party people and released through some specific websites. Since our data will be mainly related to Amazon Web Services’ (AWS) Elastic Computing (EC2), it will be consisting of some different fields. EC2 is a kind of virtual machine in the AWS’s cloud.
A virtual machine can be created just in time either on private or public cloud over AWS whenever you need it. A new virtual machine can be picked with respect to different specs and configurations in terms of CPU, RAM, storage, and network band limit before creating it once from scratch. EC2 machines also are separated and managed by AWS on different geographical regions (US East, US West, EU, Asia Pacific, South America) and zone to increase availability of virtual machines across the world. AWS has different segmentations, which were classified with respect to system specs by AWS for based on different goals (macro instance, general purpose, compute optimized, storage optimized, GPU instance, memory optimized). Payment options are dedicated, on­demand and spot instance. Since they make different cost to customer’s operation, customers may prefer different kinds of virtual machine according to their goals and budgets. In general, spot instance is cheaper than the rest of the options. However, spot instance may be interrupted if market price exceeds our max bid.
In our research, we will focus on spot instance payment. Our aim in this project will be selecting correct AWS instance from the Spot Instance Market according to the requirement of the customer. We plan to perform Decision Tree on streaming data to make a decision on the fly. It may be implemented as an incremental version of decision tree since data is changing continuously
