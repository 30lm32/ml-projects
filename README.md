
Updated: 03-16-18

In this repo, you will find a bunch of different projects' links based on machine learning.

Please, scroll down to see the detail of projects and github repo's links.

### Which one does it catch whole* SPAM SMS?
-------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-spam-sms-classification

Methods: `Naive Bayesian`, `SVM`, `Random Forest Classifier`, `Deep Learning - LSTM`

In this project, We applied supervised learning (classification) algorithms and deep learning (LSTM).

We used a public [SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which is not purely clean dataset. The data consists of two different columns (features), such as context, and class. The column context is referring to SMS. The column class may take a value that can be either `spam` or `ham` corresponding to related SMS context.

Before applying any supervised learning methods, we applied a bunch of data cleansing operations to get rid of messy and dirty data since it has some broken and messy context.

After obtaining cleaned dataset, we created tokens and lemmas of SMS corpus seperately by using [Spacy](https://spacy.io/), and then, we generated [bag-of-word](https://en.wikipedia.org/wiki/Bag-of-words_model) and [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) of SMS corpus, respectively. In addition to these data transformations, we also performed `SVD` / `PCA` / `SVC` to reduce dimension of dataset.

To manage data transformation in training and testing phase effectively and avoid [data leakage](https://www.kaggle.com/wiki/Leakage), we used Sklearn's [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class. So, we added each data transformation step (e.g. `bag-of-word`, `TF-IDF`, `SVC`) and classifier (e.g. `Naive Bayesian`, `SVM`, `Random Forest Classifier`) into an instance of class `Pipeline`.

After applying those supervised learning methods, we also perfomed deep learning.
Our deep learning architecture we used is based on [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory). To perform LSTM approching in [Keras  (Tensorflow)](https://keras.io/), we needed to create an embedding matrix of our corpus. So, we used [Gensim's Word2Vec](https://radimrehurek.com/gensim/) approach to obtain embedding matrix, rather than TF-IDF.

At the end of each processing by different classifier, we plotted [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) to compare which one the best classifier for filtering SPAM SMS.

### Which novel do I belong To?
-------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-deep-learning-keras-novel

Methods: `Deep Learning - LSTM`

In this project, you are expected to learn a Machine Learning model that classifies a given line as belonging to one of the following 12 novels:

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

`Deeplearing (LSTM)` on top of `Keras (Tensorflow)` is performing the novel corpus data to solve this problem 
after creating `word2vec` by using `Gensim`.

### 02-imbalanced_car_booking_data [will be updated]
--------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-imbalanced-car-booking-data

In this project, we tried to solve `imbalanced data` using `over/under resampling techniques`

### 03-time_series_analyis_on_sales_data [will be updated]
--------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-time-series-analysis-on-sales-data

In this project, we applied `time series decomposition techniques` and `random forest algorithm` to build a ML model

### 04-ml_model_docker_web_service [will be updated]
--------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-dockerized-microservice

In this project, a `ML micro-service` was developed by using `REST` and `Docker` after building a ML model using `random forest algoritm`

### 05-join_data_by_geolocation [will be updated]
--------------------------------
Check out its repo on Github: https://github.com/erdiolmezogullari/ml-join-spatial-data

In this project, two different data set which have location based `(GPS)` feature were joined `Kd-tree`.
