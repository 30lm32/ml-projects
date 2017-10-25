import os.path
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import gensim, logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from functools import reduce

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer


default_label_names = {0: "alice_in_wonderland", 1: "dracula", 2: "dubliners", 3: "great_expectations",
                       4: "hard_times", 5: "huckleberry_finn", 6: "les_miserable", 7: "moby_dick",
                       8: "oliver_twist", 9: "peter_pan", 10: "tale_of_two_cities", 11: "tom_sawyer"}


class Novel:

    # Spacy library is loading English dictionary.
    __nlp = spacy.load("en")
    __symbol_list = None

    # To implement a processing chain global variables were used.
    # Since I have a limited time I preferred approach rather than using pipeline libs in Pyhton
    # The other reason is that I would like to used them in visualization on Jupyter Notebook
    __clean_tokens = None
    __clean_corpus = None
    __word2vec = None
    __tokenizer = None
    __train_tokens_vector = None
    __test_tokens_vector = None
    __x_train_ps = None  # pad_sequences
    __x_test_ps = None  # pad_sequences
    __embedding_matrix = None
    __nn = None
    labelled_pas = None

    # To create word2vec, the parameters are used in Gensim
    # There are a few different sources instead of creating our word2vec model,
    # + Google's pre-trained vectors based on GoogleNews
    # + GLoVe's pre-trained vectors based on Wikipages
    # + Spacy pre-trained vectors
    __embedding_dim = 300
    __window = 10
    __workers = 4
    __cbow_mean = 1
    __alpha = 0.05

    # Creating an embedding matrix using by word2vec model and the parameters, below
    __embedding_vector_length = 300
    __max_nb_words = 200000
    __max_input_length = 50

    # Deep Learning Layers' parameters are using to build a deep network. Our network consists of the layers, below:
    # + Embedding Layer
    # + Dense Layer
    # + LSTM for RNN Layer
    # + Dense Layer
    __num_lstm = 100
    __num_dense = 300
    __rate_drop_out = 0.2
    __rate_drop_lstm = float(0.15 + np.random.rand() * 0.25)
    __rate_drop_dense = float(0.15 + np.random.rand() * 0.25)

    # Creating file names for models with respect to given parameters, above.
    __format_word2vec_model = "emb_dim:{}_window:{}_cbow:{}_apha:{}.bin"
    __format_dl_model = 'lstm_%d_%d_%.2f_%.2f.h5'

    __word2vec_file = __format_word2vec_model.format(__embedding_dim, __window, __cbow_mean, __alpha)
    __model_dl_file = __format_dl_model % (__num_lstm, __num_dense, __rate_drop_lstm, __rate_drop_dense)

    # In training step, those parameters are using, below.
    __epochs = 100
    __batch_size = 2048
    __validation_split = 0.1
    __shuffle = True

    def __init__(self,
                 x_train_file,
                 y_train_file,
                 x_test_file,
                 x_name="raw_passage",
                 y_name="novel_id",
                 label_names = default_label_names):

        self.__label_names = label_names
        self.__x_name = x_name
        self.__y_name = y_name
        self.__num_classes = len(self.__label_names)
        self.__df_raw_xtrain = pd.read_fwf(x_train_file, header=None, names=[x_name], encoding="utf-8")
        self.__df_raw_ytrain = pd.read_fwf(y_train_file, header=None, names=[y_name], encoding="utf-8")
        self.__df_raw_xtest = pd.read_fwf(x_test_file, header=None, names=[x_name], encoding="utf-8")
        self.__raw_corpus = self.__df_raw_xtrain[x_name].values.tolist() + self.__df_raw_xtest[x_name].values.tolist()

        # https://keras.io/utils/#to_categorical
        self.__y_train_one_hot = to_categorical(self.__df_raw_ytrain[y_name].values, self.__num_classes)

    def head(self):
        print(self.__df_raw_xtrain.head())
        print(self.__df_raw_ytrain.head())
        print(self.__df_raw_xtest.head())

    def describe(self):
        print(self.__label_names)
        print(self.__df_raw_xtrain.describe())
        print(self.__df_raw_ytrain.describe())
        print(self.__df_raw_xtest.describe())
        print("y_train_one_hot: {}".format(self.__y_train_one_hot.shape))

    # Finding symbols and punctuations in the corpus, which consists of train and test data
    @staticmethod
    def __find_syms_puncs_in_corpus(corpus):
        symbol_sets = list(
                map(lambda pas: set(filter(lambda ch: (not ch.isalnum() and not ch.isspace()), pas)), corpus))
        return reduce((lambda lhs_sym, rhs_sym: lhs_sym.union(rhs_sym)), symbol_sets)

    # Converting all characters to lower case
    @staticmethod
    def __lower_case(corpus):
        return list(map(lambda pas: pas.lower(), corpus))

    # Adding extra space around of the symbols and punctuations in the corpus
    @staticmethod
    def __add_space_syms_puncs(corpus, symbols):
        for sym in symbols:  # string.punctuation:
            corpus = list(map(lambda pas: pas.replace(sym, " {} ".format(sym)) if sym in pas else pas, corpus))
        return corpus

    # Removing consecutive whitespaces in the corpus
    @staticmethod
    def __remove_consecutive_spaces(corpus):
        return list(map(lambda pas: " ".join(pas.split()), corpus))

    # Removing symbols and punctuations in the corpus
    def __clean_punc(self, corpus):
        processed_corpus = []
        for pas in corpus:
            processed_pas = self.__nlp(pas)
            tmp_pas = ["" if token.is_stop or token.is_punct else token.orth_ for i, token in enumerate(processed_pas)]
            tmp_pas = list(filter(lambda tok: tok is not "", tmp_pas))
            processed_corpus.append(tmp_pas)

        return processed_corpus

    def __clean_raw_corpus(self, corpus):

        print("The raw corpus is being cleaned...")

        tmp_corpus = self.__lower_case(corpus)
        tmp_corpus = self.__add_space_syms_puncs(tmp_corpus, self.__symbol_list)
        tmp_corpus = self.__remove_consecutive_spaces(tmp_corpus)

        t = self.__clean_punc(tmp_corpus)
        c = list(map(lambda pas: " ".join(pas), t))

        print("The raw corpus was cleaned!")

        return t, c

    def __create_clean_token_corpus(self):

        if self.__clean_tokens is not None and self.__clean_corpus is not None:
            print("clean_token and clean_corpus already have been calculated and loaded...")
            return

        if self.__symbol_list is None:
            self.__symbol_list = self.__find_syms_puncs_in_corpus(self.__raw_corpus)

        print("symbol_list : {}".format(self.__symbol_list))

        # tmp_corpus = self.__lower_case(self.__raw_corpus)
        # tmp_corpus = self.__add_space_syms_puncs(tmp_corpus, self.__symbol_list)
        # tmp_corpus = self.__remove_consecutive_spaces(tmp_corpus)
        #
        # self.__clean_tokens = self.__clean_punc(tmp_corpus)
        # self.__clean_corpus = list(map(lambda p: " ".join(p), self.__clean_tokens))

        self.__clean_tokens, self.__clean_corpus = self.__clean_raw_corpus(self.__raw_corpus)

    def __create_word2vec(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        if not os.path.exists(self.__word2vec_file) or self.__word2vec is None:
            print("{} doesn't exist. A new word2vec is being built...".format(self.__word2vec_file))
            self.__word2vec = gensim.models.Word2Vec(self.__clean_tokens,
                                                 size=self.__embedding_dim,
                                                 window=self.__window,
                                                 workers=self.__workers,
                                                 cbow_mean=self.__cbow_mean,
                                                 alpha=self.__alpha)

            self.__word2vec.save(self.__word2vec_file)

        elif self.__word2vec is not None:
            print("{} has already loaded for word2vec...".format(self.__word2vec_file))
        else:
            print("{} is loading for word2vec...".format(self.__word2vec_file))
            self.__word2vec = gensim.models.Word2Vec.load(self.__word2vec_file)

    def __create_token_vectors(self):

        if self.__train_tokens_vector is not None and self.__test_tokens_vector is not None:
            print("train_token_vectors already have been calculated and loaded...")
            return

        x_train_corpus = self.__clean_corpus[:len(self.__df_raw_xtrain)]
        x_test_corpus = self.__clean_corpus[len(self.__df_raw_xtrain):]

        print("x_train_corpus: {}".format(len(x_train_corpus)))
        print("x_test_corpus: {}".format(len(x_test_corpus)))

        # https://keras.io/preprocessing/text/#tokenizer
        self.__tokenizer = Tokenizer(num_words=self.__max_nb_words)
        self.__tokenizer.fit_on_texts(self.__clean_corpus)

        print('Found %s unique tokens' % len(self.__tokenizer.word_index))

        # https://keras.io/preprocessing/text/#text_to_word_sequence
        self.__train_tokens_vector = self.__tokenizer.texts_to_sequences(x_train_corpus)
        self.__test_tokens_vector = self.__tokenizer.texts_to_sequences(x_test_corpus)

        print("train_tokens_vector: {}".format(len(self.__train_tokens_vector)))
        print("test_tokens_vector: {}".format(len(self.__test_tokens_vector)))

    def __create_sequences(self):

        if self.__x_train_ps is not None and self.__x_test_ps is not None:
            print("pad sequences already have been calculated and loaded...")
            return
        # https://keras.io/preprocessing/sequence/#pad_sequences
        self.__x_train_ps = pad_sequences(self.__train_tokens_vector, maxlen=self.__max_input_length)
        self.__x_test_ps = pad_sequences(self.__test_tokens_vector, maxlen=self.__max_input_length)
        print("x_train_padded: {}".format(self.__x_train_ps.shape))
        print("x_test_padded: {}".format(self.__x_test_ps.shape))

    def __create_embedding_matrix(self):

        if self.__embedding_matrix is not None:
            print("embedding matrix already has been calculated and loaded...")
            return

        token_index = self.__tokenizer.word_index

        number_words = min(self.__max_nb_words, len(token_index)) + 1
        self.__embedding_matrix = np.zeros((number_words, self.__embedding_vector_length))
        for word, i in token_index.items():
            if word in self.__word2vec.wv.vocab:
                self.__embedding_matrix[i] = self.__word2vec.wv.word_vec(word)

        print('Null word embeddings: %d' % np.sum(np.sum(self.__embedding_matrix, axis=1) == 0))
        print("embedding_matrix: {}".format(self.__embedding_matrix.shape))

    def __create_nn(self):

        if self.__nn is not None:
            print("Deep Learning Layers already has been built and loaded...")
            return

        def init_weights(shape, dtype=None):
            print("init_weights shape: {}".format(shape))
            # assert  shape == embedding_matrix.shape
            return self.__embedding_matrix

        self.__nn = Sequential()

        # https://keras.io/layers/embeddings/
        number_words = self.__embedding_matrix[0]
        self.__nn.add(Embedding(number_words,
                            self.__embedding_vector_length,
                            input_length=self.__max_input_length,
                            mask_zero=True,
                            embeddings_initializer=init_weights))

        # https://keras.io/layers/core/#dense
        # https://keras.io/layers/core/#activation
        self.__nn.add(Dense(self.__num_dense, activation='sigmoid'))

        self.__nn.add(Dropout(self.__rate_drop_out))

        # https://keras.io/layers/recurrent/
        self.__nn.add(LSTM(self.__num_lstm, dropout=self.__rate_drop_lstm, recurrent_dropout=self.__rate_drop_lstm))
        self.__nn.add(Dense(self.__num_classes, activation='softmax'))

        # https://keras.io/metrics/
        self.__nn.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        self.__nn.summary()

    def __build_model(self):

        self.__create_clean_token_corpus()
        self.__create_word2vec()
        self.__create_token_vectors()
        self.__create_sequences()
        self.__create_embedding_matrix()
        self.__create_nn()

    def __create_callbacks(self, tensorboard):

        callbacks = []
        # https://keras.io/callbacks/#usage-of-callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        print(self.__model_dl_file)
        # https://keras.io/callbacks/#modelcheckpoint
        model_checkpoint = ModelCheckpoint(self.__model_dl_file, save_best_only=True, save_weights_only=True)

        # https://keras.io/callbacks/#tensorboard
        if tensorboard:
            tensor_board = TensorBoard(log_dir='./logs',
                                       histogram_freq=5,
                                       write_graph=True,
                                       write_images=True,
                                       embeddings_freq=0,
                                       embeddings_layer_names=None,
                                       embeddings_metadata=None)
            callbacks.append(tensor_board)

        callbacks.append(early_stopping)
        callbacks.append(model_checkpoint)

        return callbacks

    def train(self, tensorboard_enable=False):

        self.__build_model()

        callbacks = self.__create_callbacks(tensorboard_enable)

        # https://keras.io/models/model/
        hist = self.__nn.fit(self.__x_train_ps,
                             self.__y_train_one_hot,
                             epochs=self.__epochs,
                             batch_size=self.__batch_size,
                             validation_split=self.__validation_split,
                             shuffle=self.__shuffle,
                             callbacks=callbacks)

        self.__nn.load_weights(self.__model_dl_file)

        bst_val_score = min(hist.history['val_loss'])
        bst_val_score

    def test(self, y_test_file="ytest.txt"):
        prediction_probs = self.__nn.predict(self.__x_test_ps,
                                             batch_size=self.__batch_size,
                                             verbose=1)

        pre_label_ids = list(map(lambda probs: probs.argmax(), list(prediction_probs)))

        tmp_df = pd.DataFrame(data=pre_label_ids)
        tmp_df.to_csv(y_test_file, index=False, header=False)
        print("The result was saved into file 'ytest.txt'")

        prediction_labels = list(map(lambda probs: [probs.argmax(), probs.max(), self.__label_names[probs.argmax()]], list(prediction_probs)))
        self.labelled_pas = list(zip(self.__df_raw_xtest[self.__x_name].values.tolist(), prediction_labels))



    def test_by_file(self, x_test_file, y_test_file="ytest.txt"):

        df = pd.read_fwf(x_test_file, header=None, names=[self.__x_name], encoding="utf-8")
        corpus = df[self.__x_name].values.tolist()

        clean_tokens, clean_corpus = self.__clean_raw_corpus(corpus)
        test_tokens_vector = self.__tokenizer.texts_to_sequences(clean_corpus)
        x_test_ps = pad_sequences(test_tokens_vector, maxlen=self.__max_input_length)

        prediction_probs = self.__nn.predict(x_test_ps,
                                             batch_size=self.__batch_size,
                                             verbose=1)

        pre_label_ids = list(map(lambda probs: probs.argmax(), list(prediction_probs)))

        tmp_df = pd.DataFrame(data=pre_label_ids)
        tmp_df.to_csv(y_test_file, index=False, header=False)
        print("The result was saved into file 'ytest.txt'")

        prediction_labels = list(map(lambda probs: [probs.argmax(), probs.max(), self.__label_names[probs.argmax()]],
                                     list(prediction_probs)))
        self.labelled_pas = list(zip(self.__df_raw_xtest[self.__x_name].values.tolist(), prediction_labels))

    def count_words_in_passage(self):
        return list(map(lambda pas: len(pas), self.__clean_tokens))

    def plot_cdf_nb_tokens_in_passage(self):
        p_lens = self.count_words_in_passage()
        x, y = NovelUtil.calculate_cdf(p_lens)

        figure, axes = plt.subplots(1, figsize=(15, 10))
        NovelUtil.plot_cdf(x,
                 y,
                 axes,
                 deltax=5,
                 xlim=[0, np.mean(p_lens) + 3 * np.std(p_lens) + 50],
                 deltay=0.05,
                 ylim=[0, 1.00])

    def plot_sample_word2vec(self):
        figure, axes = plt.subplots(1, figsize=(15, 10))
        wv = [self.__word2vec[k] for k in list(self.__word2vec.wv.vocab.keys())]
        vocabulary = [k for k in list(self.__word2vec.wv.vocab.keys())]

        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(wv[100:150])
        # Y = tsne.fit_transform(wv[idx_list])

        plt.scatter(Y[:, 0], Y[:, 1])
        for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.axes = axes
        plt.show()


class NovelUtil:
    @staticmethod
    def calculate_cdf(x):

        length = len(x)
        y = np.arange(1, length + 1) / float(length)
        return np.array(sorted(x)), y

    @staticmethod
    def plot_cdf(x,
                 y,
                 ax,
                 deltax=None,
                 xlog=False,
                 xlim=[0, 1],
                 deltay=0.25, ylog=False,
                 ylim=[0, 1],
                 font_size=12):

        if deltax is not None:
            x_ticks = np.arange(xlim[0], xlim[1] + deltax, deltax)
            ax.set_xticks(x_ticks)

        ax.set_xlim(xlim[0], xlim[1])

        #  https://matplotlib.org/examples/color/named_colors.html
        ax.vlines(np.min(x), min(y), max(y), color='navy', label='min', linewidth=4)
        ax.vlines(np.mean(x), min(y), max(y), color='red', label='mean', linewidth=4)
        ax.vlines(np.median(x), min(y), max(y), color='orange', label='median', linewidth=4)
        ax.vlines(np.max(x), min(y), max(y), color='yellow', label='max', linewidth=4)

        m_m_2std = np.mean(x) - 2 * np.std(x)
        m_m_3std = np.mean(x) - 3 * np.std(x)

        m_p_2std = np.mean(x) + 2 * np.std(x)
        m_p_3std = np.mean(x) + 3 * np.std(x)

        print("min: ", np.min(x))
        print("mean: ", np.mean(x))
        print("median: ", np.median(x))
        print("max: ", np.max(x))
        print("mean - 2 * std: ", m_m_2std)
        print("mean - 3 * std: ", m_m_3std)
        print("mean + 2 * std: ", m_p_2std)
        print("mean + 3 * std: ", m_p_3std)

        if m_m_2std > min(x):
            ax.vlines(m_m_2std, min(y), max(y), color='magenta', label='mean - 2 * std', linewidth=4)

        if m_m_2std > min(x):
            ax.vlines(m_m_3std, min(y), max(y), color='cyan', label='mean - 3 * std', linewidth=4)

        if m_p_2std < max(x):
            ax.vlines(m_p_2std, min(y), max(y), color='blue', label='mean + 2 * std', linewidth=4)

        if m_p_3std < max(x):
            ax.vlines(m_p_3std, min(y), max(y), color='green', label='mean + 3 * std', linewidth=4)

        y_ticks = np.arange(ylim[0], ylim[1] + deltay, deltay)

        ax.set_yticks(y_ticks)
        ax.set_ylim(ylim[0], ylim[1])

        if xlog is True:
            ax.set_xscale('log')

        if ylog is True:
            ax.set_yscale('log')

        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.9)

        ax.legend(loc=4)

        sns.set_style('whitegrid')
        sns.set(font_scale=2)
        sns.regplot(x=x, y=y, fit_reg=False, scatter=True, ax=ax)