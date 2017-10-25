import traceback

import pandas as pd
from flask import Flask, jsonify, request
from redis import Redis
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

        for col in columns:
            self.encoders[col] = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if col in X.columns:
                    output[col] = self.encoders[col].fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = self.encoders[col].fit_transform(col)
        return output

    def inverse_transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if col in X.columns:
                    output[col] = self.encoders[col].inverse_transform(X[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = self.encoders[col].inverse_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

app = Flask(__name__)
redis = Redis(host='redis', port=6379)

format_resources = '../../resources/{}'


def encode(X):
    return encoder.fit_transform(X)

@app.route('/predict', methods=['POST'])
def predict():

    if model and encoder and columns:
        try:
            # This request has unsorted features, below.
            json_ = request.json

            # To make it sorted, we are using pre-stored columns on dataframe
            df = pd.DataFrame(columns=columns)
            for k, v in json_.items():
                df[k] = [v]

            # Encoding categorical features
            e = encode(df)

            # Caching on Redis
            if redis.get(e) is None:
                # Making prediction over our pre-trained model
                y_pred = model.predict(e)
                redis.set(e, y_pred)
                cached = False
            else:
                # If we already have the prediction on the redis, we will retrieve over it.
                y_pred = redis.get(e).decode('utf-8')
                cached = True

            # Returning result in json format
            return jsonify({'x': df.to_json(orient='records'),
                            'y_pred': str(y_pred),
                            'cached': cached})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        return 'ML model is not available. Please, train your data on Notebook as a first stage'


@app.route('/predict_random')
def predict_random():

    if model and encoder:
        try:
            # Picking a row randomly to test
            s = df_val.sample(1)

            # Encoding categorical features including classLabel
            e = encode(s)

            x = e.drop(['classLabel'], 1)
            y = e['classLabel']

            # Caching on Redis
            if redis.get(e) is None:
                # Making prediction over our pre-trained model
                y_pred = model.predict(x)
                redis.set(e, y_pred)
                cached = False
            else:
                # If we already have the prediction on Redis, we will retrieve over it.
                y_pred = redis.get(x).decode('utf-8')
                cached = True

            return jsonify({'raw': s.to_json(orient='records'),
                            'x': x.to_json(orient='records'),
                            'y': str(y.values),
                            'y_pred': str(y_pred),
                            'cached': str(cached)})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

    else:
        return 'ML model is not available. Please, train your data on Notebook as a first stage'


if __name__ == "__main__":
    encoder = joblib.load(format_resources.format('encoder.pkl'))
    model = joblib.load(format_resources.format('model.pkl'))
    columns = joblib.load(format_resources.format('columns.pkl'))
    df_val = pd.read_csv(format_resources.format('validation.csv'))

    app.run(host="0.0.0.0", debug=True)


