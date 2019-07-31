"""
"""

from typing import List, Callable, NamedTuple
import argparse
import json
import pickle
import logging
import os
import random
from string import Template
import sys
import hashlib
from pathlib import Path
import numpy as np

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from lime.lime_text import LimeTextExplainer

# from allennlp.models.archival import load_archive
# from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
# from allennlp.predictors import Predictor
# from allennlp.common import Params
# from allennlp.data import Vocabulary
# from allennlp.data.dataset_readers import DatasetReader

import torch
from flask import Flask, request, Response, jsonify, render_template, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

logging.basicConfig(level=logging.INFO)

import nltk
nltk.download('wordnet')
stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [stemmer.stem(self.wnl.lemmatize(t)) for t in word_tokenize(articles) if t not in stopWords]

"""
Implements sklearn interface to NBSVM classifier.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC


def log_normalize_count_vector(arr):
    """
    Takes count vector and normalizes by L1 norm, then takes log.
    """
    return np.log(arr / np.linalg.norm(arr, 1))


class NBSVM(BaseEstimator, LinearClassifierMixin):
    """
    A NBSVM classifier following the sklearn API.
    Parameters
    ----------
    alpha : float, default=1.
        Smoothing parameter for count vectors.
    beta : float, default=0.25
        Interpolation parameter between NB and SVM.
    C : float, default=1.
        Penalty parameter of the L2 error term for SVM.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, alpha=1., beta=0.25, C=1.):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def fit(self, X, y):
        """
        Fit the NBSVM to a dataset.
        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_samples, n_features)
            The training input samples. Must be a sparse matrix containing
            no negative entries.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(
            X,
            y,
            accept_sparse='csr'
         )

        # Validate all X are non-negative
        if (X.data < 0.).any():
            raise ValueError('All X entries should be non-negative.')

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Iterate over single class predictions
        coefficients, intercepts = zip(*[
            self._fit_single_class(X, y == class_)
            for class_ in self.classes_
        ])

        self.coef_ = np.concatenate(coefficients)
        self.intercept_ = np.concatenate(intercepts)

        # Return the classifier
        return self

    def _fit_single_class(self, X, y):
        """
        Fit for a single class.
        """

        # smoothed (by alpha) count vectors
        p = (self.alpha + X[y == 1].sum(axis=0))
        q = (self.alpha + X[y == 0].sum(axis=0))

        r = log_normalize_count_vector(p) - log_normalize_count_vector(q)

        # scale X by log count ratio
        X_ = X.multiply(r)

        # fit svm classifier
        svm = LinearSVC(
            C=self.C
        ).fit(X_, y)

        mean_weight = np.abs(svm.coef_).mean()

        coef = (1 - self.beta) * mean_weight + self.beta * svm.coef_

        # reweight for prediction
        # (so that input X effectively gets elementwise product)
        coef *= r

        return coef, svm.intercept_


class BeamElement(NamedTuple):
    score: float
    prev_str: str
    next_str: str

class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict


def make_app(google_analytics_ua: str,
             bert_path: str,
             baseline_path: str,
             device: int = -1) -> Flask:
    with open(baseline_path, "rb") as f:
        model = pickle.load(f)
    nbsvm = model.steps[1][1]
    nbsvm.predict_proba = nbsvm._predict_proba_lr
    def nbsvm_predict(texts: List[str]):
        return model.predict_proba(texts)
    # splitter = SpacySentenceSplitter()
    split_expr = lambda x: x.split()
    nbsvm_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                  bow=True, split_expression=split_expr)
    # model_path = Path(bert_path)
    # archive = load_archive(model_path / "model.tar.gz")
    # bert_model = archive.model
    # bert_model.eval()
    # if device >= 0:
    #     bert_model.to(device)
    # params = Params.from_file(model_path / "config.json")
    # reader = DatasetReader.from_params(params.get("dataset_reader"))
    # batch_size = 32
    # bert_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
    #                               bow=False, split_expression=splitter.split_sentences)

    app = Flask(__name__) # pylint: disable=invalid-name

    # We hash the javascript file and use it as a cache breaker
    hasher = hashlib.md5()
    app_js = open("static/app.js")
    hasher.update(app_js.read().encode())
    js_hash=hasher.hexdigest()

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response: # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        return render_template(
            'app.html',
            google_analytics_ua=google_analytics_ua,
            js_hash=js_hash
        )

    @app.route('/static/<path:path>')
    def static_proxy(path: str) -> Response: # pylint: disable=unused-variable
        return send_from_directory('static', path)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict() -> Response:  # pylint: disable=unused-variable
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        previous_str = data["previous"]
        next_str = data.get("next")

        # Log the query
        app.logger.info(f"<{previous_str}> <{next_str}>")

        lime_tokens = split_expr(previous_str)

        model_name = data.get("model_name", "BERT")
        app.logger.info(f"Using model {model_name}")
        if model_name == 'NBSVM' or True:
            preds = model.predict([previous_str])
            class_probabilities = model.predict_proba([previous_str])[0].tolist()
            label = preds[0]
            explanation = nbsvm_explainer.explain_instance(previous_str, nbsvm_predict,
                                                     num_features=10,
                                                     labels=[1],
                                                     num_samples=100)
            score_dict = dict(explanation.as_list(1))
            lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
        # else:
        #     def _lime_predict(texts: List[str]) -> np.ndarray:
        #         with torch.no_grad():
        #             instances = [reader.text_to_instance(t) for t in texts]
        #             instance_chunks = [instances[x: x + batch_size] for x in
        #                                range(0, len(instances), batch_size)]
        #             preds = []
        #             for batch in instance_chunks:
        #                 pred = bert_model.forward_on_instances(batch)
        #                 preds.extend(pred)
        #         probs = [p['probs'] for p in preds]
        #         return np.stack(probs, axis=0)
        #
        #     inst = reader.text_to_instance(previous_str)
        #     print(inst.fields['tokens'].tokens)
        #     out = bert_model.forward_on_instance(inst)
        #     print(out.keys())
        #     class_probabilities = out['probs'].tolist()
        #     label = out['label']
        #     explanation = bert_explainer.explain_instance(previous_str, _lime_predict,
        #                                                    num_features=10,
        #                                                    labels=[1],
        #                                                    num_samples=100)
        #     score_dict = dict(explanation.as_list(1))
        #     lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
        app.logger.info(label)
        app.logger.info(lime_scores)
        app.logger.info(lime_tokens)
        app.logger.info(class_probabilities)

        return jsonify({
            "lime_scores": lime_scores,
            "lime_tokens": lime_tokens,
            "label": label,
            "class_probabilities": class_probabilities,
            "words": lime_tokens,
            "output": previous_str + (next_str or ""),
            "sentiment": label
        })

    return app


def main(args):
    # Executing this file with no extra options runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this, except possibly to test changes to the stock HTML).

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--port', type=int, default=8000, help='port to serve the demo on')
    parser.add_argument('--device', type=int, default=-1, help='device to host model on')
    parser.add_argument('--bert-path', type=str)
    parser.add_argument('--baseline-path', type=str)
    parser.add_argument('--dev', action='store_true', help='if true launch flask so that the server restarted as changes occur to the template')

    args = parser.parse_args(args)

    app = make_app(google_analytics_ua=os.environ.get(
        "GOOGLE_ANALYTICS_UA",
        "UA-120916510-5" # Defaults to the development / staging UA
    ), device=args.device,
    bert_path=args.bert_path,
        baseline_path=args.baseline_path)
    CORS(app)

    if args.dev:
        app.debug = True
        app.run(port=args.port, host='0.0.0.0')
        print(f"Model loaded, serving demo on port {args.port}")
    else:
        http_server = WSGIServer(('0.0.0.0', args.port), app, log=sys.stdout)
        print(f"Model loaded, serving demo on port {args.port}")
        http_server.serve_forever()

#
# HTML and Templates for the default bare-bones app are below
#

_HTML = """


"""

if __name__ == "__main__":
    main(sys.argv[1:])
