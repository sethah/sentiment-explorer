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

from allennlp.models.archival import load_archive
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

import torch
from flask import Flask, request, Response, jsonify, render_template, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from lm_explorer.lm.gpt2 import GPT2LanguageModel
from lm_explorer.util.sampling import random_sample

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


def make_app(google_analytics_ua: str) -> Flask:
    # model_117M = GPT2LanguageModel(model_name='117M')
    # model_345M = GPT2LanguageModel(model_name='345M')
    with open("/Users/shendrickson/nbsvm_imdb_sent_500.pkl", "rb") as f:
        model = pickle.load(f)
    nbsvm = model.steps[1][1]
    nbsvm.predict_proba = nbsvm._predict_proba_lr
    def nbsvm_predict(texts: List[str]):
        return model.predict_proba(texts)
    lime_predict = nbsvm_predict
    splitter = SpacySentenceSplitter()
    nbsvm_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                  bow=True, split_expression=splitter.split_sentences)
    model_path = Path("/Users/shendrickson/bert_large0")
    archive = load_archive(model_path / "model.tar.gz")
    bert_model = archive.model
    params = Params.from_file(model_path / "config.json")
    reader = DatasetReader.from_params(params.get("dataset_reader"))
    vocab = Vocabulary.from_files(model_path / "vocabulary")
    batch_size = 32
    bert_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                  bow=False, split_expression=splitter.split_sentences)

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

        lime_tokens = splitter.split_sentences(previous_str)

        model_name = data.get("model_name", "BERT")
        app.logger.info(f"Using model {model_name}")
        if model_name == 'NBSVM':
            preds = model.predict([previous_str])
            class_probabilities = model.predict_proba([previous_str])[0].tolist()
            label = preds[0]
            explanation = nbsvm_explainer.explain_instance(previous_str, nbsvm_predict,
                                                     num_features=10,
                                                     labels=[1],
                                                     num_samples=100)
            score_dict = dict(explanation.as_list(1))
            lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
        else:
            def _lime_predict(texts: List[str]) -> np.ndarray:
                with torch.no_grad():
                    instances = [reader.text_to_instance(t) for t in texts]
                    instance_chunks = [instances[x: x + batch_size] for x in
                                       range(0, len(instances), batch_size)]
                    preds = []
                    for batch in instance_chunks:
                        pred = bert_model.forward_on_instances(batch)
                        preds.extend(pred)
                probs = [p['probs'] for p in preds]
                return np.stack(probs, axis=0)

            # explanation = bert_explainer.explain_instance(previous_str, _lime_predict,
            #                                                num_features=10,
            #                                                labels=[1],
            #                                                num_samples=500)
            # score_dict = dict(explanation.as_list(1))
            # lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
            out = bert_model.forward_on_instance(reader.text_to_instance(previous_str))
            class_probabilities = out['probs'].tolist()
            # class_probabilities = [random.random() for _ in range(2)]
            lime_scores = [random.random() for _ in range(len(lime_tokens))]
            # class_probabilities = [x / max(class_probabilities) for x in class_probabilities]
            label = 'neg' if class_probabilities[0] > class_probabilities[1] else 'pos'
            # out = predictor.predict_json({'sentence': previous_str})
            # label = out['label']
        app.logger.info(label)
        app.logger.info(lime_scores)
        app.logger.info(lime_tokens)
        app.logger.info(class_probabilities)

        return jsonify({
            "logits": [-1.0, 2.0],
            "lime_scores": lime_scores,
            "lime_tokens": lime_tokens,
            "label": label,
            "class_probabilities": class_probabilities,
            "words": lime_tokens,
            "output": previous_str + (next_str or ""),
            "sentiment": label
        })

    # This endpoint isn't used, so it's commented out. You can re-enable
    # it by uncommenting it.
    #
    # @app.route('/random', methods=['POST', 'OPTIONS'])
    # def random() -> Response:  # pylint: disable=unused-variable
    #     if request.method == "OPTIONS":
    #         return Response(response="", status=200)
    #     data = request.get_json()
    #     previous_str = data["previous"]
    #     next_str = data.get("next", None)
    #     topk = data.get("topk", 10)
    #     num_steps = data.get('numsteps', 1)
    #     temperature = data.get("temperature", 1.0)
    #     logits = model.predict(previous_str, next_str)
    #     probabilities = torch.nn.functional.softmax(logits / temperature)
    #     samples = torch.multinomial(probabilities, num_samples=topk, replacement=False)
    #     outputs = [(f"{previous_str}{next_str or ''}", model[idx.item()]) for idx in samples]
    #     for _ in range(num_steps - 1):
    #         new_outputs = []
    #         for p, n in outputs:
    #             logits = model.predict(p, n)
    #             probabilities = torch.nn.functional.softmax(logits / temperature)
    #             random_id = random_sample(logits / temperature)
    #             random_word = model[random_id]
    #             random_word_logit = logits[random_id].item()
    #             random_word_probability = probabilities[random_id].item()
    #             new_outputs.append((f"{p}{n}", random_word))
    #         outputs = new_outputs
    #     return jsonify({
    #         "previous": previous_str,
    #         "words": [f"{p}{n}" for p, n in outputs],
    #         "logits": [0 for _ in outputs],
    #         "probabilities": [0 for _ in outputs]
    #     })

    # This endpoint isn't used, so it's commented out. You can re-enable
    # it by uncommenting it.
    #
    # @app.route('/beam', methods=['POST', 'OPTIONS'])
    # def beam() -> Response:  # pylint: disable=unused-variable
    #     if request.method == "OPTIONS":
    #         return Response(response="", status=200)
    #     data = request.get_json()
    #     previous_str = data["previous"]
    #     next_str = data.get("next", "")
    #     topk = data.get("topk", 10)
    #     num_steps = data['numsteps']
    #     def candidates(s1: str = "", s2: str = None, score: float = 0.0) -> List[BeamElement]:
    #         logits = model.predict(previous_str + s1, s2)
    #         log_probabilities = torch.nn.functional.log_softmax(logits) + score
    #         best_log_probabilities, best_indices = log_probabilities.topk(topk)
    #         new_str = s1 if s2 is None else s1 + s2
    #         beam = [BeamElement(lp.item() + score, new_str, model[idx.item()])
    #                 for lp, idx in zip(best_log_probabilities, best_indices)]
    #         return beam
    #     # Initial step
    #     beam = candidates(next_str)
    #     for i in range(num_steps - 1):
    #         new_beam: List[BeamElement] = []
    #         for element in beam:
    #             new_beam.extend(candidates(element.prev_str, element.next_str, element.score))
    #         new_beam.sort(key=lambda elt: elt.score, reverse=True)
    #         beam = new_beam[:topk]
    #     return jsonify({
    #         "previous": previous_str,
    #         "words": [elt.prev_str + elt.next_str for elt in beam],
    #         "logits": [elt.score for elt in beam],
    #         "probabilities": [elt.score for elt in beam]
    #     })


    return app


def main(args):
    # Executing this file with no extra options runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this, except possibly to test changes to the stock HTML).

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--port', type=int, default=8000, help='port to serve the demo on')
    parser.add_argument('--dev', action='store_true', help='if true launch flask so that the server restarted as changes occur to the template')

    args = parser.parse_args(args)

    app = make_app(google_analytics_ua=os.environ.get(
        "GOOGLE_ANALYTICS_UA",
        "UA-120916510-5" # Defaults to the development / staging UA
    ))
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
