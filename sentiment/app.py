from typing import List
from pathlib import Path
import pickle
import logging
import hashlib
import numpy as np
import spacy
import sys
import torch
sys.path.append("nbsvm")

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
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

from nbsvm import NBSVM

from flask import Flask, request, Response, jsonify, render_template, send_from_directory

logging.basicConfig(level=logging.INFO)

stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [stemmer.stem(self.wnl.lemmatize(t)) for t in word_tokenize(articles) if t not in stopWords]

# this was done to make sure the model unpickles correctly (may not actually be necessary)
setattr(sys.modules["__main__"], LemmaTokenizer.__name__, LemmaTokenizer)


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


app = Flask(__name__) # pylint: disable=invalid-name

# We hash the javascript file and use it as a cache breaker
hasher = hashlib.md5()
app_js = open("static/app.js")
hasher.update(app_js.read().encode())
js_hash=hasher.hexdigest()

nlp = spacy.load('en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
with open("/models/nbsvm_imdb_sent_500.pkl", "rb") as f:
    model = pickle.load(f)
nbsvm = model.steps[1][1]
nbsvm.predict_proba = nbsvm._predict_proba_lr


def nbsvm_predict(texts: List[str]) -> np.ndarray:
    return model.predict_proba(texts)

split_expr = lambda text: [sent.string.strip() for sent in nlp(text).sents]
nbsvm_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                        bow=True, split_expression=split_expr)

model_path = Path("/models/bert_base_1000.tar.gz")
archive = load_archive(model_path)
bert_model = archive.model
bert_model.eval()
device = -1
if device >= 0:
    bert_model.to(device)
# params = Params.from_file(model_path / "config.json")
params = archive.config
reader = DatasetReader.from_params(params.get("dataset_reader"))
batch_size = 32
bert_explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                   bow=False, split_expression=split_expr)


@app.errorhandler(ServerError)
def handle_invalid_usage(error: ServerError) -> Response: # pylint: disable=unused-variable
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def index() -> Response: # pylint: disable=unused-variable
    return render_template(
        'app2.html',
        google_analytics_ua="UA-120916510-5",  # TODO:don't hardcode this!
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

    # TODO: the "next" is left over from the AllenAI app this was copied from
    previous_str = data["previous"]
    next_str = data.get("next")

    # Log the query
    app.logger.info(f"<{previous_str}> <{next_str}>")

    lime_tokens = split_expr(previous_str)

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
    
        inst = reader.text_to_instance(previous_str)
        print(inst.fields['tokens'].tokens)
        out = bert_model.forward_on_instance(inst)
        print(out.keys())
        class_probabilities = out['probs'].tolist()
        label = out['label']
        explanation = bert_explainer.explain_instance(previous_str, _lime_predict,
                                                        num_features=10,
                                                        labels=[1],
                                                        num_samples=100)
        score_dict = dict(explanation.as_list(1))
        lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=False)
