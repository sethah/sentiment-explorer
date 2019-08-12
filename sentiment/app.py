from typing import Dict, List
from pathlib import Path
import pickle
import logging
import hashlib
import numpy as np
import spacy
import shutil
import sys
import tarfile
import tempfile
import torch
sys.path.append("nbsvm")

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from lime.lime_text import LimeTextExplainer

from allennlp.models.archival import load_archive
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


class LimePredictor(object):

    def __init__(self, idx2label: Dict[int, str]):
        self.idx2label = idx2label
        self.label2idx = {v: k for k, v in idx2label.items()}
        self.class_names = [idx2label[i] for i in range(len(self.idx2label))]

    def predict(self, text: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict_batch(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class NBSVMLimePredictor(LimePredictor):

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        nbsvm = self.model.steps[1][1]
        nbsvm.predict_proba = nbsvm._predict_proba_lr
        self.idx2label = {i: l for i, l in enumerate(nbsvm.classes_.tolist())}
        super(NBSVMLimePredictor, self).__init__(self.idx2label)

    def predict(self, text: str) -> Dict[str, np.ndarray]:
        out = {}
        out['label'] = self.model.predict([text])[0]
        logits = self.model.predict_proba([text])[0]
        out['logits'] = logits
        out['probs'] = logits
        return out

    def predict_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.predict_proba(texts)


class AllenNLPLimePredictor(LimePredictor):

    def __init__(self, archive_path: str, device: int = -1, batch_size: int = 32):
        archive_path = Path(archive_path)
        archive = load_archive(archive_path)
        self.params = archive.config
        self.model = archive.model.eval()
        self.batch_size = batch_size
        self.reader = DatasetReader.from_params(self.params.get("dataset_reader"))
        self.vocab = self._load_vocab(archive_path)
        self.idx2label = self.vocab.get_index_to_token_vocabulary('labels')
        if device != -1:
            self.model.to(f"cuda:{device}")
        super(AllenNLPLimePredictor, self).__init__(self.idx2label)

    @staticmethod
    def _load_vocab(archive_path: Path) -> Vocabulary:
        # an annoying hack to load the vocab file
        tempdir = tempfile.mkdtemp()
        with tarfile.open(archive_path, 'r:gz') as _archive:
            _archive.extractall(tempdir)
        vocab_path = Path(tempdir) / "vocabulary"
        vocab = Vocabulary.from_files(vocab_path)
        shutil.rmtree(tempdir)
        return vocab

    def predict(self, text: str) -> Dict[str, np.ndarray]:
        return self.model.forward_on_instance(self.reader.text_to_instance(text))

    def predict_batch(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            instances = [self.reader.text_to_instance(t) for t in texts]
            instance_chunks = [instances[x: x + self.batch_size] for x in
                               range(0, len(instances), self.batch_size)]
            preds = []
            for batch in instance_chunks:
                pred = self.model.forward_on_instances(batch)
                preds.extend(pred)
        probs = [p['probs'] for p in preds]
        return np.stack(probs, axis=0)


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
hasher.update(app_js.read().encode('utf-8'))
js_hash = hasher.hexdigest()

nlp = spacy.load('en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
split_expr = lambda text: [sent.string.strip() for sent in nlp(text).sents]

nbsvm_predictor = NBSVMLimePredictor("/models/nbsvm_imdb_sent_500.pkl")
device = 0 if torch.cuda.is_available() else -1
bert_predictor = AllenNLPLimePredictor("/models/bert_large_2000.tar.gz", device=device)

nbsvm_explainer = LimeTextExplainer(class_names=nbsvm_predictor.class_names,
                                  bow=True, split_expression=split_expr)
bert_explainer = LimeTextExplainer(class_names=bert_predictor.class_names,
                                    bow=False, split_expression=split_expr)
models = {
    'bert': {'explainer': bert_explainer, 'predictor': bert_predictor},
    'nbsvm': {'explainer': nbsvm_explainer, 'predictor': nbsvm_predictor}
}


@app.errorhandler(ServerError)
def handle_invalid_usage(error: ServerError) -> Response: # pylint: disable=unused-variable
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def index() -> Response: # pylint: disable=unused-variable
    return render_template(
        'app.html',
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

    model_name = data.get("model_name", "BERT").lower()
    predictor = models[model_name]['predictor']
    explainer = models[model_name]['explainer']
    app.logger.info(f"Using model {model_name}")
    out = predictor.predict(previous_str)
    class_probabilities = out['probs'].tolist()
    label = out['label']
    explanation = explainer.explain_instance(previous_str, predictor.predict_batch,
                                             num_features=10, labels=[1], num_samples=100)
    score_dict = dict(explanation.as_list(1))
    lime_scores = [score_dict.get(tok, 0.) for tok in lime_tokens]
    if predictor.label2idx['neg'] != 0:
        # we need to reverse the lime scores
        lime_scores = [-1 * score for score in lime_scores]
    # make sure class probabilities are always consistently ordered
    class_probabilities = [class_probabilities[predictor.label2idx[lbl]] for lbl in ['neg', 'pos']]
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
