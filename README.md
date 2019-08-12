## Running with Docker

Download models to host so that they don't need to be downloaded on startup every time.

```bash
mkdir $HOME/.models
mkdir -p /$HOME/.pytorch_pretrained_bert
pushd $HOME/.models
curl -O https://sentiment-explorer.s3-us-west-1.amazonaws.com/nbsvm_imdb_sent_500.pkl
curl -O https://sentiment-explorer.s3-us-west-1.amazonaws.com/bert_base_1000.tar.gz
popd
```

Run with docker-compose (webapp + nginx)

```bash
docker-compose build
docker-compose up
```

Run just the web app with docker

```bash
cd sentiment
nvidia-docker build -t sentiment-explorer:latest .
nvidia-docker run -p 5000:5000 \
-v $HOME/.models/:/models \
-v $HOME/.pytorch_pretrained_bert/:/root/.pytorch_pretrained_bert \
sentiment-explorer:latest
```

## Run without docker

Create a Python virtual environment (Python 3.6 or later)

```bash
cd sentiment
pip install -r requirements.txt
gunicorn --timeout 60 --bind 0.0.0.0:5000 wsgi
```



