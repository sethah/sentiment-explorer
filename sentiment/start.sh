#!/bin/bash

mkdir models

#aws s3 sync  s3://sentiment-explorer ./models/
wget -P ./models/ https://sentiment-explorer.s3-us-west-1.amazonaws.com/nbsvm_imdb_sent_500.pkl
wget -P ./models/ https://sentiment-explorer.s3-us-west-1.amazonaws.com/bert_base_1000.tar.gz

ls -lh ./models

gunicorn --bind 0.0.0.0:5000 wsgi