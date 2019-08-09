#!/bin/bash

# mkdir models

#aws s3 sync  s3://sentiment-explorer ./models/
# wget -P /models/ https://sentiment-explorer.s3-us-west-1.amazonaws.com/nbsvm_imdb_sent_500.pkl
# wget -P /models/ https://sentiment-explorer.s3-us-west-1.amazonaws.com/bert_base_1000.tar.gz

ls -lh /models

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_TYPE=en_US.UTF-8
echo "LC_ALL=en_US.UTF-8" >> /etc/environment
echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
echo "LANG=en_US.UTF-8" > /etc/locale.conf
locale-gen en_US.UTF-8
LANGUAGE=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LC_TYPE=en_US.UTF-8 python -c "import sys; print(sys.stdout.encoding)"
LANGUAGE=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LC_TYPE=en_US.UTF-8 gunicorn --timeout 60 --bind 0.0.0.0:5000 wsgi
