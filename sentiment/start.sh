#!/bin/bash

ls -lh /models

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_TYPE=en_US.UTF-8
locale-gen en_US.UTF-8
gunicorn --timeout 60 --bind 0.0.0.0:5000 wsgi
