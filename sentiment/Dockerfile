FROM nvidia/cuda:9.0-base

WORKDIR /local
COPY requirements.txt /local/

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6-dev && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        libcurl4-openssl-dev \
	locales \
        libssl-dev \
	libffi-dev \
	python3-dev \
        python-setuptools \
        python-h5py \
        libhdf5-dev \
        libhdf5-serial-dev \
        daemontools \
        wget \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        git \
        gfortran \
        libopenblas-dev \
        nginx \
        supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN cd /tmp && \
    python3.6 -m pip install -U pip && \
    rm -rf /tmp/* ~/.cache/pip

RUN python3.6 -m pip install -f https://download.pytorch.org/whl/cu90/stable -r requirements.txt
RUN python3.6 -m pip install awscli
RUN python3.6 -m nltk.downloader punkt
RUN python3.6 -m nltk.downloader wordnet
RUN python3.6 -m nltk.downloader stopwords

COPY . /local

RUN chmod 777 ./start.sh

CMD ["./start.sh"]
