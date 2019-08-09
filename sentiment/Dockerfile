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

ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN cd /tmp && \
    python3.6 -m pip install -U pip && \
    python3.6 -m pip install -U cython awscli && \
    rm -rf /tmp/* ~/.cache/pip 

RUN python3.6 -m pip install -f https://download.pytorch.org/whl/cu90/stable -r requirements.txt
RUN python3.6 -m pip install awscli

COPY . /local
# RUN mkdir /root/.aws
# COPY s3config/config /root/.aws/
# COPY s3config/credentials /root/.aws/
# COPY  /root/.aws/config

RUN chmod 777 ./start.sh

CMD ["./start.sh"]
