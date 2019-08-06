FROM python:3.6.8

WORKDIR /local
COPY requirements.txt /local/

RUN pip install -r requirements.txt
RUN pip install awscli

COPY . /local
# RUN mkdir /root/.aws
# COPY s3config/config /root/.aws/
# COPY s3config/credentials /root/.aws/
# COPY  /root/.aws/config

RUN chmod 777 ./start.sh

CMD ["./start.sh"]