FROM python:3.6.8

WORKDIR /local
COPY requirements.txt /local/

RUN pip install -r requirements.txt
RUN pip install awscli

COPY . /local
RUN mkdir /root/.aws
COPY config /root/.aws/
COPY credentials /root/.aws/
# COPY  /root/.aws/config

RUN chmod 777 ./start.sh

CMD ["./start.sh"]