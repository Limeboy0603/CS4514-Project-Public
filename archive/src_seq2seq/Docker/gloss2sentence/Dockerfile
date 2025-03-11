FROM python:3.10-alpine

RUN apk add --no-cache gcc musl-dev linux-headers

WORKDIR /app

COPY ./src_gloss2sentence/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src_gloss2sentence/deploy.py deploy.py
ENV FLASK_APP=deploy.py
ENV FLASK_HOST=0.0.0.0
EXPOSE 5000

CMD ["flask", "run", "--debug"]