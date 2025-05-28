
FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y && apt install -y awscli

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "app.py"]
