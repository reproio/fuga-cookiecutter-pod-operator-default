FROM python:3.7

WORKDIR /work
COPY requirements.txt /work/requirements.txt

RUN pip install -r /work/requirements.txt

COPY . /work

CMD ["python", "main.py"]
