FROM python:3.5
RUN mkdir resources
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
