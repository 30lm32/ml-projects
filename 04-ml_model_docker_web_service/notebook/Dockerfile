FROM python:3.5
RUN mkdir resources
ADD . /code
WORKDIR /code
COPY jupyter_notebook_config.py /root/.jupyter/
RUN pip install -r requirements.txt
CMD ["sh", "run_jupyter.sh", "--allow-root"]