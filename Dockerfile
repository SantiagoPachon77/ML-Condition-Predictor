FROM python:3.10.16

WORKDIR /workdir

COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the rest of code
COPY . .

# set default import directories for process
ENV PYTHONPATH="/workdir/src:/workdir/dev"

ENTRYPOINT ["/bin/bash", "-l", "-c"]