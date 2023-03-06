FROM tensorflow/tensorflow:latest-gpu
RUN mkdir /usr/bin/code
COPY . /usr/bin/code
RUN ls /usr/bin/code -la
RUN ls /usr/bin/code/hyper_data_loader -la
RUN ls /usr/bin/code/models -la
ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/code"
RUN pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir keras &&\
pip install --no-cache-dir numpy &&\
pip install --no-cache-dir scipy &&\
pip install --no-cache-dir Pillow &&\
pip install --no-cache-dir pathos &&\
pip install --no-cache-dir IPython


RUN pip install scikit-learn
RUN pwd
RUN cd /usr/bin/code
RUN ls -la

RUN export PYTHONPATH=/usr/bin/code
RUN export TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /usr/bin/code/Experiments
CMD python ExtensiveSearch.py 1

#docker build . -t hyper:0.1
#docker run -d -v /var/log:/var/log hyper:0.0
