FROM python:3.5

RUN pip3 install fbchat-archive-parser \
                 numpy \
                 pandas \
                 tensorflow
