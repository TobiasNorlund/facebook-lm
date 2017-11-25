FROM python:3.5

RUN pip3 install fbchat-archive-parser==1.2 \
                 numpy==1.13.3 \
                 pandas==0.21.0 \
                 tensorflow==1.4.0 \
                 tqdm==4.19.4
