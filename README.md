# Tutorial: Build a chatbot of your friend using recurrent neural networks

We all know about those moments when you feel lonely and none of your friends are responding you on facebook, not even
reading what you've sent. That's when you'll need some artificial company, together with an artificial friend that you 
already like.

In this tutorial we will build a chatbot that try to mimic your best friend, using all of your chat history from
facebook messenger.

So how will this work? What we are going to build is a _Recurrent Neural Network_ and train it to generate messages from
your friend, based on a current conversation. The steps are straightforward:

1. Download a dump of all your data from Facebook
2. Inspect your data and select which friend to "botify"
3. Train your model
4. Start chatting

### Prerequisite
And by the way, I provide a Dockerfile to build an environment with all requirements needed, so if you haven't installed
Docker already, go ahead.

[Download docker >>](https://docs.docker.com/engine/installation/)

Build the image we will use with:

```
make build
```

## 1. Download facebook data

Head to your [Facebook Settings](https://www.facebook.com/settings) and click "Download a copy".

You will be prompted for your facebook password and told to wait for an email with a link when the download is ready. 
This usually takes some 10-15 minutes. When you get your email, download the dump and unpack the zip.

## 2. Inspect your data and generate training data

The dump contains _all_ your data, including status updates, photos etc. We are only interested in the messages, which
are contained in the `html/messages.htm` page. The data comes in ugly HTML format but fortuantely, there is a handy 
[python library](https://github.com/ownaginatious/fbchat-archive-parser) to parse these 
files and extract the data into a better format.

To collect some basic stats of your messaging history, lets run:

```
$ make run DATA_DIR=/path/to/your/facebook-NAME
root@1a2b3c4d5f6e:/# fbcap /data/html/messages.htm -f stats

-------------------------------                                                                                           
 Statistics for John Smith
-------------------------------

Top 10 longest threads:

  [1] Andrew Ng (28262)
      - Andrew Ng [14338|50.73%]
      - John Smith [13924|49.27%]

  [2] Andrey Karpathy (23136)
      - Andrey Karpathy [11764|50.85%]
      - John Smith [11372|49.15%]
  
  ...
```

Now, pick a name you want to botify. In the example, we will take Mr. Karpathy. Note that you can only pick
a conversation with *two* parties, you and your friend, or else the script will fail.

To generate a training file, run:

```
root@1a2b3c4d5f6e:/# fbcap /data/html/messages.htm -t Karpathy -f json > /data/train-karpathy.json
```

## 3. Train your model

To train your bot model run the following:

```
$ make run DATA_DIR=/path/to/your/facebook-NAME
root@1a2b3c4d5f6e:/# python /code/train.py --train-data /data/train-karpathy.json --save-dir /data/
```

Training can take a long time, but after each epoch (full pass of your data) a checkpoint is saved. As soon as you
get a checkpoint you can (while still training) try to chat with your model as described below.

Once your training reaches losses < 1.0, the model start to generate somewhat language like text

## 4. Chat using your model

To start chatting with your trained model, run:

```
$ make run 
root@1a2b3c4d5f6e:/# python /code/chat.py --save-dir /data/ --my-name "<MY NAME>" --friend-name "<FRIEND NAME>"
```