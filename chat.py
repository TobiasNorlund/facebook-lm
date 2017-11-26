from model import FriendChatBot

if __name__ == "__main__":
    import argparse
    import sys
    import logging
    import tensorflow as tf

    # Setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser("Chat using your trained chatbot model")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--my-name", dest="my_name")
    parser.add_argument("--friend-name", dest="friend_name")
    params = parser.parse_args()

    model = FriendChatBot(max_vocab_size=100, unk_token=False, save_dir=params.save_dir, text_col="message_chunk")

    # Don't occupy any GPU during chatting
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    model.session = tf.Session(config=config)

    # Load existing checkpoint is available
    if model.can_load():
        model.load()
    else:
        raise RuntimeError("Could not load a pretrained model from '{}'".format(params.save_dir))

    # Start chatting
    model.chat(params.my_name, params.friend_name)
