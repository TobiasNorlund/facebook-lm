from model import FriendChatBot
from data import load_from_json, load_slack_data
import logging

if __name__ == "__main__":
    import argparse
    import sys

    # Setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser("Trains the chatbot model")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--train-data", dest="train_data")
    params = parser.parse_args()

    logging.info("Loading training data ...")
    message_chunks = load_slack_data(params.train_data)

    model = FriendChatBot(max_vocab_size=100, unk_token=False, save_dir=params.save_dir, text_col="message_chunk")

    # Load existing checkpoint is available
    if model.can_load():
        model.load()

    model.fit(message_chunks)
