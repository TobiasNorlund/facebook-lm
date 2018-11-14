import json
import pandas as pd

ME_START_CHAR = "\u001C"
FRIEND_START_CHAR = "\u001D"

USER_CHAR_MAP = {
    "U0QLW8NF5": "\u001C", # Tobias
    "U0C9CPXPF": "\u001D", # Samuel
    "U096C2P3R": "\u001E", # Fredrik
    "U4J26EJ8L": "\u001F", # Ruslan
    "U3NJT3FU4": "\u0020", # Nuno
    "U03JW9T8A": "\u0021", # Carl
    "U51DGR3FH": "\u0022"  # Euan
}

USER_NAME_MAP = {
    "\u001C": "Tobias", # Tobias
    "\u001D": "Samuel", # Samuel
    "\u001E": "Fredrik", # Fredrik
    "\u001F": "Ruslan", # Ruslan
    "\u0020": "Nuno", # Nuno
    "\u0021": "Carl", # Carl
    "\u0022": "Euan"  # Euan
}


def load_from_json(file, chunk_length=10):
    with open(file) as f:
        data = json.load(f)
        me = data["user"]
        assert len(data["threads"][0]["participants"]) == 1, "Please use a conversation between two persons only"
        conversation = data["threads"][0]["messages"]

    chunks = []
    current_chunk = []
    for message in conversation:
        if message["sender"] == me:
            current_chunk.append(ME_START_CHAR + message["message"])
        else:
            current_chunk.append(FRIEND_START_CHAR + message["message"])

        if len(current_chunk) >= chunk_length:
            chunks.append("".join(current_chunk))
            current_chunk = []

    df = pd.DataFrame(chunks, columns=["message_chunk"])
    df["len"] = df["message_chunk"].apply(lambda x: len(x))  # Process shorter chunks first
    df = df[df["len"] < 2000]  # Only process chunks shorter than 2000 chars, otherwise memory blows up
    df.sort_values(by="len", ascending=True, inplace=True)
    print("Num chunks: {}".format(len(df)))
    return df


def load_slack_data(file, chunk_length=10):
    with open(file) as f:
        data = json.load(f)

    data["messages"].reverse()

    chunks = []
    current_chunk = []
    for message in data["messages"]:
        if "user" in message and message["user"] in USER_CHAR_MAP:
            current_chunk.append(USER_CHAR_MAP[message["user"]] + message["text"])

            if len(current_chunk) >= chunk_length:
                chunks.append("".join(current_chunk))
                current_chunk = []

    df = pd.DataFrame(chunks, columns=["message_chunk"])
    df["len"] = df["message_chunk"].apply(lambda x: len(x))  # Process shorter chunks first
    df = df[df["len"] < 2000]  # Only process chunks shorter than 2000 chars, otherwise memory blows up
    df.sort_values(by="len", ascending=True, inplace=True)
    print("Num chunks: {}".format(len(df)))
    return df