import json
import pandas as pd

def load_from_json(file):
    with open("/data/jorun-train.json") as f:
        data = json.load(f)
        return pd.DataFrame(data["threads"][0]["messages"])


if __name__ == "__main__":
    df = load_from_json("/data/jorun-train.json")