import argparse
import os
import time
import toml
import torch
from model import GPTLangModel
import pickle
import neetbox
from neetbox.logging import logger


parser = argparse.ArgumentParser(prog="GPTv1 train", description="Train GPTv1 model")
parser.add_argument("-c", "--config")
launch_args = parser.parse_args()

# load config file
config = toml.load(launch_args.config)

# get configs
train_config = config["train"]
test_config = config["test"]
model_config = config["model"]

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# read vocabulary set from file
with open(config["vocabulary"], "r", encoding="utf-8") as f:
    text = f.read()
    vocabulary = sorted(list(set(text)))

# build char embedding from vocabulary set
string2int = {ch: i for i, ch in enumerate(vocabulary)}
int2string = {i: ch for i, ch in enumerate(vocabulary)}
encode = lambda s: [string2int[c] for c in s]
decode = lambda l: "".join([int2string[i] for i in l])

# get batch size and block size from config
block_size = model_config["block_size"]
batch_size = train_config["batch_size"]

# build model
model = GPTLangModel(
    vocabulary_size=len(vocabulary),
    n_decoder=model_config["n_decoder"],
    n_head=model_config["n_head"],
    n_embed=model_config["n_embed"],
    block_size=block_size,
)

# try load model weight from file
try:
    logger.log("loading model prams...")
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            logger.ok("model loaded from existing pkl file")
    else:
        raise Exception(
            "model weights not found! Please run train before run inference."
        )
except:
    logger.err(
        RuntimeError("error occured while loading from existing weight. Aborting...")
    )
    os._exit(255)  # die

# move model to target device
model = model.to(device)
logger.ok(f"{model.__class__} now on {device}")


@neetbox.action(name="exit")
def stop_and_exit():
    """stop and exit"""
    os._exit(0)


@neetbox.action(name="chat")
def inference(prompt: str):
    """chat with model

    Args:
        prompt (str): your input

    Returns:
        str: model's response
    """
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(
        model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()
    )
    return generated_chars


logger.ok("model ready, please open http://localhost:20202/ in your browser to chat")
while True:
    time.sleep(1)
