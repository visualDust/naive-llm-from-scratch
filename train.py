import os
import mmap
import time
import toml
import torch
import pickle
import neetbox
import argparse
from tqdm import tqdm
from random import randint
from model import GPTLangModel
from neetbox.logging import logger
from neetbox.utils import ResourceLoader

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

# load dataset from parsed .txt files. Please run 'convert_data.py' before run train process.
txt_loader = ResourceLoader(
    folder=config["data_folder"], file_types=["txt"], force_rescan=True
)  # scan .txt files from ./data
train_txts = [
    txt for txt in txt_loader.get_file_list() if "test" not in txt
]  # .txt file list for train
test_txts = [
    txt for txt in txt_loader.get_file_list() if "test" in txt
]  # .txt file for test


def get_random_text_chunk(train_or_test, batch_size, block_size):
    """get random text chunk from all .txt files

    Args:
        train_or_test (str): for train or test.

    Returns:
        Torch.Tensor: the encoded text chunk
    """
    random_file = (
        train_txts[randint(0, len(train_txts) - 1)]
        if train_or_test == "train"
        else test_txts[randint(0, len(test_txts) - 1)]
    )
    with open(random_file, "rb") as f:
        # using memory mapping
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # determine the file size and a random position to start with
            filesize = len(mm)
            pos_start = randint(0, filesize - block_size * batch_size)
            # seek to the random position and read the block into memory
            mm.seek(pos_start)
            mem_block = mm.read(block_size * batch_size - 1)
            # decode the block into a string, ignoring any invalid sequences
            block_decoded = mem_block.decode("utf-8", errors="ignore").replace("\r", "")
            # convert to tensor
            data = torch.tensor(encode(block_decoded), dtype=torch.long)
    return data


def get_batch(
    data, batch_size, block_size
):  # generate a each-time-random batch with batchsize
    # ix indicates random index in the text. Its size is batch_size that you can use it to sample batch_size times.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix) # ix is random location(index) in text
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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
if train_config["resume"]:
    try:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
                logger.ok("model loaded from existing pkl  file")
    except:
        logger.err("Error occured while loading from existing   weight. ignoring...")
    # move model to target device

model = model.to(device)
logger.ok(f"{model.__class__} now on {device}, ready to train.")

@neetbox.action(name="chat")
def inference(prompt: str):
    """chat with model

    Args:
        prompt (str): your input

    Returns:
        str: model's response
    """
    model.eval()
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(
        model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()
    )
    model.train()
    return generated_chars

# define how to test
@torch.no_grad()
def test(num_steps, batch_size, block_size):
    model.eval()
    loss_list = torch.zeros(num_steps)
    for i in range(num_steps):
        random_text_chunk = get_random_text_chunk("test", batch_size, block_size)
        test_x, test_y = get_batch(random_text_chunk, batch_size, block_size)
        _, loss = model(test_x, test_y)
        loss_list[i] = loss.item()
    loss = loss_list.mean()
    model.train()
    return loss


@neetbox.action(name="stop")
def stop_and_exit_anyway():
    """stop and exit train anyway"""
    os._exit(0)


learning_rate = train_config["learning_rate"]
max_iter = train_config["max_iter"]
eval_per_iter = test_config["eval_frequency"]

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train process
neetbox.add_hyperparams(config)
logger.info(f"train started")
current_loss = 114514.0
for iter in neetbox.progress(tqdm(range(max_iter)), name="Train"):
    # random batch data
    random_text_chunk = get_random_text_chunk("train", batch_size, block_size)
    xb, yb = get_batch(random_text_chunk, batch_size, block_size)
    # get predict and loss
    logits, loss = model.forward(xb, yb)
    neetbox.add_scalar(name="loss train", x=iter, y=loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if iter and iter % eval_per_iter == 0:
        test_loss = test(test_config["test_iter"], batch_size, block_size)
        test_loss = test_loss.item()
        neetbox.add_scalar(name="loss test", x=iter, y=test_loss)
        if test_loss <= current_loss:
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)  # save model to file
                logger.log("model saved to pkl file")
            current_loss = test_loss
    optimizer.step()

# test model
test_prompt = test_config["test_prompt"]
logger.info(f"prompt: {test_prompt}")
context = torch.tensor(encode(test_prompt), dtype=torch.long, device=device)
generated_chars = decode(
    model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()
)
logger.info(f"response: {generated_chars}")

logger.ok("train complete.")

logger.ok("model ready, please open http://localhost:20202/ in your browser to chat")
while True:
    time.sleep(1)
