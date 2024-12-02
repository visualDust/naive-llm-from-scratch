# Hands on Large(?) Language model from scratch

tutorial: [LLM basics from scratch](https://www.gong.host/blog/2024/2/2/llm-from-scratch) provide step by step explanation.

---

## how to run

### Download Dataset

cd to `data` folder

```bash
cd data
```

Initialize Git LFS for Large Files

```bash
git lfs install
```

Clone the dataset:

```bash
git clone https://huggingface.co/datasets/Skylion007/openwebtext
```

Unzip dataset:

```python
bash unzip.sh
```


### Convert Data

Back to the root folder, run the following command:

```bash
python convert_data.py
```

It converts all the `.xz` files in `data/openwebtext/subsets` and put the converted `.txt` files in folder `data/extracted`.


We are using [neetbox](https://neetbox.550w.host) for monitoring, open [localhost:20202](http://localhost:20202/) (neetbox's default port) in your browser and you can check the progresses. If you are working on a remote server, you can use `ssh -L 20202:localhost:20202 user@remotehost` to forward the port to your local machine, or you can directly access the server's IP address with the port number, and you will see all the processes:

![image-20231226202536338](./imgs/readme/image-20231226202536338.png)

Optionally, the script will ask you if you'd like to delete the original `.xz` files to save disk space. If you want to keep them, type `n` and press Enter.

### train

```bash
python train.py --config config/gptv1_s.toml
```

Since we are using [neetbox](https://neetbox.550w.host) for monitoring, open [localhost:20202](http://localhost:20202/) (neetbox's default port) in your browser and you can check the progresses:

![image-20231226195105751](./imgs/readme/image-20231226194339598.png)

### predict

```bash
python inference.py --config config/gptv1_s.toml
```

Open [localhost:20202](http://localhost:20202/) (neetbox's default port) in your browser and feed text to your model via action button.

![image-20231226202121711](./imgs/readme/image-20231226202121711-1703604781869-4.png)



---

## further

more information see also [LLM basics from scratch](https://gavin.gong.host/blog/2023/11/19/llm-from-scratch)
