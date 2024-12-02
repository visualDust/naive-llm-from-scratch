import os
import math
import lzma
import neetbox
from neetbox import logger
from neetbox.utils import ResourceLoader
from neetbox.extension.machine import hardware
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures

vocabulary_set = set()  # vocabulary set


def xzs2txt(xz_file_list, txt_file_name):
    global vocabulary_set
    with open(txt_file_name, "w", encoding="utf-8") as outfile:
        for file_path in neetbox.progress(xz_file_list, name=txt_file_name):
            with lzma.open(file_path, "rt", encoding="utf-8") as current_xz:
                text = current_xz.read()
                outfile.write(text)
                characters = set(text)
                vocabulary_set.update(characters)
    return f"converted {len(xz_file_list)} files into {txt_file_name}"


def multi_thread_convert(xz_files, out_file_pattern):
    num_xz_files = len(xz_files)
    max_files_in_memory = int(
        hardware.memory.available * 0.8
    )  # assert each xz file will take 1MB to be loaded in to ram, and we would use 80% of availiable memory to convert xz  files
    num_workers = math.ceil(num_xz_files / max_files_in_memory)
    num_workers = 1 if num_workers == 0 else num_workers
    num_workers = max(num_workers, len(hardware.cpus))
    num_files_in_memory = num_xz_files // num_workers
    logger.log(
        f"max files in memory = {max_files_in_memory}, available memory {hardware.memory.available}MB"
    )
    logger.log(f"Using {num_workers} workers to convert")
    futures = []
    with ThreadPoolExecutor(num_workers) as executor:
        for i in range(num_workers):  # using num_workers workers
            start_index = i * num_files_in_memory
            end_index = (
                num_xz_files
                if (i + 1) * num_files_in_memory > num_xz_files
                else (i + 1) * num_files_in_memory
            )
            future = executor.submit(
                xzs2txt, xz_files[start_index:end_index], out_file_pattern.format(i)
            )  # submit convert task to excutor
            futures.append(future)
            logger.log(
                f"submitted convert task for files in range [{start_index}, {end_index})"
            )
    logger.info("waiting for converters...")
    wait_futures(futures)
    for future in futures:
        try:
            logger.info(f"returned from converter. {future.result()}")
        except Exception as e:
            logger.err(
                RuntimeError(f"converter encountered {e}, check file status."),
                reraise=True,
            )


if __name__ == "__main__":
    xz_file_folder = "./data/openwebtext/subsets"
    output_file_train = "./data/extracted/train{}.txt"
    output_file_test = "./data/extracted/test{}.txt"
    vocabulary_file = "./data/extracted/vocab.voc"

    # create extracted folder if not exist
    os.makedirs(os.path.dirname(output_file_train), exist_ok=True)

    xz_files = ResourceLoader(
        folder=xz_file_folder, file_types=["xz"], sub_dirs=False
    ).get_file_list()

    num_xz_files = len(xz_files)
    test_ratio = 0.1  # k fold ratio
    num_xz_files_test = max(int(num_xz_files * test_ratio), 1)
    test_xz_files = xz_files[-num_xz_files_test:]  # xz files for test
    train_xz_files = xz_files[:-num_xz_files_test]  # xz files for train

    # do convert
    logger.info("-> convert train set")
    multi_thread_convert(train_xz_files, output_file_train)
    logger.info("-> convert test set")
    multi_thread_convert(test_xz_files, output_file_test)

    # write vocabulary set into file
    logger.log(f"saving vocabulary set into {vocabulary_file}...")
    with open(vocabulary_file, "w", encoding="utf-8") as voc_file:
        for char in vocabulary_set:
            voc_file.write(char + "\n")

    logger.log("All tasks completed.")

    # open localhost:20202 in browser to see the progress

    # ask user whether to delete data/openwebtext folder
    # father folder of xz_file_folder
    raw_folder = os.path.dirname(xz_file_folder)
    if input(f"Do you want to delete folder {raw_folder}? (y/n): ").lower() == "y":
        os.system(f"rm -rf {raw_folder}")
        logger.log(f"folder {raw_folder} deleted.")
    else:
        logger.log("folder {raw_folder} not deleted.")
