import argparse
import os

import cv2
from PIL import Image
import torch
from tqdm import tqdm


def main(args):
    progress_main = tqdm(
        ["/trainA", "/trainB", "/testA", "/testB"], leave=False
    )
    for folder in progress_main:
        cur_folder = args.path + folder
        os.mkdir(cur_folder + "_preprocessed")
        progress_folder = tqdm(sorted(os.listdir(cur_folder)), leave=False)
        for file in progress_folder:
            img = Image.open(cur_folder + "/" + file)
            torch.save(img, cur_folder + "_preprocessed/" + file[:-4] + ".pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    main(args)
