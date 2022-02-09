import fitz
import scipy.fftpack
import numpy as np
import argparse
import os
from typing import Callable, Union
from pathlib import Path
from tqdm import tqdm, trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help='path to a pdf file or a directory'
    )
    parser.add_argument(
        "-o", "--out", help="path to the output file")
    return parser.parse_args()


def pix2np(pix):
    '''
    Credit: https://stackoverflow.com/questions/53059007/python-opencv
    '''
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def scale(im, nR, nC):  
    '''
    Credit: https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image/48121996
    '''
    nR0 = len(im)
    nC0 = len(im[0])
    return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]


def phash(image, hash_size=32, highfreq_factor=4):
    pixels = np.asarray(image)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.astype(np.uint16)


def all_hash(images, hash_func: Callable[[np.ndarray], int]=phash) -> list([np.ndarray]):
    '''
    Calculate image hashes on a list of images.
    Arguments:
        - images (list): A list of images
        - hash_func (Callable[[np.ndarray], int]): Used image hash function. phash works the best.
    Returns:
        list - A list of numpy arrays, each representing an image hash.
    '''
    SIZE = 32
    total_len = len(images)
    hash_list = []
    print(f'Calculating hashes...')
    for i in trange(total_len):
        image = images[i]
        image = scale(image, SIZE, SIZE)
        hash_list.append(phash(image).reshape(-1))
    return hash_list

def cluster(diff: list, maxiter: int=200, seed: int=42, epsilon: int=0.1) -> tuple([int, list]):
    '''
    Run an unsupervised cluster algorithm on the list diff. Assumes the list can be divided by a clear threshold, return that threshold using the mean of the two clusters' centers.
    Arguments:
        - diff (list): The list of hash differences.
        - maxiter (int): Max number of K-means iteration.
        - epsilon (float): Range is (0, 1], the damping coefficient.
    Returns:
        int - The calculated threshold 
        list - The centroids
    '''
    assert len(diff) >= 2, f'The slide must have at least 3 pages'
    diff = np.array(diff)
    distances = np.zeros((diff.shape[0], 2))
    # initialize center of clusters
    np.random.seed(seed)
    np.random.shuffle(diff)

    group_center = diff.mean()
    centroids = np.random.normal(loc=group_center, scale=2, size=(2, ))
    
    for i in range(maxiter):
        centroids_temp = centroids.copy()
        distances[:, 0] = np.abs(diff - centroids[0])
        distances[:, 1] = np.abs(diff - centroids[1])
        classes = np.argmin(distances, axis=1)            
        centroids[0] = epsilon * np.mean(diff[classes == 0]) + (1 - epsilon) * centroids[0]
        centroids[1] = epsilon * np.mean(diff[classes == 1]) + (1 - epsilon) * centroids[1]
        if np.linalg.norm(centroids - centroids_temp) < 100:
            break

    centroids_sorted = centroids.tolist()
    centroids_sorted.sort()

    return (np.mean(centroids), centroids_sorted)

def drop_duplicates(fpath: Path, output: str=None, prefix: str='modified') -> None:
    '''
    Given a path to a pdf file, run the drop algorithm on it. Outputs the modified file.
    Three scnarios: 
        1. Single file with specified output. Will output to that location.
        2. Single file with specified output. Will output to ./$(prefix) using the original file name.
        3. Being called on a set of files. Each call is equal to (2).
    Arguments:
        - fpath (Path): Path to the pdf file.
        - output (str): string to the output location. If left None, will automatically create a file at the same location, with the prefix
    '''
    # Obtain hash values
    print(f'Handling {fpath}...')
    file = fitz.open(fpath)
    toc = file.getToC()
    images = [pix2np(page.getPixmap()) for page in file]
    hash_list = all_hash(images)
    # Calculate Hamming distances
    hash_altr = hash_list[1::]
    diff = []
    for i in range(len(hash_altr)):
        hash_diff = abs(hash_altr[i]-hash_list[i])
        hamming_dist = hash_diff.sum()
        diff.append(hamming_dist)
    threshold, centroids = cluster(diff)
    # The first and last pages will be kept
    diff = [centroids[1]] + diff + [centroids[1]]
    # Select pages to keep.
    geq = lambda i: i >= threshold
    pages_tokeep = list(map(geq, diff))
    page_numbers = []
    for i in range(len(pages_tokeep)-1):
        if pages_tokeep[i+1]:
            page_numbers.append(i)
    kept_page_count = sum(list(map(int, pages_tokeep)))
    file.select(page_numbers)
    # Refractor TOC.
    for i in range(len(toc)):
        pgn = toc[i][2]
        while not pages_tokeep[pgn]:
            pgn += 1
        pgn -= pgn - sum(list(map(int, pages_tokeep))[0:pgn])
        toc[i][2] = pgn
    file.setToC(toc)
    if output is None:
        output_loc = fpath.with_name(f'{prefix}').with_suffix('')
        try:
            output_loc.mkdir()
        except:
            pass
        output = output_loc/fpath.name
    else:
        output = Path(output)
    file.save(output)
    print(f'Dropped {len(hash_list) - kept_page_count} pages out of {len(hash_list)} pages.')
    print(f'New file saved at {output}')

def main():
    args = parse_args()
    fpath = Path(args.path)
    output = args.out        
    if fpath.is_dir():
        files = fpath.glob('*.pdf')
        for file in files:
            drop_duplicates(file)
    else:
        assert fpath.suffix in ['.pdf', '.PDF'], f'path argument is not pointing to a pdf file'
        drop_duplicates(fpath, output=output)

if __name__ == "__main__":
    main()
