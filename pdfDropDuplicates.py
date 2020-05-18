import fitz
import scipy.fftpack
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", help="The directory where your source folder/PDF file at.")
    return parser.parse_args()


def pix2np(pix):  # Credit: https://stackoverflow.com/questions/53059007/python-opencv
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def scale(im, nR, nC):  # Credit: https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image/48121996
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


def all_hash(images):
    SIZE = 32
    total_len = len(images)
    hash_list = []
    for i in range(total_len):
        image = images[i]
        print("Page", str(i+1)+"/"+str(total_len), "hash obtained.")
        image = scale(image, SIZE, SIZE)
        hash_list.append(phash(image).reshape(-1))
    return hash_list


def drop_duplicates(fpath):
    # Obtain hash values
    print("Reading PDF at", fpath+'...')
    file = fitz.open(fpath)
    toc = file.getToC()
    images = list(pix2np(page.getPixmap()) for page in file)
    hash_list = all_hash(images)
    # Calculate Hamming distances
    hash_altr = hash_list[1::]
    diff = [15000000]
    for i in range(len(hash_altr)):
        hash_diff = abs(hash_altr[i]-hash_list[i])
        hamming_dist = hash_diff.sum()
        diff.append(hamming_dist)
    diff.append(15000000)
    print("All Hamming distance calculated.")
    # Select pages to keep.
    THRESHOLD = 10000000
    def geq(i): return i >= THRESHOLD
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
    new_fpath = fpath.replace(".pdf", "_modified.pdf")
    file.save(new_fpath)
    print("Dropped", len(hash_list)-kept_page_count,
          "pages out of", len(hash_list), "pages.")
    print("New file saved at", new_fpath+'.')


def scan(dirc, file_list=[]):
    os.chdir(dirc)
    dir_list = os.listdir(dirc)
    for item in dir_list:
        if os.path.isfile(os.path.join(dirc, item)) and (('.pdf' in item) or ('.PDF' in item)):
            file_list.append(os.path.join(dirc, item))
        if os.path.isdir(os.path.join(dirc, item)):
            file_list = scan(os.path.join(dirc, item), file_list)
        else:
            pass
    return file_list


if __name__ == "__main__":

    args = parse_args()
    if args.dir == None:
        print("Argument missing: input path")
    else:
        fpath = args.dir
        if os.path.isdir(fpath):
            files = scan(fpath)
            for file in files:
                drop_duplicates(file)
        else:
            if '.pdf' in fpath or '.PDF' in fpath:
                drop_duplicates(fpath)
            else:
                print('Format ERROR：not a valid PDF file')
