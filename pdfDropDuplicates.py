from pdf2image import convert_from_path, convert_from_bytes
import scipy.fftpack
import cv2
import numpy as np
import PyPDF2
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", help="The directory where your source folder/PDF file at.")
    return parser.parse_args()


def phash(image, hash_size=32, highfreq_factor=4):
    pixels = np.asarray(image)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.astype(np.uint16)


def all_hash(images):
    SIZE = 32
    REWIND = 5
    total_len = len(images)
    hash_list = []
    for i in range(total_len):
        image = np.asarray(images[i].convert("L")).astype(np.float64)
        print("Page", str(i+1)+"/"+str(total_len), "hash obtained.")
        image = cv2.resize(image, (SIZE, SIZE))
        hash_list.append(phash(image).reshape(-1))
    return hash_list


def drop_duplicates(fpath):
    # Obtain hash values
    print("Reading PDF at", fpath+'...')
    images = convert_from_path(fpath)
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
    # Write to new file
    pdfReader = PyPDF2.PdfFileReader(fpath)
    pdfnums = pdfReader.numPages
    pdfWriter = PyPDF2.PdfFileWriter()
    kept_page_count = 0
    for num in range(pdfnums):
        if diff[num+1] >= 10000000:
            pageObj = pdfReader.getPage(num)
            pdfWriter.addPage(pageObj)
            kept_page_count += 1
    new_fpath = fpath.replace('.pdf', '_modified.pdf')
    with open(new_fpath, 'wb') as pdfOutputFile:
        pdfWriter.write(pdfOutputFile)
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
