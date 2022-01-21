# Nyquisis

Ever downloaded a lecture slide like this?

![](images/before.png)

The duplicates are used for presentation, but they become annoying when one wants to read the PDF slides. This simple tool automatically drops the duplicates in the PDF files, for a better readability. 

## Install

Use the following command to install Nyquisis, with its dependencies:

```shell 
$ pip3 install nyquisis
```

## Usage

Use command line to interact:

```shell
$ nyquisis ./ve216,chap1,teach.pdf
```

In this case, it will generate` ./output/ve216,chap1,teach.pdf`.

You can also specify the output location:

```shell
$ nyquisis ./ve216,chap1,teach.pdf -o ./chap1_output.pdf
```



![](images/modified.png)

The path can also be a directory:

```
$ nyquisis lecture_slides
```

This will traverse all PDF files under the directory, and generate duplicate-dropped versions in `./output` at the same directory as the original PDF file. In this mode the `-o ` or `--out` argument is invalid.

## Features
- [X] Generate PDF files that without duplicates.
- [X] Traverse a given directory to perform drop tasks.
- [X] Preserve the TOC structure in the original PDF file.

## Update Logs:

- 2022/1/20: Remastered.
- 2020/5/19: Now pdfDropDuplicates will preserve the original TOC structure.
- 2020/5/18: Code Refactored. Uses PyMuPDF now instead of PyPDF2 and pdf2image, and also rewrited some part to remove OpenCV dependency.
- 2020/5/17: Project INIT.

## Licence
[GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)
Visit my blog: [enoch2090](https://enoch2090.me)