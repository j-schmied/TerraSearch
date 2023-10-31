#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Helper script to convert a PDF file to tiff image.
#
from os import mkdir, path
from sys import argv
from wand.api import library
from wand.image import Image


BITRATE = 8  # 8 Bit because 16 bit are not supported by jTessBoxEditor
OUTPUT_DIR = '../output'
RESOLUTION = 300  # required resolution for box files


def main():
    if len(argv) != 2:
        print("Usage: %s <pdf_file>" % argv[0])
        exit(1)

    file = str(argv[1])

    # Check if file exists
    if not path.exists(file):
        print("File '%s' does not exist." % file)
        exit(1)

    with Image(filename=file, resolution=RESOLUTION, depth=BITRATE) as img:
        img.type = 'grayscale'
        img.compression = 'lzw'
        library.MagickResetIterator(img.wand)

        for i in range(library.MagickGetNumberImages(img.wand)):
            library.MagickSetIteratorIndex(img.wand, i)
            img.alpha_channel = 'off'

        if not path.exists(OUTPUT_DIR):
            mkdir(OUTPUT_DIR)
        
        destination_file = f"{OUTPUT_DIR}/{file.split('/')[-1].split('.')[0]}.tiff"
        img.save(filename=destination_file)

        exit(0)


if __name__ == '__main__':
    main()
