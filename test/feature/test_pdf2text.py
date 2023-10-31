#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Helper script to convert a PDF file to text.
#   Implemented as finite state machine
#
from difflib import SequenceMatcher
from enum import Enum
from pdf2image import convert_from_path
from pytesseract import image_to_string
import contextualSpellCheck as csc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import spacy


nlp = spacy.load("de_dep_news_trf")
CSC_IN_PIPELINE = False


class OcrPreparationException(Exception):
    pass


class OcrFailedOrNoTextFoundException(Exception):
    pass


class NlpNotReadyException(Exception):
    pass


class CscException(Exception):
    pass


class State(Enum):
    INITIALIZED = 1
    OCR_READY = 2 
    OCR_DONE_CSC_NOT_READY = 3
    NLP_CSC_READY = 4
    NLP_CSC_DONE = 5
    FINISHED = 6
    

class OCRContext:
    def __init__(self, file_path: str, file_lang: str = "deu", verbose: bool = False):
        if not os.path.exists(file_path):
            raise FileExistsError
        # Class variables from kwargs
        self.file_path: str = file_path
        self.file_lang: str = file_lang
        self.verbose: bool = verbose
        
        # Additional variables necessary
        self.content: list = list()
        self.csc_initialized: bool = False
        self.document_nlp = None
        self.document_plain = None
        self.state = State.INITIALIZED
        self.text_ocr: str = str()
        
    def prepare_file(self):
        if self.state == State.INITIALIZED:
            self.document_plain = convert_from_path(self.file_path)
            self.state = State.OCR_READY
            if self.verbose:
                print(f"[i] State changed: {self.state}")
            
        if self.state != State.OCR_READY:
            raise OcrPreparationException

    @staticmethod
    def preprocess(image, show: bool = False, blocksize: int = 51, constant: int = 15):
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_gray
        image = cv2.dilate(image, np.ones((7, 7), np.uint8))
        img_bg = cv2.medianBlur(image, 21)
        image = 255 - cv2.absdiff(img_gray, img_bg)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, constant)

        # view image
        if show:
            cv2.imshow("Augmented Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image
        
    def perform_ocr(self, bsc):
        if self.state == State.OCR_READY:
            for page_number, page_data in enumerate(self.document_plain):
                # Reset states
                if self.state == State.NLP_CSC_DONE:
                    self.state = State.OCR_READY
                    if self.verbose:
                        print(f"[i] State changed: {self.state}")

                # TODO: Remove after testing
                bs, c = bsc

                page_data = self.preprocess(np.array(page_data), show=False, blocksize=bs, constant=c)

                # Perform OCR
                self.text_ocr = image_to_string(page_data, lang=self.file_lang)
                
                # Update states
                if self.text_ocr != "":
                    if not self.csc_initialized:
                        self.state = State.OCR_DONE_CSC_NOT_READY
                        if self.verbose:
                            print(f"[i] State changed: {self.state}")
                    else:
                        self.state = State.NLP_CSC_READY
                        if self.verbose:
                            print(f"[i] State changed: {self.state}")
                else:
                    raise OcrFailedOrNoTextFoundException
                
                # Perform NLP and CSC
                self.perform_nlp(page_number)
            
            self.state = State.FINISHED
            if self.verbose:
                print(f"[i] State changed: {self.state}")
                
    def perform_nlp(self, page_number):
        if self.state == State.OCR_DONE_CSC_NOT_READY:
            # Add contextualSpellChecker to pipeline
            global CSC_IN_PIPELINE
            if not CSC_IN_PIPELINE:
                csc.add_to_pipe(nlp)
                self.csc_initialized = True
                self.state = State.NLP_CSC_READY
                CSC_IN_PIPELINE = True
                if self.verbose:
                    print(f"[i] State changed: {self.state}")
            else:
                self.state = State.NLP_CSC_READY
            
        # Perform NLP
        if self.state == State.NLP_CSC_READY:
            self.document_nlp = nlp(self.text_ocr)
        else:
            raise NlpNotReadyException
        
        if self.document_nlp._.performed_spellCheck:    
            page = dict()
            page["number"] = page_number + 1
            page["content"] = self.document_nlp._.outcome_spellCheck
            self.content.append(page)
            
            if self.verbose:
                print(f"[*] Page: {page['number']}:\n{page['content']}")
            
            self.state = State.NLP_CSC_DONE
            if self.verbose:
                print(f"[i] State changed: {self.state}")
        else:
            raise CscException


def main():
    # Disable parallelism (better performance as classification takes too long
    # and tokenization process gets forked before finishing)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ratio = list()

    c_low = 10  # 1
    c_high = 15  # 35
    c_step = 1
    b_low = 39  # 3
    b_high = 51
    b_step = 2

    iteration = 0
    max_iterations = len(range(b_low, b_high, b_step)) * len(range(c_low, c_high, c_step))

    for c in range(c_low, c_high, c_step):
        for bs in range(b_low, b_high, b_step):
            iteration += 1
            print(f"[i] Iteration {iteration}/{max_iterations}")

            # Perform OCR, tokenize, perform spell check
            bsc = (bs, c)

            try:
                ctx = OCRContext(file_path="../../resources/scans/sync/Benchmark.pdf", verbose=False)
                ctx.prepare_file()
                ctx.perform_ocr(bsc=bsc)
            except OcrFailedOrNoTextFoundException:
                print("[!] No text found")
                ratio.append([bs, c, 0.0])
                continue
            except NlpNotReadyException:
                print("[!] NLP/CSC Failed")
                ratio.append([bs, c, 0.0])
                continue

            # Diff result with transcript
            with open("../../resources/transcript_valid.txt", "r") as transcript:
                transcript_txt = str()

                while transcript.readline():
                    transcript_txt += transcript.readline()

                ocr_txt = str()

                for page in ctx.content:
                    ocr_txt += page["content"]

                diff = SequenceMatcher()
                diff.set_seq1(transcript_txt)
                diff.set_seq2(ocr_txt)
                match = diff.ratio()
                ratio.append([bs, c, match])

    bs = [b for b, c, r in ratio]
    c = [c for b, c, r in ratio]
    r = [r for b, c, r in ratio]

    # Plot results
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(xs=bs, ys=c, zs=r, zdir='z')
    ax.set_xlabel('Blocksize')
    ax.set_ylabel('Threshhold Constant')
    ax.set_zlabel('Match Ratio')
    ax.set_zlim(0, 1)

    plt.title("Sequence Match Ratio by OpenCV AdaptiveThreshold")
    plt.show()

    print(f"[i] Best ratio achieved: {np.max(np.array(r))}")

    exit(0)


if __name__ == "__main__":
    main()
