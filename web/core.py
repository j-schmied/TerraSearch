import contextualSpellCheck as csc
import cv2
import googlemaps
import numpy as np
import os
import pandas as pd
import spacy
from enum import Enum
from geopy.geocoders import GoogleV3
from hashlib import md5
from os import environ, getenv, urandom
from pdf2image import convert_from_path
from pytesseract import image_to_string

# Clients
geolocator = GoogleV3(api_key=getenv("GOOGLE_API_KEY"))
gmaps = googlemaps.Client(key=getenv("GOOGLE_API_KEY"))

# CONFIG
ALLOWED_EXTENSIONS = {'pdf'}
CERT_FILE = '/data/certs/ca/ca.crt'
CSC_INITIALIZED = False
DEBUG = True
ELASTIC_HOST = 'https://es:9200'
ELASTIC_INDEX = 'ocr'
ELASTIC_MAPPING = {
    "properties": {
        "document_pdf": {
            "type": "nested",
            "properties": {
                "file_name": {"type": "text"},
                "raw_file": {"type": "text"},
                "pages": {"type": "integer"},
                "signature": {"type": "text"}
            }
        },
        "document_ocr": {
            "type": "nested",
            "properties": {
                "page_number": {"type": "integer"},
                "page_content": {"type": "text"}
            }
        },
        "document_metadata": {
            "type": "nested",
            "properties": {
                "location": {"type": "geo_point"},
                "timestamp": {"type": "date"}
            }
        }
    }
}
ELASTIC_SETTINGS = {
    "number_of_shards": 1,
    "index": {
        "similarity": {
            "default": {
                "type": "BM25",
                "b": 0.75,
                "k1": 1.2
            }
        }
    }
}
ELASTIC_USER = getenv('ELASTIC_USER')
ELASTIC_PASSWORD = getenv('ELASTIC_PASSWORD')
SECRET_KEY = urandom(64)  # using /dev/urandom so there is no blocking
SYNC_DIR = getenv("SYNC_DIR")
DOWNLOAD_FOLDER = './static'
UPLOAD_FOLDER = '/data/files'


# HELPERS
# check if file is allowed (or has the right extension at least)
def is_valid_file(filename) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_document_hash(document_path: str) -> str:
    hasher = md5()
    bs = 65536

    with open(document_path, 'rb') as fp:
        while True:
            data = fp.read(bs)
            if not data:
                break
            hasher.update(data)

    return hasher.hexdigest()


# PARSE
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
        self.nlp = spacy.load("de_core_news_md")
        self.file_path: str = file_path
        self.file_lang: str = file_lang
        self.verbose: bool = verbose

        # Additional variables necessary
        self.content: list = list()
        self.csc_initialized: bool = False
        self.document_nlp = None
        self.document_plain = None
        self.location = []
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
    def preprocess(image):
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_gray
        image = cv2.dilate(image, np.ones((7, 7), np.uint8))
        img_bg = cv2.medianBlur(image, 21)
        image = 255 - cv2.absdiff(img_gray, img_bg)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 18)

        return image

    def perform_ocr(self):
        if self.state == State.OCR_READY:
            for page_number, page_data in enumerate(self.document_plain):
                # Reset states
                if self.state == State.NLP_CSC_DONE:
                    self.state = State.OCR_READY
                    if self.verbose:
                        print(f"[i] State changed: {self.state}")

                # Apply OpenCV preprocessing
                page_data = self.preprocess(np.array(page_data))

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
                    print("[!] No text found in page.")
                    raise OcrFailedOrNoTextFoundException

                # Perform NLP and CSC
                self.perform_nlp(page_number)

            self.state = State.FINISHED
            if self.verbose:
                print(f"[i] State changed: {self.state}")

    def perform_nlp(self, page_number):
        if self.state == State.OCR_DONE_CSC_NOT_READY:
            # Add contextualSpellChecker to pipeline
            global CSC_INITIALIZED
            if not CSC_INITIALIZED:
                # csc.add_to_pipe(self.nlp)
                CSC_INITIALIZED = True
            self.csc_initialized = True
            self.state = State.NLP_CSC_READY
            if self.verbose:
                print(f"[i] State changed: {self.state}")

        # Perform NLP
        if self.state == State.NLP_CSC_READY:
            self.document_nlp = self.nlp(self.text_ocr)
        else:
            raise NlpNotReadyException

        if self.document_nlp._.performed_spellCheck:
            page = dict()
            page["page_number"] = page_number + 1
            page["page_content"] = f"{self.document_nlp._.outcome_spellCheck}".strip()

            if page_number + 1 == 1:
                self.location = get_location_info(self.document_nlp._.outcome_spellCheck)

            self.content.append(page)

            if self.verbose:
                print(f"[*] Page: {page['page_number']}: {page['page_content']}")

            self.state = State.NLP_CSC_DONE
            if self.verbose:
                print(f"[i] State changed: {self.state}")
        else:
            raise CscException


def parse_file(filename) -> tuple:
    pages: list = list()
    locations: set = set()

    environ["TOKENIZERS_PARALLELISM"] = "false"

    ctx = OCRContext(file_path=filename, verbose=False)
    ctx.prepare_file()
    ctx.perform_ocr()

    pages = ctx.content
    locations = ctx.location

    return pages, locations


# Parse CSV file containing german zip codes
def parse_plz_csv_file() -> np.ndarray:
    locations_csv = pd.read_csv('../resources/plz_de_east.csv', sep=',', encoding='iso-8859-1')
    locations = np.array(pd.Series(locations_csv['Ort'].values).drop_duplicates())
    return locations


# LOCATION
def get_location_info(text) -> set:
    """https://www.geeksforgeeks.org/extracting-locations-from-text-using-python/"""
    possible_locations: np.ndarray = parse_plz_csv_file()
    confirmed_locations: set = set()

    nlp = spacy.load("de_core_news_md")
    doc = nlp(text)

    for token in doc:
        for location in possible_locations:
            if token.text == location:
                confirmed_locations.add(location)

    print(f"[+] Found the following locations: {confirmed_locations}")

    return confirmed_locations


def get_coords_from_location(location: str) -> tuple:
    coords = geolocator.geocode(location)
    if coords is not None:
        return coords.latitude, coords.longitude
    return 51.0375142, 13.7329536
