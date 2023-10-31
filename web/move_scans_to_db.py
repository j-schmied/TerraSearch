from core import CERT_FILE, ELASTIC_HOST, ELASTIC_INDEX, get_coords_from_location, get_document_hash, get_id, parse_file
from datetime import datetime
from elasticsearch import Elasticsearch
from os import listdir, getenv

ELASTIC_PASSWORD = getenv("ELASTIC_PASSWORD")
ELASTIC_USER = getenv("ELASTIC_USER")
SYNC_DIR = getenv("SYNCDIR")


def main():
    client = Elasticsearch(ELASTIC_HOST, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD), ca_certs=CERT_FILE)

    files = len(listdir(SYNC_DIR))

    for i, file in enumerate(listdir(SYNC_DIR)):
        if file.split('.')[-1] != "pdf":
            continue

        print(f"[i] Processing Document {i}/{files}", end='\r')

        pages, location_info = parse_file(file)

        doc = {
            "document_pdf": [{
                "file_name": file,  # from classification form
                "pages": len(pages),
                "raw_file": file,
                "signature": get_document_hash(file)
            }],
            "document_metadata": [{
                "location": get_coords_from_location(location_info),
                "timestamp": datetime.now()
            }],
            "document_ocr": pages
        }

        resp = client.index(index=ELASTIC_INDEX, id=get_id(), document=doc)
        if not resp.meta.status == 200:
            print(f"[!] Error while processing document {i} (Status code {resp.meta.status})")

        client.indices.refresh(index=ELASTIC_INDEX)


if __name__ == "__main__":
    main()
