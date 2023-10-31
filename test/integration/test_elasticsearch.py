from elasticsearch import Elasticsearch
from os import getenv

ELASTIC_HOST = 'https://localhost:9200'
ELASTIC_INDEX = 'ocr'
ELASTIC_USER = "elastic"
ELASTIC_PASSWORD = getenv('ELASTIC_PASSWORD')
CERT_FILE = '/data/certs/ca/ca.crt'


def main():
    try:
        client = Elasticsearch(ELASTIC_HOST, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD), verify_certs=True, ca_certs=CERT_FILE)
        info = client.info()
        print("[+] Passed (Elasticsearch version: {})".format(info['version']['number']))
    except Exception as e:
        print("[-] Failed ({})".format(e))


if __name__ == '__main__':
    main()
