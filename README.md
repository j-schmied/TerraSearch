# TerraSearch

OCR und Suche in digitalisierten Bodenuntersuchungsdokumenten

## Tech Stack

| Part       | Technology                                  |
|------------|---------------------------------------------|
| Frontend   | Flask (Web), Kibana (DB)                    |
| Backend    | Elasticsearch (Storage, Indexing)           |
| OCR        | Google Tesseract, jTessBoxEditor (Training) |
| NLP        | spaCy, ContextualSpellCheck, LocationTagger |      
| Deployment | Docker/docker-compose (Containerisation)    |

## Setup and Installation

1. Install Dependencies (Docker, docker-compose)
2. Create .env File containing the following values (see example below):
    * ELASTIC_USER
    * ELASTIC_PASSWORD
    * GOOGLE_API_KEY
    * KIBANA_PASSWORD
    * STACK_VERSION
    * LICENSE
    * ES_PORT
    * KIBANA_PORT
    * FLASK_PORT
    * SYNC_DIR
    * MEM_LIMIT
    * CLUSTER_NAME
    * SYNC_DIR
    * SYNC_DIR_LOCAL
3. Create Docker Network for Project `docker network create terrasearch-net`
4. Start Docker Compose `docker-compose up -d`

### Environment File (.env) Example

```
VERSION="0.5.3"

# Username for elasticsearch
ELASTIC_USER="elastic"

# Password for the 'elastic' user (at least 6 characters)
ELASTIC_PASSWORD="elastic123"

# API Key for Google Maps
GOOGLE_API_KEY="GetMeFromGoogleCloudPlatform"

# Password for the 'kibana_system' user (at least 6 characters)
KIBANA_PASSWORD="kibana123"

# Version of Elastic products
STACK_VERSION=8.3.1

# Set to 'basic' or 'trial' to automatically start the 30-day trial
LICENSE=basic

# Port to expose Elasticsearch HTTP API to the host
ES_PORT=9200

# Port to expose Kibana to the host
KIBANA_PORT=5001

# Port to expose Frontend to the host
FLASK_PORT=8443

# Increase or decrease based on the available host memory (in bytes)
MEM_LIMIT=4294967296

# Set the cluster name
CLUSTER_NAME=terrasearch-cluster

# Sync directory inside container
SYNC_DIR="/data/sync"

# Local Dir containing pdf files
SYNC_DIR_LOCAL="C:\Downloads"
```

## Sources

* [jTessBoxEditor](https://github.com/nguyenq/jTessBoxEditor)
* [Poppler4Windows](https://blog.alivate.com.au/poppler-windows/)
* [Tesseract4Windows](https://github.com/UB-Mannheim/tesseract/wiki)
* [spaCy](https://spacy.io/usage)
