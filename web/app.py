#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from core import *
from datetime import datetime
from elasticsearch import Elasticsearch
from flask import Flask, jsonify, render_template, request, redirect
from importlib.metadata import version
from os import getenv, path, listdir
from shutil import copyfile
from werkzeug.utils import secure_filename

# Flask config
app = Flask(__name__)
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SYNC_DIR"] = getenv("SYNC_DIR")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GOOGLEMAPS_KEY"] = getenv("GOOGLE_API_KEY")

# Maps config
gmaps = googlemaps.Client(key=getenv("GOOGLE_API_KEY"))

# Parameters: pages: list, location: list, filename: str
CURRENT_DOC = dict()

# Parameters: *dict(lat: float, lng: float, infobox: str)
MARKERS = list()
MARKERS_COMPLETE = list()


def get_id():
    hits = client.search(index=ELASTIC_INDEX, query={"match_all": {}})
    next_id = len(hits["hits"]["hits"]) + 1
    return next_id


@app.route('/')
def index():
    return render_template("query_database.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "GET":
        return render_template("classify.html")

    if request.method == "POST":
        # Check if there is a file part in the request
        if "file" in request.files:
            file = request.files["file"]

            # If user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                return render_template("classify.html", error="No file selected for uploading")

            # Check if the file is one of the allowed types/extensions
            if file and is_valid_file(file.filename):
                # Handle filename (anti-XSS)
                filename = secure_filename(file.filename)

                # Construct path to file
                file_path = path.join(app.config["UPLOAD_FOLDER"], filename)

                # Save file to upload folder temporarily
                file.save(file_path)

                pages, location_info = parse_file(file_path)

                CURRENT_DOC["pages"] = pages
                CURRENT_DOC["filename"] = filename
                CURRENT_DOC["location"] = list()

                if len(location_info) == 0:
                    CURRENT_DOC["location"].append((51.0375142, 13.7329536))

                for location in location_info:
                    CURRENT_DOC["location"].append(get_coords_from_location(location))

                return render_template("classified_text.html", pages=pages, locations=location_info, filename=file_path)

            return redirect(request.url)

        if request.form.get("file_name") != "":
            file = request.form.get("file_name")

            # Check if the file is one of the allowed types/extensions
            if file and is_valid_file(file):
                fname = file.split('/')[-1]
                copyfile(file, f"{UPLOAD_FOLDER}/{fname}")

                pages, location_info = parse_file(file)

                CURRENT_DOC["pages"] = pages
                CURRENT_DOC["filename"] = fname
                CURRENT_DOC["location"] = list()

                if len(location_info) == 0:
                    CURRENT_DOC["location"].append((51.0375142, 13.7329536))

                for location in location_info:
                    CURRENT_DOC["location"].append(get_coords_from_location(location))

                return render_template("classified_text.html", pages=pages, locations=location_info, filename=f"{SYNC_DIR}/{fname}")

        return render_template("classify.html", error="No file part")


def write_document_to_db(current_file_path, document):
    doc = {
        "document_pdf": [{
            "file_name": document["filename"],  # from classification form
            "pages": len(document["pages"]),
            "raw_file": current_file_path,
            "signature": get_document_hash(current_file_path)
        }],
        "document_metadata": [{
            "location": document["location"],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        }],
        "document_ocr": document["pages"]
    }

    doc_id = get_id()

    resp = client.index(index=ELASTIC_INDEX, id=doc_id, document=doc)
    if not resp.meta.status == 200:
        print(f"[!] Error (Status code {resp.meta.status})")

    client.indices.refresh(index=ELASTIC_INDEX)


@app.route('/push', methods=["GET", "POST"])
def push():
    status = "post success"

    if request.method == "GET":
        info = client.info()
        status = "get success (" + info["version"]["number"] + ')'

    if request.method == 'POST':
        current_file_path = f"{getenv('SYNC_DIR')}/{CURRENT_DOC['filename']}"
        write_document_to_db(current_file_path, CURRENT_DOC)
        return redirect("/sync")

    return jsonify({"status": status})


@app.route("/query", methods=["POST"])
def query():
    """https://elasticsearch-py.readthedocs.io/en/v8.4.0/"""
    query = request.form.get("input-query")
    results = list()
    location = list()

    if not query:
        return redirect('/')

    # TODO: Fine tune query
    hits = client.search(index=ELASTIC_INDEX, query={"match_all": {}})

    for hit in hits['hits']['hits']:
        for data in hit["_source"]["document_pdf"]:
            document = data["file_name"]

        for data in hit["_source"]["document_metadata"]:
            metadata = data

        for data in hit["_source"]["document_ocr"]:
            if data["page_number"] == 1:
                preview = data["page_content"][0:300]

        results.append({
            "document": document,
            "pages": preview,
            "metadata": metadata
        })

    results_count = len(results)

    if results_count > 0:
        MARKERS.clear()

        # Extract Locations
        for result in results:
            name = result["document"]
            loc = np.array(result["metadata"]["location"])

            if loc.shape == (2,):
                MARKERS.append({"lat": loc[0], "lng": loc[1], "infobox": name})
                location.append(loc[0])
                location.append(loc[1])
            else:
                MARKERS.append({"lat": loc[0][0], "lng": loc[0][1], "infobox": name})
                location.append(loc[0][0])
                location.append(loc[0][1])

    return render_template("query_database.html", raw=False, results=results, query_string=query,
                           results_count=results_count, location=location)


@app.route("/map", methods=["GET"])
def map_view():
    if len(MARKERS) > 0:
        return render_template("map_view.html", markers=MARKERS, GOOGLEMAPS_KEY=getenv("GOOGLE_API_KEY"))

    # Get all markers from Database
    results = list()

    hits = client.search(index=ELASTIC_INDEX, query={"match_all": {}})

    for hit in hits['hits']['hits']:
        for data in hit["_source"]["document_pdf"]:
            document = data["file_name"]

        for data in hit["_source"]["document_metadata"]:
            metadata = data

        results.append({
            "document": document,
            "metadata": metadata
        })

    if len(results) > 0:
        MARKERS_COMPLETE.clear()

        # Extract Locations
        for result in results:
            name = result["document"]
            loc = np.array(result["metadata"]["location"])

            if loc.shape == (2,):
                MARKERS_COMPLETE.append({"lat": loc[0], "lng": loc[1], "infobox": name})
            else:
                MARKERS_COMPLETE.append({"lat": loc[0][0], "lng": loc[0][1], "infobox": name})

    return render_template("map_view.html", markers=MARKERS_COMPLETE, GOOGLEMAPS_KEY=getenv("GOOGLE_API_KEY"))


@app.route("/vis")
def vis():
    # TODO: add visualizations
    # Plotly Mapbox Density Heatmap
    """https://plotly.com/python/mapbox-density-heatmaps/"""
    return render_template("visualization.html")


@app.route('/pdf', methods=["POST"])
def pdf():
    url = DOWNLOAD_FOLDER + '/' + request.form.get("file")
    copyfile(f"{getenv('SYNC_DIR')}/{request.form.get('file')}", url)

    return render_template("pdf_render.html", filename=request.form.get("file"))


@app.route("/sync", methods=["GET", "POST"])
def sync():
    db_hash_list = list()
    hits = client.search(index=ELASTIC_INDEX, query={"match_all": {}})

    for hit in hits["hits"]["hits"]:
        for data in hit["_source"]["document_pdf"]:
            signature = data["signature"]

        db_hash_list.append(signature)

    hash_list = list()

    for file in listdir(getenv("SYNC_DIR")):
        if file.split('.')[-1] != "pdf":
            continue

        doc_hash = get_document_hash(f"{getenv('SYNC_DIR')}/{file}")

        # Check if File is already in DB
        if doc_hash not in db_hash_list:
            hash_list.append({"name": file, "signature": doc_hash})

    if request.method == "POST":
        for file in hash_list:
            app.logger.info(f"Sync: Processing file {file['name']} ({file['signature']})")

            file_path = f"{getenv('SYNC_DIR')}/{file['name']}"

            pages, location_info = parse_file(file_path)

            CURRENT_DOC["pages"] = pages
            CURRENT_DOC["filename"] = file["name"]
            CURRENT_DOC["location"] = list()

            if len(location_info) == 0:
                CURRENT_DOC["location"].append((51.0375142, 13.7329536))

            for location in location_info:
                coords = get_coords_from_location(location)
                CURRENT_DOC["location"].append(coords)

            write_document_to_db(file_path, CURRENT_DOC)

        return redirect("/sync")

    return render_template("sync.html", local_sync_dir=getenv("SYNC_DIR"), sync_dir=getenv("SYNC_DIR_LOCAL"),
                           files=hash_list, new_files_count=len(hash_list), dbhl=db_hash_list)


@app.route("/info")
def info():
    packages = list()

    with open("../requirements.txt", 'r') as fp:
        for line in fp.readlines():
            package = line.strip().split("==")[0]
            packages.append({"name": package, "version": version(package)})

    app_version = getenv("VERSION")

    return render_template('version_info.html', packages=packages, app_version=app_version)


@app.route("/manage")
def manage():
    return render_template("manage.html")


@app.route("/cleardb")
def cleardb():
    hits = client.search(index=ELASTIC_INDEX, query={"match_all": {}})["hits"]["hits"]
    hit_count = len(hits)

    ids = sorted([int(hit["_id"]) for hit in hits])

    app.logger.info(f"Database contains {hit_count} documents ({ids})")

    for idx in ids:
        if idx == 1:
            continue
        client.delete(index=ELASTIC_INDEX, id=str(idx))
        app.logger.info(f"Deleted /{ELASTIC_INDEX}/_doc/{idx}")

    return render_template("manage.html", message="Database has been cleared.")


if __name__ == "__main__":
    # Setup Elasticsearch Client
    client = Elasticsearch(ELASTIC_HOST, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD), ca_certs=CERT_FILE)

    # Create index if it doesn't exist
    # response = client.options(ignore_status=[400]).indices.create(index=ELASTIC_INDEX)

    # if response.meta.status == 400:
    #    client.options(ignore_status=[400]).indices.put_mapping(index=ELASTIC_INDEX, body=ELASTIC_MAPPING)

    # Setup App
    app.debug = DEBUG
    ssl_context: tuple = ("cert.pem", "key.pem")
    app.run(host="0.0.0.0", port=int(getenv("FLASK_PORT")), ssl_context=ssl_context)
