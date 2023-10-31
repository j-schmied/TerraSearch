FROM debian:bullseye

# Update system and install python and tesseract
RUN apt-get --fix-missing update && apt-get --fix-broken install && apt-get install -y poppler-utils && apt-get install -y tesseract-ocr && \
        apt-get install -y libtesseract-dev && apt-get install -y libleptonica-dev && ldconfig && apt-get install -y python3 && \
        apt-get install -y python3-pip && apt-get install -y libsm6 libxext6 && apt-get install -y openssl && apt-get install -y libssl-dev && \
        apt-get install -y libopencv-dev

# Install pytorch and other requirements
WORKDIR /data
RUN mkdir files
COPY requirements.txt .
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download de_core_news_md

# Copy language files
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
COPY resources/training/deu.traineddata $TESSDATA_PREFIX
COPY resources/training/deu_frak.traineddata $TESSDATA_PREFIX

# Setup Flask environment
WORKDIR /data/web
RUN openssl req -x509 -newkey rsa:4096 -nodes -subj "/C=DE/ST=Dresden/L=Saxony/O=HTWDD/OU=IM /CN=terrasearch.htw-dresden.de" -out cert.pem -keyout key.pem -days 365
COPY web .

WORKDIR /data/resources
COPY resources/plz_de_east.csv .

# Expose port for flask webserver
EXPOSE 5000

WORKDIR /data/web
CMD ["python3", "/data/web/app.py"]