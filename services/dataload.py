'''
File for download dataset
'''

import zipfile as zf

import requests

# Download dataset and unzip it
url = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"

req = requests.get(url, stream=True)

with open("kion.zip", "wb") as fd:
    total_size_in_bytes = int(req.headers.get("Content-Length", 0))
    for chunk in req.iter_content(chunk_size=2**20):
        fd.write(chunk)

files = zf.ZipFile("kion.zip", "r")
files.extractall()
files.close()
