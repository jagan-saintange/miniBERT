import chardet
import os
os.chdir(r"/home/silver/Oscar")

with open("corpus_utf8.txt", "rb") as f:
    for i, chunk in enumerate(iter(lambda: f.read(200000), b"")):
        result = chardet.detect(chunk)
        print(i, result)
