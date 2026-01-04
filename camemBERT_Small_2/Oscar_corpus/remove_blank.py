import os
os.chdir(r"/home/silver/Oscar")
def remove_null_bytes(input_file, output_file):
    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        for chunk in fin:
            fout.write(chunk.replace(b"\x00", b""))

remove_null_bytes("corpus_utf8.txt", "corpus_utf8_nonull.txt")
