#from datasets import load_dataset
#load_dataset("xnli", "fr", cache_dir="data/xnli")

# import urllib.request, tarfile

# urls = {
#     "GSD": "https://github.com/UniversalDependencies/UD_French-GSD/archive/r2.2.tar.gz",
#     "Sequoia": "https://github.com/UniversalDependencies/UD_French-Sequoia/archive/r2.2.tar.gz",
#     "Spoken": "https://github.com/UniversalDependencies/UD_French-Spoken/archive/r2.2.tar.gz",
#     "ParTUT": "https://github.com/UniversalDependencies/UD_French-ParTUT/archive/r2.2.tar.gz",
# }

# for name, url in urls.items():
#     tar_path = f"{name}.tar.gz"
#     urllib.request.urlretrieve(url, tar_path)
#     with tarfile.open(tar_path, "r:gz") as tar:
#         tar.extractall("data/ud")
