from pathlib import Path

root_directory = Path('.')
sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())