from datasets import load_dataset
import io

TARGET_SIZE = 4 * 1024 * 1024 * 1024   # 4 Go
OUTPUT_FILE = "oscar_4GB.txt"

def clean_text(text):
    if not text:
        return None
    text = text.replace("\n", " ").strip()
    return " ".join(text.split())

def main():
    print("Chargement OSCAR FR en streaming + shuffle...")
    ds = load_dataset(
        "oscar", 
        "unshuffled_deduplicated_fr", 
        split="train", 
        streaming=True
    ).shuffle(buffer_size=100_000)

    written_bytes = 0
    count = 0

    with io.open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in ds:
            text = clean_text(sample.get("text"))
            if not text:
                continue

            line = text + "\n"
            size = len(line.encode("utf-8"))

            if written_bytes + size > TARGET_SIZE:
                print("Objectif atteint : 4 Go écrits.")
                break

            f.write(line)
            written_bytes += size
            count += 1

            if count % 10000 == 0:
                print(f"{written_bytes/1e9:.2f} Go écrits...")

    print(f"Extraction terminée : {written_bytes/1e9:.2f} Go")
    print(f"Fichier généré : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

