import os

input_file = "Oscar/Raw/corpus_4go.txt"
output_dir = "Oscar/shards"
os.makedirs(output_dir, exist_ok=True)

lines_per_shard = 100_000
shard_id = 0
buffer = []

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        buffer.append(line)
        if len(buffer) == lines_per_shard:
            with open(f"{output_dir}/shard_{shard_id:03d}.txt", "w", encoding="utf-8") as out:
                out.writelines(buffer)
            buffer = []
            shard_id += 1

if buffer:
    with open(f"{output_dir}/shard_{shard_id:03d}.txt", "w", encoding="utf-8") as out:
        out.writelines(buffer)
