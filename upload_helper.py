"""
Uploads wildfire simulation files to Kaggle in batches of 20.
Saves progress to upload_progress.json so you can stop and resume anytime.
"""
import json
import os
import kagglehub
from pathlib import Path

USERNAME  = "muhammadalihashar"
BATCH_SIZE = 20
PROGRESS_FILE = Path("upload_progress.json")

# Load progress (which parts are already done)
if PROGRESS_FILE.exists():
    done_parts = set(json.loads(PROGRESS_FILE.read_text()))
    print(f"Resuming — {len(done_parts)} parts already uploaded: {sorted(done_parts)}")
else:
    done_parts = set()

# Recover any files stuck in staging from a previous interrupted run
staging = Path("upload_staging")
staging.mkdir(exist_ok=True)
stuck = list(staging.glob("*.pt"))
if stuck:
    print(f"Recovering {len(stuck)} stuck files from previous run...")
    for f in stuck:
        os.rename(f, Path("dataset/simulations") / f.name)

sim_files = sorted(Path("dataset/simulations").glob("*.pt"))
print(f"Found {len(sim_files)} simulation files")

shared = {}
for name in ["normalization.json"]:
    p = Path("dataset") / name
    if p.exists():
        shared[name] = p.read_bytes()

batches = [sim_files[i:i + BATCH_SIZE] for i in range(0, len(sim_files), BATCH_SIZE)]
remaining = [bi for bi in range(len(batches)) if (bi + 1) not in done_parts]
print(f"{len(remaining)} parts remaining out of {len(batches)} total\n")

for bi in remaining:
    batch    = batches[bi]
    part_num = bi + 1
    handle   = f"{USERNAME}/wildfire-drone-p{part_num}"

    meta = {
        "title": f"wildfire-drone-p{part_num}",
        "id": handle,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(staging / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    for name, data in shared.items():
        (staging / name).write_bytes(data)

    for fp in batch:
        os.rename(fp, staging / fp.name)

    print(f"Uploading part {part_num}/{len(batches)}: {batch[0].name} … {batch[-1].name}")
    try:
        kagglehub.dataset_upload(handle, str(staging))
        done_parts.add(part_num)
        PROGRESS_FILE.write_text(json.dumps(sorted(done_parts)))
        print(f"  Part {part_num} done. Progress saved.\n")
    except Exception as e:
        print(f"  ERROR on part {part_num}: {e}")
        print("  Moving files back. Run script again to resume.")
    finally:
        for fp in batch:
            dest = Path("dataset/simulations") / fp.name
            src  = staging / fp.name
            if src.exists():
                os.rename(src, dest)

print("Done for now. Run again to continue where you left off.")
