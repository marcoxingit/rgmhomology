
# download_arc_tasks.py
# Download ARC tasks from GitHub (ARC-AGI-1 or ARC-AGI-2).
# Usage examples:
#   python download_arc_tasks.py --version 1 --split training --dest ./arc_data --sample 50
#   python download_arc_tasks.py --version 2 --split all --dest ./arc2_data
#
import argparse, os, sys, time, random, json
from typing import List, Dict
import requests

GH_API = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"

DATASETS = {
    1: {"owner": "fchollet", "repo": "ARC-AGI"},
    2: {"owner": "arcprize", "repo": "ARC-AGI-2"},
}

def list_json_files(owner: str, repo: str, subpath: str) -> List[Dict]:
    url = GH_API.format(owner=owner, repo=repo, path=subpath)
    r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"}, timeout=30)
    r.raise_for_status()
    items = r.json()
    return [it for it in items if it.get("type")=="file" and it.get("name","").endswith(".json")]

def download(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, choices=[1,2], default=1, help="ARC-AGI version")
    ap.add_argument("--split", type=str, choices=["training","evaluation","all"], default="training")
    ap.add_argument("--dest", type=str, required=True, help="Destination directory")
    ap.add_argument("--sample", type=int, default=0, help="If >0, randomly sample this many tasks per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = DATASETS[args.version]
    splits = ["training","evaluation"] if args.split=="all" else [args.split]
    random.seed(args.seed)

    for sp in splits:
        subpath = f"data/{sp}"
        print(f"Listing {ds['owner']}/{ds['repo']}:{subpath} ...")
        files = list_json_files(ds["owner"], ds["repo"], subpath)
        print(f"Found {len(files)} files in {sp}")
        if args.sample and args.sample < len(files):
            files = random.sample(files, args.sample)
            print(f"Sampling {len(files)} files")
        for i, it in enumerate(files, 1):
            out_path = os.path.join(args.dest, f"v{args.version}", sp, it["name"])
            if os.path.exists(out_path):
                continue
            print(f"[{i}/{len(files)}] {it['name']}")
            download(it["download_url"], out_path)
    print("Done.")

if __name__ == "__main__":
    main()
