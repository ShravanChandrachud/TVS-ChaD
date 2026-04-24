"""
Pexels Image Scraper for Cooking State Dataset
================================================
Scrapes images from the Pexels API for specific cooking object-state classes
that don't have dedicated datasets available (whisked egg, burned egg,
pan with food, plate with food, bowl with whisked egg).

Each class has multiple diverse search queries to maximize variety.
Images are downloaded, deduplicated by content hash, resized to 512x512,
and saved to data/raw/<object>/<state>/.

The script is fully resumable — if interrupted, it detects existing images
and continues from where it left off. If a folder already has images,
it prompts whether to replace or append.

Requires PEXELS_API_KEY in root/.env file.
Free tier: 200 requests/hour, 20,000/month.

Input:  Pexels API
Output: data/raw/<object>/<state>/*.jpg
"""

import os
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

SCRIPT_DIR_ENV = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR_ENV.parent / ".env")

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TARGET_SIZE = (512, 512)

MAX_PER_QUERY = 200

TARGET_PER_CLASS = 500

REQUEST_DELAY = 2.0

PEXELS_BASE_URL = "https://api.pexels.com/v1"

CLASSES = [
    # ── EGG ──
    (
        "egg",
        "whisked",
        [
            "whisked eggs in bowl",
            "beaten eggs bowl",
            "egg mixture bowl whisk",
            "raw scrambled eggs bowl",
            "egg wash bowl cooking",
            "beating eggs with fork bowl",
            "raw egg batter in bowl",
        ],
    ),
    # ── PAN ──
    (
        "pan",
        "with-food",
        [
            "omelette cooking in frying pan",
            "egg omelette in skillet",
            "making omelette frying pan",
            "eggs cooking in pan",
            "omelette preparation pan stove",
            "frying egg in pan",
            "scrambled eggs frying pan cooking",
        ],
    ),
    # ── PLATE ──
    (
        "plate",
        "with-food",
        [
            "omelette on plate",
            "scrambled eggs plate breakfast",
            "fried egg on plate",
            "egg dish served plate",
            "plated eggs breakfast",
            "sunny side up egg plate",
            "egg breakfast plate overhead",
        ],
    ),
    # ── BOWL ──
    (
        "bowl",
        "with-whisked-egg",
        [
            "beaten eggs glass bowl",
            "whisked egg mixture bowl",
            "egg batter in bowl",
            "bowl of beaten eggs cooking",
            "raw egg mixture mixing bowl",
            "whisking eggs in bowl kitchen",
            "yellow egg mixture in bowl",
        ],
    ),
]


class PexelsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.request_count = 0

    def search_photos(self, query: str, page: int = 1, per_page: int = 80) -> dict:
        per_page = min(per_page, 80)
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "size": "medium",
        }
        resp = self.session.get(f"{PEXELS_BASE_URL}/search", params=params)
        self.request_count += 1

        if resp.status_code == 429:
            print(" Rate limited! Waiting 60 seconds")
            time.sleep(60)
            return self.search_photos(query, page, per_page)

        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return {"photos": [], "total_results": 0}

        return resp.json()

    def download_image(self, url: str) -> bytes | None:
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.content
        except requests.RequestException as e:
            print(f"  Download failed: {e}")
        return None


def process_and_save(image_bytes: bytes, save_path: Path, target_size: tuple) -> bool:
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  Image processing failed: {e}")
        return False


def content_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def scrape_class(
    client: PexelsClient,
    object_name: str,
    state_name: str,
    queries: list[str],
    output_dir: Path,
    target_count: int,
    max_per_query: int,
) -> int:
    class_dir = output_dir / object_name / state_name
    class_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.name for f in class_dir.glob("*.jpg"))
    seen_hashes = set()
    total_saved = len(existing)

    if total_saved > 0:
        print(f"  Found {total_saved} existing images in {class_dir}")
        ans = input(f"  Replace them? [y/N]: ").strip().lower()
        if ans == "y":
            for f in class_dir.glob("*"):
                f.unlink()
            existing.clear()
            total_saved = 0
            print(f"  Cleared folder. Starting fresh.")
        else:
            if total_saved >= target_count:
                print(f"  Already have {total_saved}/{target_count}, skipping.")
                return total_saved
            print(f"   Keeping existing. Will add more to reach {target_count}.")

    for query_idx, query in enumerate(queries):
        if total_saved >= target_count:
            break

        print(f'   Query {query_idx + 1}/{len(queries)}: "{query}"')
        query_downloaded = 0
        page = 1

        while query_downloaded < max_per_query and total_saved < target_count:
            data = client.search_photos(query, page=page, per_page=80)
            photos = data.get("photos", [])
            total_results = data.get("total_results", 0)

            if not photos:
                print(f" No more results (page {page}, total: {total_results})")
                break

            for photo in photos:
                if total_saved >= target_count or query_downloaded >= max_per_query:
                    break

                img_url = (
                    photo.get("src", {}).get("large")
                    or photo.get("src", {}).get("medium")
                    or photo.get("src", {}).get("original")
                )
                if not img_url:
                    continue

                photo_id = photo.get("id", "unknown")
                filename = f"{object_name}_{state_name}_{photo_id}.jpg"

                if filename in existing:
                    continue

                img_bytes = client.download_image(img_url)
                if img_bytes is None:
                    continue

                h = content_hash(img_bytes)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                save_path = class_dir / filename
                if process_and_save(img_bytes, save_path, TARGET_SIZE):
                    total_saved += 1
                    query_downloaded += 1

                    if total_saved % 25 == 0:
                        print(f"      Progress: {total_saved}/{target_count}")

            page += 1
            time.sleep(REQUEST_DELAY)

        print(f"  Got {query_downloaded} from this query (total: {total_saved})")

    return total_saved


def main():
    if not PEXELS_API_KEY:
        print("\nERROR: No Pexels API key found!")
        print()
        print("1. Go to https://www.pexels.com/api/ and sign up (free)")
        print("2. Copy your API key")
        print("3. Create a .env file in the project root with:")
        print("   PEXELS_API_KEY=your-key-here")
        print()
        print(f"   Expected .env location: {PROJECT_ROOT / '.env'}\n")
        sys.exit(1)

    print("\nPexels Image Scraper — Cooking State Dataset")
    print(f"Output directory: {RAW_DATA_DIR}")
    print(f"Target: {TARGET_PER_CLASS} images per class")
    print(f"Image size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Total classes: {len(CLASSES)}\n")

    client = PexelsClient(PEXELS_API_KEY)
    results = {}
    start_time = datetime.now()

    for idx, (obj, state, queries) in enumerate(CLASSES):
        class_label = f"{obj}/{state}"
        print(f"\n[{idx + 1}/{len(CLASSES)}] Scraping: {class_label}")
        print(f"   Queries: {len(queries)}")

        count = scrape_class(
            client=client,
            object_name=obj,
            state_name=state,
            queries=queries,
            output_dir=RAW_DATA_DIR,
            target_count=TARGET_PER_CLASS,
            max_per_query=MAX_PER_QUERY,
        )

        results[class_label] = count
        print(f"   ✓ Final count: {count}/{TARGET_PER_CLASS}")

    elapsed = datetime.now() - start_time
    print("\nSCRAPING COMPLETE")
    print(f"Time elapsed: {elapsed}")
    print(f"Total API requests: {client.request_count}\n")

    print(f"\n{'Class':<30} {'Downloaded':>12} {'Target':>8} {'Status':>8}")
    for class_label, count in results.items():
        status = " 1 " if count >= TARGET_PER_CLASS else f"0 {count}"
        print(f"{class_label:<30} {count:>12} {TARGET_PER_CLASS:>8} {status:>8}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "target_per_class": TARGET_PER_CLASS,
        "image_size": list(TARGET_SIZE),
        "source": "Pexels API",
        "results": results,
        "api_requests_used": client.request_count,
    }
    meta_path = RAW_DATA_DIR / "scrape_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    shortfalls = {k: v for k, v in results.items() if v < TARGET_PER_CLASS}
    if shortfalls:
        print("\n The following classes didn't reach 500 images:")
        print("  Consider using Stable Diffusion or Pixabay to fill gaps:")
        for class_label, count in shortfalls.items():
            deficit = TARGET_PER_CLASS - count
            print(f"  - {class_label}: need {deficit} more images")


if __name__ == "__main__":
    main()
