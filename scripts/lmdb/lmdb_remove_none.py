import lmdb
import pickle
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Check LMDB for None values in a single-modality dataset")
    parser.add_argument("lmdb_path", help="Path to the LMDB database")
    parser.add_argument("--modality", required=True, help="Expected modality name (e.g., LULC_LULC)")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional: limit number of entries to scan")
    
    args = parser.parse_args()
    lmdb_path = args.lmdb_path
    modality = args.modality

    env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=512)

    total = 0
    bad = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in tqdm(enumerate(cursor)):
            try:
                obj = pickle.loads(value)
                # print(obj)

                if modality not in obj:
                    print(f"❌ Key {key.decode()} missing modality {modality}")
                    bad += 1
                elif obj[modality] is None:
                    print(f"❌ Key {key.decode()} has None value for modality {modality}")
                    bad += 1

            except Exception as e:
                print(f"❌ Failed to decode key {key.decode()}: {e}")
                bad += 1

            total += 1
            if args.max_entries is not None and total >= args.max_entries:
                break

    print(f"\nScanned {total} entries")
    print(f"Entries with issues: {bad}")
    if bad == 0:
        print("✅ All entries contain non-None values for the given modality")


if __name__ == "__main__":
    main()
