import json
from collections import defaultdict


def count_duplicate_match_ids(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    match_id_counts = defaultdict(int)
    for match in data:
        match_id_counts[match["match_id"]] += 1

    duplicate_match_ids = {match_id: count for match_id, count in match_id_counts.items() if count > 1}

    total_duplicates = sum(count - 1 for count in duplicate_match_ids.values())
    max_duplicate_count = max(duplicate_match_ids.values(), default=0)

    print(f"Total number of duplicates: {total_duplicates}")
    print("\nDuplicate match IDs:")
    for match_id, count in duplicate_match_ids.items():
        print(f"{match_id}: seen {count} times")

    print(f"\nMaximum times one duplicate ID has been duplicated: {max_duplicate_count - 1}")


if __name__ == "__main__":
    filepath = "parse-data/batches/bubblegum1.json"
    count_duplicate_match_ids(filepath)
    from combine_files import remove_duplicates
    remove_duplicates(filepath)
