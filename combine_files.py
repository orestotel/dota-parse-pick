import os
import json
from collections import defaultdict

def combine_json_files(input_folder, output_folder):
    output_filename = os.path.join(output_folder, "combined_data.json")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = []
    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            with open(os.path.join(input_folder, file), "r") as f:
                file_data = json.load(f)
                data.extend(file_data)

    seen = set()
    unique_data = []
    for match in data:
        match_id = match["match_id"]
        if match_id not in seen:
            seen.add(match_id)
            unique_data.append(match)

    with open(output_filename, "w") as outfile:
        json.dump(unique_data, outfile, indent=4)

    return unique_data

def print_duplicate_counts(input_folder):
    duplicate_counts = defaultdict(int)

    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            with open(os.path.join(input_folder, file), "r") as f:
                file_data = json.load(f)
                file_matches = set(match["match_id"] for match in file_data)
                duplicate_counts[file] += len(file_data) - len(file_matches)

    for file, count in duplicate_counts.items():
        print(f"{file}: {count} duplicates")

if __name__ == "__main__":
    input_folder = "parse-data"
    output_folder = os.path.join(input_folder, "batches")

    combine_json_files(input_folder, output_folder)
    print_duplicate_counts(input_folder)
