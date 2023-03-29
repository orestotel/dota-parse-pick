import os
import json
from collections import defaultdict

def combine_json_files(input_folder, output_folder):
    output_filename = os.path.join(output_folder, "combined_data2.json")

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


def concatenate_json_files(filepaths):
    concatenated_data = []

    for filepath in filepaths:
        with open(filepath, "r") as f:
            file_data = json.load(f)
            concatenated_data.extend(file_data)

    return concatenated_data


def remove_duplicates(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    seen = set()
    unique_data = []
    duplicates_count = 0
    for match in data:
        match_id = match["match_id"]
        if match_id not in seen:
            seen.add(match_id)
            unique_data.append(match)
        else:
            duplicates_count += 1

    with open(filepath, "w") as outfile:
        json.dump(unique_data, outfile, indent=4)

    print(f"{filepath}: {duplicates_count} duplicates removed")


if __name__ == "__main__":
    input_folder = "parse-data"
    output_folder = os.path.join(input_folder, "batches")

    # Use hardcoded filepaths to concatenate JSON files
    hardcoded_filepaths = [
        "parse-data/bigflow.json",
        "parse-data/dataset1.json",
        "parse-data/dataset2.json",
        "parse-data/dataset3pro.json",
        "parse-data/dataset4pro.json",
        "parse-data/dataset5pro.json",
        "parse-data/dataset6basic.json",
        "parse-data/dataset7basic.json",
        "parse-data/new_matches.json",
        "parse-data/newmatches1.json",
        "parse-data/newmatches2.json",
        "parse-data/newmatches3.json",
        "parse-data/newmatches4.json",
        "parse-data/newmatches5.json"
        # Add more file paths here...
    ]
    concatenated_data = concatenate_json_files(hardcoded_filepaths)

    # Write concatenated data to a new file
    output_filepath = os.path.join(output_folder, "bubblegum1.json")
    with open(output_filepath, "w") as outfile:
        json.dump(concatenated_data, outfile, indent=4)

    # do not Check for duplicates in the new file and remove them
