import json
import os
import requests
from tqdm import tqdm
import time
import multiprocessing
from multiprocessing import Process

def fetch_and_save_matches(api_key, n_matches, min_mmr, output_file, n_processes, process_id):
    starting_match_offset = int(n_matches / n_processes) * process_id
    # rest of the function
    matches_url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}&mmr_ascending={min_mmr}&offset={starting_match_offset}"

    # ... (rest of the function)


    heroes_url = f"https://api.opendota.com/api/heroes?api_key={api_key}"
    heroes_data = requests.get(heroes_url).json()
    id_to_hero = {hero["id"]: hero["localized_name"] for hero in heroes_data}

    matches = []

    def save_matches_to_file(output_file, matches):
        with open(output_file, "w") as f:
            json.dump(matches, f, indent=4)

    with tqdm(total=n_matches, desc=f"Fetching matches (Process {process_id})") as pbar:
        while len(matches) < n_matches:
            try:
                response = requests.get(matches_url)
                new_matches = response.json()
            except Exception as e:
                print(f"Error occurred while fetching matches (Process {process_id}): {str(e)}")
                time.sleep(1)
                continue

            for match in new_matches:
                if len(matches) >= n_matches:
                    break

                try:
                    radiant_team = [id_to_hero[int(hero_id)] for hero_id in match["radiant_team"].split(",")]
                    dire_team = [id_to_hero[int(hero_id)] for hero_id in match["dire_team"].split(",")]
                except Exception as e:
                    print(f"Error occurred while processing match {match['match_id']} (Process {process_id}): {str(e)}")
                    continue

                matches.append({
                    "match_id": match["match_id"],
                    "radiant_win": match["radiant_win"],
                    "radiant_team": radiant_team,
                    "dire_team": dire_team
                })

                pbar.update(1)

            matches_url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}&mmr_ascending={min_mmr}&sort=start_time&order=desc"

            time.sleep(0.01)

            try:
                save_matches_to_file(output_file, matches)
            except Exception as e:
                print(f"Error occurred while saving matches to file (Process {process_id}): {str(e)}")
                continue

    save_matches_to_file(output_file, matches)

def fetch_parallel(api_key, n_matches, min_mmr, output_file, n_processes):
    processes = []
    for i in range(n_processes):
        process_output_file = f"{output_file[:-5]}_process_{i}.json"
        process = multiprocessing.Process(target=fetch_and_save_matches, args=(api_key, n_matches, min_mmr, process_output_file, n_processes, i))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def combine_files(output_file, process_files):
    combined_matches = []

    for process_file in process_files:
        with open(process_file, "r") as f:
            matches = json.load(f)
            combined_matches.extend(matches)

    with open(output_file, "w") as f:
        json.dump(combined_matches, f, indent=4)

if __name__ == "__main__":
    api_key = "2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75"  # Replace with your OpenDota API key
    n_matches = 2000000000 // 1
    min_mmr = 20
    output_file = "parse-data/chewtree.json"
    n_processes = 1

    fetch_parallel(api_key, n_matches, min_mmr, output_file, n_processes)
    process_files = [f"{output_file[:-5]}_process_{i}.json" for i in range(n_processes)]
    combine_files("parse-data/combined_chewtree.json", process_files)
