import requests
import json
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def fetch_and_save_matches(api_key, n_matches, min_mmr, output_file, existing_matches):
    matches_url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}&mmr_ascending={min_mmr}"

    heroes_url = f"https://api.opendota.com/api/heroes?api_key={api_key}"
    heroes_data = requests.get(heroes_url).json()
    id_to_hero = {hero["id"]: hero["localized_name"] for hero in heroes_data}

    matches = []

    def save_matches_to_file(output_file, matches):
        with open(output_file, "w") as f:
            json.dump(matches, f, indent=4)

    error_counter = 0
    max_errors = 120

    with tqdm(total=n_matches, desc="Fetching matches") as pbar:
        while len(matches) < n_matches:
            response = requests.get(matches_url)
            try:
                new_matches = response.json()
            except json.JSONDecodeError:
                error_counter += 1
                if error_counter >= max_errors:
                    print("Reached maximum allowed errors. Saving gathered data and stopping.")
                    save_matches_to_file(output_file, matches)
                    break
                else:
                    print(f"Error decoding JSON response. Error count: {error_counter}")
                    continue

            for match in new_matches:
                if len(matches) >= n_matches:
                    break

                if match["match_id"] in existing_matches:
                    continue

                radiant_team = [id_to_hero[int(hero_id)] for hero_id in match["radiant_team"].split(",")]
                dire_team = [id_to_hero[int(hero_id)] for hero_id in match["dire_team"].split(",")]

                matches.append({
                    "match_id": match["match_id"],
                    "radiant_win": match["radiant_win"],
                    "radiant_team": radiant_team,
                    "dire_team": dire_team
                })

                pbar.update(1)

            matches_url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}&mmr_ascending={min_mmr}&sort=start_time&order=desc"

            time.sleep(0.01)

    save_matches_to_file(output_file, matches)


def load_existing_match_ids(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return {match["match_id"] for match in data}


if __name__ == "__main__":
    api_key = "2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75"  # Replace with your OpenDota API key
    n_matches = 1750000
    min_mmr = 20
    output_file = "parse-data/chewtree.json"
    existing_matches_file = "parse-data/bubblegum1.json"
    existing_matches = load_existing_match_ids(existing_matches_file)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_and_save_matches, api_key, n_matches, min_mmr, output_file, existing_matches)        for _ in range(os.cpu_count())
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in one of the processes: {e}")

    fetch_and_save_matches(api_key, n_matches, min_mmr, output_file, existing_matches)
