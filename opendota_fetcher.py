import requests
import json
from tqdm import tqdm
import time


def fetch_and_save_matches(api_key, n_matches, min_mmr, output_file):
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

                radiant_team = [id_to_hero[int(hero_id)] for hero_id in match["radiant_team"].split(",")]
                dire_team = [id_to_hero[int(hero_id)] for hero_id in match["dire_team"].split(",")]

                matches.append({
                    "match_id": match["match_id"],
                    "radiant_win": match["radiant_win"],
                    "radiant_team": radiant_team,
                    "dire_team": dire_team
                })

                pbar.update(1)

            matches_url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}&less_than_match_id={matches[-1]['match_id']}"
            time.sleep(1)

    save_matches_to_file(output_file, matches)


if __name__ == "__main__":
    api_key = "2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75"  # Replace with your OpenDota API key
    n_matches = 170000
    min_mmr = 600
    output_file = "parse-data/dataset2.json"

    fetch_and_save_matches(api_key, n_matches, min_mmr, output_file)
