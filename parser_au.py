import requests
import time
import tqdm
import json

api_key = 'your_opendota_api_key_here'
count = 1000  # Increase this value to fetch more matches

def get_public_matches():
    url = f"https://api.opendota.com/api/publicMatches?api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

matches = []
for _ in tqdm.tqdm(range(count // 100)):
    batch = get_public_matches()
    matches.extend(batch)
    time.sleep(1)  # To avoid rate limits

# Save the matches data to a JSON file
with open('D:/sofardata/batchira.json', 'w') as f:
    json.dump(matches, f)
