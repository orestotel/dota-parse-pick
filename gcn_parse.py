import requests
import torch
from torch_geometric.data import Data
import json

def load_heroes():
    with open("parse-data/heroes.json") as f:
        heroes = json.load(f)
    return heroes


def fetch_heroes():
    # Fetch the list of all heroes and their IDs from the OpenDota API
    response = requests.get('https://api.opendota.com/api/heroes')
    heroes_data = response.json()

    # Create a dictionary that maps hero names to indices
    hero_to_idx = {}
    for idx, hero in enumerate(heroes_data, start=1):
        hero_name = hero['localized_name']
        hero_to_idx[hero_name] = idx

    return hero_to_idx


def parse_data(json_file):
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    hero_to_idx = fetch_heroes()
    data_list = []

    for match in dataset:
        nodes = []
        edges = []
        edge_index = []
        y = torch.tensor([1.0 if match['radiant_win'] else 0.0], dtype=torch.float)

        for team, heroes in zip(['radiant_team', 'dire_team'], [match['radiant_team'], match['dire_team']]):
            for hero in heroes:
                idx = hero_to_idx[hero]
                nodes.append(idx)
                edges.append((team, idx))

        for i, (src_team, src_idx) in enumerate(edges):
            for j, (dst_team, dst_idx) in enumerate(edges):
                if i != j and src_team == dst_team:
                    edge_index.append((src_idx - 1, dst_idx - 1))  # Subtract 1 to adjust for 0-based indexing

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(nodes, dtype=torch.long).view(-1, 1)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list
