import json
import torch
from torch_geometric.data import Data
from collections import defaultdict

def load_heroes():
    # Load hero names and indices from an external source or define them manually
    heroes = ["Lich", "Phantom Assassin", "Arc Warden", ...] # list
    hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)} # dict
    return hero_to_idx # dict

def recursionlimit():
    import sys
    sys.setrecursionlimit(10000)


def parse_data(json_file): # json file
    with open(json_file, 'r') as f: # open json file
        json_file = 'parse-data/bubblegum1.json' # json file
        dataset = parse_data(json_file) # list of dicts




    hero_to_idx = load_heroes()
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
                    edge_index.append((src_idx, dst_idx))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(nodes, dtype=torch.long).view(-1, 1)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list




if __name__ == "__main__":
    json_file = 'parse-data/bubblegum1.json'
    data_list = parse_data(json_file)
    print(data_list)
