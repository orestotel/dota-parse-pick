import netron
import torch
import argparse
from dgl2 import FNNModel
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
import os
import pickle
from tqdm import tqdm
import sys
import threading
import time
import glob
import combine_files
import training_module

def visualize_model(model_path):
    model = torch.load(model_path)
    input_size = model.fc1.in_features
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=False)
    netron.start("model.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PyTorch model using Netron")
    parser.add_argument("model_path", help="Path to the saved PyTorch model (.pt or .pth)")
    args = parser.parse_args()

    visualize_model(args.model_path)



#  python netron_visualization.py path/to/your/model.pt
