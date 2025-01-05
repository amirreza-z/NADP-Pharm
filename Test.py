import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from ModelPMP import PerishablePharmaceuticalModelMultiProduct
from LPSolver import LPSolver
from DriverScript import PolicyNetwork  # Assuming this is saved as a separate file
from DriverScript import PharmaDataset  # Assuming this is saved as a separate file
from DriverScript import load_datasets, compute_stats, plot_rewards, train  # Utility functions

# ------------------------------
# Hyperparameters
# ------------------------------
FOLDER_PATH = "dataMP"
IS_CUDA = True
PRODUCT_NAMES = ["Product1", "Product2", "Product3", "Product4"]
INPUT_SIZE = 4  # Number of state variables per product
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPISODES = 10
INITIAL_ENTROPY_BETA = 0.005
MODEL_SAVE_PATH = "policy_net.pth"

# ------------------------------
# Helper Function: Print Results
# ------------------------------
def print_summary(title, data):
    print(f"\n{'=' * 10} {title} {'=' * 10}")
    for key, value in data.items():
        print(f"{key}: {value}")

# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    # 1. Load Data
    print("Loading datasets...")
    datasets = load_datasets(FOLDER_PATH)
    train_datasets = datasets[:100]
    eval_datasets = datasets[100:120]

    # 2. Compute Stats for Normalization
    print("Computing stats...")
    inventory_stats = compute_stats(train_datasets, "inventory")
    shelf_life_stats = compute_stats(train_datasets, "short_shelf_life")

    # 3. Initialize Model and Dataset
    print("Initializing models...")
    init_state = {
        product: {
            "Demand": 0,
            "Forecast": 0,
            "PharmaceuticalInventory": 10,
            "ShelfLife": 5,
            "Cost": 1.0,
        }
        for product in PRODUCT_NAMES
    }
    model = PerishablePharmaceuticalModelMultiProduct(
        PRODUCT_NAMES, init_state, decision_variable={}
    )

    train_dataset = PharmaDataset(
        train_datasets, model, PRODUCT_NAMES, inventory_stats, shelf_life_stats
    )
    eval_dataset = PharmaDataset(
        eval_datasets, model, PRODUCT_NAMES, inventory_stats, shelf_life_stats
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Policy Network
    policy_net = PolicyNetwork(INPUT_SIZE, HIDDEN_SIZE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # 5. CUDA Check
    if torch.cuda.is_available() and IS_CUDA:
        print("CUDA is available. Training on GPU.")
        policy_net = policy_net.cuda()
    else:
        print("CUDA is not available. Training on CPU.")

    # 6. Train the Policy Network
    print("Starting training...")
    time_start = time.time()
    train_rewards = train(
        train_loader,
        model,
        policy_net,
        optimizer,
        IS_CUDA,
        PRODUCT_NAMES,
        NUM_EPISODES,
        INITIAL_ENTROPY_BETA,
    )
    time_end = time.time()

    # Training Summary
    print_summary("Training Summary", {
        "Total Episodes": NUM_EPISODES,
        "Final Avg Reward": f"{train_rewards[-1]:.2f}",
        "Training Time": f"{time_end - time_start:.2f} seconds",
    })

    # Save the Model
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Policy network saved to {MODEL_SAVE_PATH}.")

    # 7. Evaluate the Policy Network
    print("Evaluating the model...")
    eval_rewards = evaluate(eval_loader, model, policy_net, IS_CUDA, PRODUCT_NAMES)
    print_summary("Evaluation Summary", {
        "Evaluation Reward Mean": f"{np.mean(eval_rewards):.2f}",
        "Evaluation Reward Std": f"{np.std(eval_rewards):.2f}",
    })

    # 8. Visualize Results
    plot_rewards(train_rewards, "Training Rewards")
    plot_rewards(eval_rewards, "Evaluation Rewards")

    # Optional: Print Final Policy Decisions
    print("Sample Policy Decisions:")
    for batch_states, dataset_indices in train_loader:
        for state_vector in batch_states[:3]:  # Print for first 3 batches
            decisions = []
            for product_state in state_vector:
                mean, log_std = policy_net(product_state)
                std = torch.exp(log_std)
                action_distribution = torch.distributions.Normal(mean, std)
                raw_action = action_distribution.sample()
                clipped_action = torch.clamp(raw_action, 0, 100)
                decisions.append(torch.round(clipped_action).item())
            print(f"Decisions: {dict(zip(PRODUCT_NAMES, decisions))}")
        break  # Only show for the first batch
