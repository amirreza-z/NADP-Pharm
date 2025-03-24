import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import wandb  # WandB for logging and monitoring
from tqdm import tqdm

# Import custom modules
from ModelPMP import PerishablePharmaceuticalModelMultiProduct
from LPSolver import LPSolver
from DriverScript import PolicyNetwork, PharmaDataset  
from DriverScript import load_datasets, compute_stats, plot_rewards, train, evaluate  # Utility functions


# Ensure API key and silent mode are set
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_API_KEY"] = "fca285677b0b05dee2f27489f4063497be8ba82e"
# ------------------------------
# WandB Integration
# ------------------------------
wandb.login()

# ------------------------------
# Hyperparameters
# ------------------------------
config = {
    "folder_path": "dataMP",
    "is_cuda": True,
    "product_names": ["Product1"],  # Only Product1 for now
    # "product_names": ["Product1", "Product2", "Product3", "Product4"],
    "input_size": 4,  # Number of state variables per product
    "hidden_size": 128,
    "learning_rate": 1e-4, 
    "batch_size": 16,
    "num_episodes": 200,
    "initial_entropy_beta": 0.005,
    "model_save_path": "policy_net.pth",
}
wandb.init(project="NADP-Pharm-Project", name="updated_logging", config=config, mode="online")


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
    datasets = load_datasets(config["folder_path"])
    train_datasets = datasets[:800]
    eval_datasets = datasets[800:999]

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
        for product in config["product_names"]
    }
    model = PerishablePharmaceuticalModelMultiProduct(
        config["product_names"], init_state, decision_variable={}
    )

    train_dataset = PharmaDataset(
        train_datasets, model, config["product_names"], inventory_stats, shelf_life_stats
    )
    eval_dataset = PharmaDataset(
        eval_datasets, model, config["product_names"], inventory_stats, shelf_life_stats
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)

    # 4. Initialize Policy Network
    policy_net = PolicyNetwork(config["input_size"], config["hidden_size"])
    optimizer = optim.Adam(policy_net.parameters(), lr=config["learning_rate"])

    # 5. CUDA Check
    if torch.cuda.is_available() and config["is_cuda"]:
        print("CUDA is available. Training on GPU.")
        policy_net = policy_net.cuda()
    else:
        print("CUDA is not available. Training on CPU.")

    # 6. Train the Policy Network
    print("Starting training...")
    time_start = time.time()
    train_rewards = []
    
    # Training logic
    episode_reward, episode_loss = train(
        train_loader,
        model,
        policy_net,
        optimizer,
        config["is_cuda"],
        config["product_names"],
        config["num_episodes"],
        initial_entropy_beta=config["initial_entropy_beta"],
    )
    
    

    # Log weights for all parameters
    for name, param in policy_net.named_parameters():
        wandb.log({f"Weights/{name}": wandb.Histogram(param.data.cpu().numpy())})

    time_end = time.time()

    # Training Summary
    print_summary("Training Summary", {
        "Total Episodes": config["num_episodes"],
        "Final Avg Reward": f"{np.mean(train_rewards):.2f}",
        "Training Time": f"{time_end - time_start:.2f} seconds",
    })

    # Save the Model
    torch.save(policy_net.state_dict(), config["model_save_path"])
    wandb.save(config["model_save_path"])
    print(f"Policy network saved to {config['model_save_path']}.")

    # 7. Evaluate the Policy Network
    print("Evaluating the model...")
    eval_rewards = []

    for batch in tqdm(eval_loader, desc="Evaluation Batches"):
        eval_reward = evaluate(batch, model, policy_net, config["is_cuda"], config["product_names"])
        eval_rewards.append(eval_reward)
    
    eval_mean_reward = np.mean(eval_rewards)
    eval_std_reward = np.std(eval_rewards)

    # Log evaluation metrics to WandB
    wandb.log({"Eval Reward Mean": eval_mean_reward, "Eval Reward Std": eval_std_reward})

    print_summary("Evaluation Summary", {
        "Evaluation Reward Mean": f"{eval_mean_reward:.2f}",
        "Evaluation Reward Std": f"{eval_std_reward:.2f}",
    })

    # 8. Visualize Results
    plt.plot(train_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Time")
    wandb.log({"Training Rewards Plot": wandb.Image(plt)})
    plt.show()

    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.title("Evaluation Rewards Over Time")
    wandb.log({"Evaluation Rewards Plot": wandb.Image(plt)})
    plt.show()

    # Finish WandB Run
    wandb.finish()
