import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ModelPMP import PerishablePharmaceuticalModelMultiProduct
from LPSolver import LPSolver
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_log_std = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        mean = torch.sigmoid(self.fc_mean(x)) * 100  # Scale mean to [0, 100]
        log_std = self.fc_log_std(x)
        return mean, log_std


class PharmaDataset(Dataset):
    def __init__(self, dataframes, model, product_names):
        """Prepare dataset rows for training."""
        self.dataframes = dataframes
        self.model = model
        self.product_names = product_names
        self.samples = []

        #Inintialize statistics for normalization
        self.inventory_mean = 300
        self.inventory_std = 50
        self.short_shelf_life_mean = 30
        self.short_shelf_life_std = 5
        self.sample_count = 0
        for df_index, df in enumerate(dataframes):
            for row_index in range(len(df)):
                self.samples.append((df_index, row_index))

    def update_running_stats(self, value, mean, std):
        self.sample_count += 1
        new_mean = (mean * (self.sample_count - 1) + value) / self.sample_count
        new_std = ((std**2) * (self.sample_count - 1) + (value - new_mean)**2) / self.sample_count
        return new_mean, new_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_index, row_index = self.samples[idx]
        row = self.dataframes[dataset_index].iloc[row_index].tolist()
        state_vector = []
        
        for product, forecast, demand in zip(self.product_names, row[::2], row[1::2]):
            self.model.states[product]["Forecast"] = forecast
            self.model.states[product]["Demand"] = demand
            inventory = sum(batch["Quantity"] for batch in self.model.pharm_invs[product])
            short_shelf_life = sum(batch["Quantity"] for batch in self.model.pharm_invs[product] if batch["ShelfLife"] <= 2)

            # Normalize variables
            normalized_demand = demand/100
            normalized_forecast = forecast/100

            # Update running statistics for inventory and short shelf life
            self.inventory_mean, self.inventory_std = self.update_running_stats(inventory, self.inventory_mean, self.inventory_std)
            self.short_shelf_life_mean, self.short_shelf_life_std = self.update_running_stats(short_shelf_life, self.short_shelf_life_mean, self.short_shelf_life_std)
            
            normalized_inventory = (inventory - self.inventory_mean) / (self.inventory_std + 1e-5)
            normalized_short_shelf_life = (short_shelf_life - self.short_shelf_life_mean) / (self.short_shelf_life_std + 1e-5)
            state_vector.append([normalized_demand, normalized_forecast, normalized_inventory, normalized_short_shelf_life])
            
        
        return torch.tensor(state_vector, dtype=torch.float32), dataset_index


def load_datasets(folder_path):
    """Load all datasets from CSV files."""
    datasets = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            datasets.append(pd.read_csv(filepath))
    return datasets


def train(train_loader, model, policy_net, optimizer, is_cuda, product_names, num_episodes):
    policy_net.train()
    train_rewards_history = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        episode_rewards = []

        for batch_states, dataset_indices in tqdm(train_loader):
            if is_cuda:
                batch_states = batch_states.cuda()
            
            batch_log_probs, batch_rewards = [], []
            batch_losses = []
            batch_mean_rewards = []
            
            for state_vector in batch_states:
                product_log_probs, decisions = [], []
                
                for product_state in state_vector:
                    mean, log_std = policy_net(product_state)
                    log_std = torch.clamp(log_std, min=-10, max=2)
                    std = torch.exp(log_std)
                    action_distribution = torch.distributions.Normal(mean, std)
                    raw_action = action_distribution.sample()
                    clipped_action = torch.clamp(raw_action, 0, 100)
                    decision = torch.round(clipped_action).int().item()
                    product_log_probs.append(action_distribution.log_prob(clipped_action).sum())
                    decisions.append(decision)
                
                log_probs = torch.stack(product_log_probs).sum()
                order_decision = {product: qty for product, qty in zip(product_names, decisions)}
                model.build_decision(order_quantities=order_decision)
                result = LPSolver.solve_ilp(model)
                reward = result["objective_value"]
                model.transition_fn()
                
                batch_log_probs.append(log_probs)
                batch_rewards.append(torch.tensor(reward, dtype=torch.float32))
                batch_mean_rewards.append(reward)

            batch_log_probs = torch.stack(batch_log_probs)
            batch_rewards = torch.stack(batch_rewards)

            if is_cuda:
                batch_log_probs = batch_log_probs.cuda()
                batch_rewards = batch_rewards.cuda()
            
            loss = -torch.sum(batch_log_probs * batch_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_rewards.append(torch.mean(batch_rewards).item())
            batch_losses.append(loss.item())

        avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
        avg_batch_loss = sum(batch_losses) / len(batch_losses)
        print(f"Episode {episode + 1} Metrics: Avg Reward: {avg_episode_reward:.2f}, Avg Loss: {avg_batch_loss:.4f}")
        train_rewards_history.append(avg_episode_reward)
    return train_rewards_history


def evaluate(eval_loader, model, policy_net, is_cuda, product_names):
    policy_net.eval()
    eval_rewards_history = []

    with torch.no_grad():
        for batch_states, dataset_indices in tqdm(eval_loader):
            if is_cuda:
                batch_states = batch_states.cuda()
            
            for state_vector in batch_states:
                decisions = []
                for product_state in state_vector:
                    mean, log_std = policy_net(product_state)
                    log_std = torch.clamp(log_std, min=-10, max=2)
                    std = torch.exp(log_std)
                    action_distribution = torch.distributions.Normal(mean, std)
                    raw_action = action_distribution.sample()
                    clipped_action = torch.clamp(raw_action, 0, 100)
                    decisions.append(torch.round(clipped_action).int().item())
                
                order_decision = {product: qty for product, qty in zip(product_names, decisions)}
                model.build_decision(order_quantities=order_decision)
                result = LPSolver.solve_ilp(model)
                eval_rewards_history.append(result["objective_value"])
                model.transition_fn()
    
    return eval_rewards_history


def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    folder_path = "dataMP"
    is_cuda = True
    product_names = ["Product1", "Product2", "Product3", "Product4"]

    # Load datasets
    datasets = load_datasets(folder_path)
    train_datasets = datasets[:100]
    eval_datasets = datasets[100:120]

    # Initialize model
    init_state = {product: {"Demand": 0, "Forecast": 0, "PharmaceuticalInventory": 10, "ShelfLife": 5, "Cost": 1.0} for product in product_names}
    model = PerishablePharmaceuticalModelMultiProduct(product_names, init_state, decision_variable={})
    
    train_dataset = PharmaDataset(train_datasets, model, product_names)
    eval_dataset = PharmaDataset(eval_datasets, model, product_names)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    input_size = 4  # State size for each product
    hidden_size = 128
    policy_net = PolicyNetwork(input_size, hidden_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    if torch.cuda.is_available() and is_cuda:
        print("CUDA is available. Training on GPU.")
        policy_net = policy_net.cuda()
    else:
        print("CUDA is not available. Training on CPU.")

    # Train the model
    num_episodes = 10
    time_start = time.time()
    train_rewards = train(train_loader, model, policy_net, optimizer, is_cuda, product_names, num_episodes)
    eval_rewards = evaluate(eval_loader, model, policy_net, is_cuda, product_names)
    time_end = time.time()

    print(f"Training time: {time_end - time_start:.2f} seconds")
    torch.save(policy_net.state_dict(), "policy_net.pth")

    plot_rewards(train_rewards, "Training Rewards")
    plot_rewards(eval_rewards, "Evaluation Rewards")
