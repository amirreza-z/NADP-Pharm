import torch
from DriverScript import PolicyNetwork
from LPSolver import LPSolver
from ModelPMP import PerishablePharmaceuticalModelMultiProduct

# Load the trained model
input_size = 4  # Same as during training
hidden_size = 128  # Same as during training
policy_net = PolicyNetwork(input_size, hidden_size)
policy_net.load_state_dict(torch.load("policy_net.pth", map_location=torch.device('cpu'))) # Load trained weights
policy_net.eval()  # Set to evaluation mode

import os
import pandas as pd

def load_dataset(filepath):
    """Load a single dataset from a CSV file."""
    return pd.read_csv(filepath)

# Path to the new dataset
new_dataset_path = "dataMP/dataset100.csv"  # Replace with the path to your desired dataset
new_dataset = load_dataset(new_dataset_path)


def prepare_state_vector(dataframe, model, product_names):
    """Prepare state vectors for all rows in a dataframe."""
    states = []
    for _, row in dataframe.iterrows():
        state_vector = []
        for product, forecast, demand in zip(product_names, row[::2], row[1::2]):
            model.states[product]["Forecast"] = forecast
            model.states[product]["Demand"] = demand
            inventory = sum(batch["Quantity"] for batch in model.pharm_invs[product])
            short_shelf_life = sum(batch["Quantity"] for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 2)

            # Normalize variables
            normalized_demand = demand / 100
            normalized_forecast = forecast / 100
            normalized_inventory = (inventory - 300) / 50  # Use same mean and std as training
            normalized_short_shelf_life = (short_shelf_life - 30) / 5  # Use same mean and std as training

            state_vector.append([normalized_demand, normalized_forecast, normalized_inventory, normalized_short_shelf_life])
        states.append(torch.tensor(state_vector, dtype=torch.float32))
    return states

def apply_policy(policy_net, states, product_names, model):
    """Apply the trained policy network to a dataset and compute rewards."""
    rewards = []
    for state_vector in states:
        decisions = []
        for product_state in state_vector:
            with torch.no_grad():
                mean, log_std = policy_net(product_state)
                std = torch.exp(log_std)
                action_distribution = torch.distributions.Normal(mean, std)
                raw_action = action_distribution.sample()
                clipped_action = torch.clamp(raw_action, 0, 100)
                decisions.append(torch.round(clipped_action).int().item())
        
        # Map decisions to products
        order_decision = {product: qty for product, qty in zip(product_names, decisions)}
        model.build_decision(order_quantities=order_decision)
        
        # Solve the model and transition state
        result = LPSolver.solve_ilp(model)
        reward = result["objective_value"]
        rewards.append(reward)
        model.transition_fn()  # Move to the next state
    return rewards

if __name__ == "__main__":
    # Define paths and products
    product_names = ["Product1", "Product2", "Product3", "Product4"]
    new_dataset_path = "dataMP/dataset100.csv"

    # Load the dataset
    new_dataset = load_dataset(new_dataset_path)

    # Initialize the model
    init_state = {product: {"Demand": 0, "Forecast": 0, "PharmaceuticalInventory": 10, "ShelfLife": 5, "Cost": 1.0} for product in product_names}
    model = PerishablePharmaceuticalModelMultiProduct(product_names, init_state, decision_variable={})

    # Prepare states
    states = prepare_state_vector(new_dataset, model, product_names)

    # Load the trained policy
    policy_net = PolicyNetwork(input_size=4, hidden_size=128)
    policy_net.load_state_dict(torch.load("policy_net.pth", map_location=torch.device('cpu')))
    policy_net.eval()

    # Apply policy and calculate rewards
    rewards = apply_policy(policy_net, states, product_names, model)

    # Display results
    print("Rewards:", rewards)
    print("Average Reward:", sum(rewards) / len(rewards))
