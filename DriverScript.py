import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ModelPMP import PerishablePharmaceuticalModelMultiProduct
from LPSolver import LPSolver
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# from matplotlib.animation import FuncAnimation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_size, 1)  # Mean for a single product
        # Log-std for a single product
        self.fc_log_std = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        mean = torch.sigmoid(self.fc_mean(x)) * 100  # Scale mean to [0, 100]
        log_std = self.fc_log_std(x)  # Unbounded log-std
        return mean, log_std


def load_datasets(folder_path):
    """Load all datasets from a folder into a list of DataFrames."""
    datasets = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            datasets.append(pd.read_csv(filepath))
    return datasets


def update_model(model, dataset_row, product_names):
    """Update the model's demand and forecast from a dataset row."""
    for product, forecast, demand in zip(product_names, dataset_row[::2], dataset_row[1::2]):
        model.states[product]["Forecast"] = forecast
        model.states[product]["Demand"] = demand


# # Global list for storing rewards for real-time plotting
# rewards_over_time = []

# # Initialize the Matplotlib figure and axis
# fig, ax = plt.subplots()
# x_data, y_data = [], []
# line, = ax.plot([], [], lw=2)

# def init_plot():
#     """Initialize the plot with limits and labels."""
#     ax.set_xlim(0, 10)  # Adjust x-axis dynamically during training
#     ax.set_ylim(0, 1500)  # Adjust y-axis based on expected reward range
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Reward')
#     ax.set_title('Real-Time Reward Tracking')
#     return line,

# def update_plot(frame):
#     """Update the plot with the latest reward data."""
#     x_data.append(frame)
#     y_data.append(rewards_over_time[frame])
#     line.set_data(x_data, y_data)
#     ax.set_xlim(0, max(10, len(x_data)))  # Extend x-axis dynamically
#     return line,

# ani = FuncAnimation(fig, update_plot, init_func=init_plot, blit=True, interval=100, cache_frame_data=False)
# plt.show(block=False)  # Non-blocking so training continues


def train(datasets, model, policy_net, optimizer, max_steps, is_cuda, num_episodes=3):
    """
    Train the policy network using datasets.

    Args:
    - datasets: List of DataFrames, each representing a dataset.
    - model: Instance of PerishablePharmaceuticalModelMultiProduct.
    - policy_net: Policy network for decision making.
    - optimizer: Optimizer for training the policy network.
    - max_steps: Maximum steps per dataset.
    - num_episodes: Number of training episodes for each dataset step.
    """
    product_names = model.product_names
    train_datasets = datasets[:10]
    eval_datasets = datasets[10:12]

    train_rewards_history = []
    eval_rewards_history = []

    for episode in range(num_episodes):

        for dataset_index, dataset in tqdm(enumerate(train_datasets)):

            dataset_train_rewards = []

            for step in range(max_steps):
                if step >= len(dataset):  # End if we exceed dataset length
                    break

                # Update model with the current row from the dataset
                dataset_row = dataset.iloc[step].tolist()
                update_model(model, dataset_row, product_names)

                # Run multiple episodes for this step
                step_rewards = []

                states, actions, rewards = [], [], []
                log_probs = []

                # Build current state vector
                state_vector = []
                for product in product_names:
                    product_state = [
                        model.states[product]["Demand"],
                        model.states[product]["Forecast"],
                        sum(batch["Quantity"]
                            for batch in model.pharm_invs[product]),
                        sum(batch["Quantity"]
                            for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 2)
                    ]
                    state_vector.append(product_state)

                state_vector = torch.tensor(state_vector, dtype=torch.float32)
                if is_cuda:
                    state_vector = state_vector.cuda()

                # Policy selects actions (order quantities)
                decisions, product_log_probs = [], []
                for product_state in state_vector:
                    mean, log_std = policy_net(product_state)
                    log_std = torch.clamp(log_std, min=-10, max=2)
                    std = torch.exp(log_std)
                    action_distribution = torch.distributions.Normal(mean, std)
                    raw_action = action_distribution.sample()
                    clipped_action = torch.clamp(raw_action, 0, 100)
                    decision = torch.round(clipped_action).int().item()
                    log_prob = action_distribution.log_prob(
                        clipped_action).sum()
                    product_log_probs.append(log_prob)
                    decisions.append(decision)

                log_probs = torch.stack(product_log_probs).sum()
                

                # Update model with chosen actions
                order_decision = {product: qty for product,
                                  qty in zip(product_names, decisions)}
                model.build_decision(order_quantities=order_decision)

                # Use LP solver to compute rewards and get optimal order quantities
                result = LPSolver.solve_ilp(model)
                reward = result["objective_value"]
                optimal_order_quantities = result["order_quantities"]

                # use optimal_order_quantities to adjust policy decisions
                blended_decisions = {product: int(0.5 * policy_decision + 0.5 * optimal_order)
                                     for product, policy_decision, optimal_order in zip(product_names, decisions, optimal_order_quantities.values())}
                model.build_decision(order_quantities=blended_decisions)

                # Transition the model's state
                model.transition_fn()

                # Save state, action, and reward
                states.append(state_vector)
                actions.append(torch.tensor(decisions, dtype=torch.long))
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                step_rewards.append(reward)

                # Convert collected data to tensors and update policy
                states = torch.stack(states)
                actions = torch.stack(actions).squeeze()
                rewards = torch.tensor(rewards, dtype=torch.float32)

                loss = -torch.sum(log_probs * rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss = update_policy(log_probs, optimizer, states, actions, rewards)

                # # Append reward to global list for plotting
                # rewards_over_time.append(reward)

            dataset_train_rewards.append(sum(step_rewards) / len(step_rewards))

        train_rewards_history.append(
            sum(dataset_train_rewards) / len(dataset_train_rewards))

        # Evaluation phase (optional)
        episode_eval_rewards = []
        for eval_dataset in eval_datasets:
            for step in range(max_steps):
                if step >= len(eval_dataset):  # End if we exceed dataset length
                    break

                dataset_row = eval_dataset.iloc[step].tolist()
                update_model(model, dataset_row, product_names)

                # Build state vector
                state_vector = []
                for product in product_names:
                    product_state = [
                        model.states[product]["Demand"],
                        model.states[product]["Forecast"],
                        sum(batch["Quantity"]
                            for batch in model.pharm_invs[product]),
                        sum(batch["Quantity"]
                            for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 2)
                    ]
                    state_vector.append(product_state)

                state_vector = torch.tensor(state_vector, dtype=torch.float32)

                # Evaluate policy network
                decisions = []
                for product_state in state_vector:
                    with torch.no_grad():
                        mean, log_std = policy_net(product_state)
                        log_std = torch.clamp(log_std, min=-10, max=2)
                        std = torch.exp(log_std)
                        action_distribution = torch.distributions.Normal(
                            mean, std)
                        raw_action = action_distribution.sample()
                        clipped_action = torch.clamp(raw_action, 0, 100)
                        decision = torch.round(clipped_action).int().item()
                        decisions.append(decision)

                eval_decision = {product: qty for product,
                                 qty in zip(product_names, decisions)}
                model.build_decision(order_quantities=eval_decision)
                model.transition_fn()

                # Solve using LP solver and compute evaluation reward
                result = LPSolver.solve_ilp(model)
                eval_reward = result["objective_value"]
                episode_eval_rewards.append(eval_reward)

        eval_rewards_history.append(
            sum(episode_eval_rewards) / len(episode_eval_rewards))

        # Print metrics
        print(f"Dataset {dataset_index + 1}: Train Reward = {train_rewards_history[-1]:.2f}, "
              f"Eval Reward = {eval_rewards_history[-1]:.2f}")

    return train_rewards_history, eval_rewards_history


def update_policy(log_probs, optimizer, states, actions, rewards):
    """Update the policy network based on collected rewards."""
    loss = -torch.sum(log_probs * rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def plot_rewards_train(train_rewards):
    """Plot the training rewards."""
    plt.plot(train_rewards, label="Train Reward")
    plt.xlabel("Dataset")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.show()


def plot_rewards_eval(eval_rewards):
    """Plot the evaluation rewards."""
    plt.plot(eval_rewards, label="Eval Reward")
    plt.xlabel("Dataset")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    folder_path = "dataMP"
    is_training = True
    is_cuda = True
    datasets = load_datasets(folder_path)
    product_names = ["Product1", "Product2", "Product3", "Product4"]
    init_state = {product: {"Demand": 0, "Forecast": 0, "PharmaceuticalInventory": 10, "ShelfLife": 5, "Cost": 1.0}
                  for product in product_names}
    model = PerishablePharmaceuticalModelMultiProduct(
        product_names, init_state, decision_variable={})

    input_size = len(product_names)
    hidden_size = 128
    policy_net = PolicyNetwork(input_size, hidden_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # check if cuda is available
    if torch.cuda.is_available() and is_cuda:
        print("CUDA is available. Training on GPU.")
        policy_net = policy_net.cuda()
    else:
        print("CUDA is not available. Training on CPU.")

    if is_training:
        time_start = time.time()

        train_rewards, eval_rewards = train(
            datasets, model, policy_net, optimizer, is_cuda, max_steps=50)
        time_end = time.time()
        print(f"Training time: {time_end - time_start:.2f} seconds")
        torch.save(policy_net.state_dict(), "policy_net.pth")
        plot_rewards_train(train_rewards)
        plot_rewards_eval(eval_rewards)
    else:
        policy_net.load_state_dict(torch.load("policy_net.pth"))
