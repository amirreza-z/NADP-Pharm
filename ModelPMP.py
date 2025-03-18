import numpy as np
from collections import namedtuple

class PerishablePharmaceuticalModelMultiProduct:

    """
    Model for multi-product pharmaceutical inventory management considering perishability.
    Tracks inventory by batches with shelf life, handles demand fulfillment using FIFO, 
    and includes holding costs and waste penalties for perishable products.
    """

    def __init__(self, product_names, init_state, decision_variable):
        self.product_names = product_names
        self.decision_variable = decision_variable
        self.State = namedtuple("State", product_names)
        self.states = {product: init_state[product] for product in product_names}
        self.Decisions = {product: {"OrderQuantity": 0.0} for product in product_names}
        self.num_expired = 0

        # Initialize inventory as a list of batches, each with quantity and shelf life
        self.pharm_invs = {
            product: [{"Quantity": init_state[product]["PharmaceuticalInventory"], "ShelfLife": init_state[product]["ShelfLife"]}]
            for product in product_names
        }
        self.cost = {product: init_state[product]["Cost"] for product in product_names}
        self.price = {product: 10.0 for product in product_names}

        # Add max inventory limit for each product
        self.max_inventory = {product: 25.0 for product in product_names}  # Set a default maximum inventory

        self.obj_fn = 0.0
        self.start = True

    def build_state(self, product, info):
        """
        Update the state for a specific product with new information.
        :param info: dict - contains updated state information for the product.
        """
        self.states[product] = info

    def build_decision(self, order_quantities=None):
        """
        Update the decision variable (order quantities) for each product.
        :param order_quantities: dict - contains the quantity to order for each product.
        """

        if order_quantities is not None:

            for product in self.product_names:
                self.Decisions[product]["OrderQuantity"] = order_quantities[product]

    def transition_fn(self):
        """
        Transition function to update the inventory state:
        - Reduce the shelf life of all batches.
        - Remove expired products.
        - Fulfill demand from the oldest batches (FIFO).
        """
        for product in self.product_names:
            # Age the inventory (reduce shelf life)
            for batch in self.pharm_invs[product]:
                batch["ShelfLife"] -= 1

            # Remove expired products
            self.remove_expired_products(product)

            # Fulfill demand using the oldest batches (FIFO)
            demand = self.states[product]["Demand"]
            for batch in sorted(self.pharm_invs[product], key=lambda x: x["ShelfLife"]):
                if demand <= 0:
                    break
                used_quantity = min(demand, batch["Quantity"])
                batch["Quantity"] -= used_quantity
                demand -= used_quantity

            # Add new order batch with fresh inventory and maximum shelf life (e.g., 5 periods)
            order_quantity = self.Decisions[product]['OrderQuantity']
            if order_quantity > 0:
                self.pharm_invs[product].append({"Quantity": order_quantity, "ShelfLife": 5})

    def remove_expired_products(self, product):
        """
        Remove expired batches from the inventory.
        :param product: str - the product whose expired batches are to be removed.
        """
        self.num_expired = sum(1 for batch in self.pharm_invs[product] if batch["ShelfLife"] <= 0)
        self.pharm_invs[product] = [batch for batch in self.pharm_invs[product] if batch["ShelfLife"] > 0]

    def objective_fn(self):
        total_obj_fn = 0.0
        for product in self.product_names:
            # Calculate holding costs for near-expiry items (penalty for items with low shelf life)
            holding_cost = sum(batch["Quantity"] * 0.2 for batch in self.pharm_invs[product] if batch["ShelfLife"] <= 3)

            # Calculate cost of expired products (fully expired items)
            expired_cost = sum(batch["Quantity"] * self.cost[product] * 2 for batch in self.pharm_invs[product] if batch["ShelfLife"] <= 0)

            # Calculate profit from selling products based on demand satisfaction
            fulfilled_demand = sum(batch["Quantity"] for batch in self.pharm_invs[product])
            revenue = self.price[product] * min(fulfilled_demand, self.states[product]["Demand"])

            # Penalty for overstocking (inventory above a threshold)
            inventory_level = sum(batch["Quantity"] for batch in self.pharm_invs[product])
            overstock_penalty = max(0, inventory_level - self.max_inventory[product]) * 0.1

            # Update total objective function (profit - holding cost - expired cost - overstock penalty)
            total_obj_fn += revenue - holding_cost - expired_cost - overstock_penalty

        return total_obj_fn

    