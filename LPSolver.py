import pulp

class LPSolver:
    @staticmethod
    def solve_ilp(model):
        prob = pulp.LpProblem("PharmaceuticalInventoryOptimization", pulp.LpMaximize)
        
        # Define integer order quantity variables for each product
        order_vars = {product: pulp.LpVariable(f"Order_{product}", lowBound=0, cat='Integer') for product in model.product_names}
        
        total_profit = 0
        for product in model.product_names:
            # Inventory, demand, forecast, and near-expiry inventory details
            current_inventory = sum(batch["Quantity"] for batch in model.pharm_invs[product])
            demand = model.states[product]["Demand"]
            forecast = model.states[product]["Forecast"]
            near_expiry_inventory = sum(batch["Quantity"] for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 2)
            
            # Adaptive max inventory with higher forecast weight
            adaptive_max_inventory = model.max_inventory[product] - near_expiry_inventory + (forecast * 0.3)
            
            # Effective sales constrained by available stock and demand
            effective_sales = pulp.LpVariable(f"Effective_Sales_{product}", lowBound=0)
            prob += effective_sales <= current_inventory + order_vars[product]
            prob += effective_sales <= demand

            # Revenue from effective sales
            revenue = model.price[product] * effective_sales
            
            # Enhanced holding and expired costs to prioritize fresh inventory
            holding_cost = sum(batch["Quantity"] * 0.3 for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 2)
            expired_cost = sum(batch["Quantity"] * model.cost[product] * 2.0 for batch in model.pharm_invs[product] if batch["ShelfLife"] <= 0)

            # Total profit includes revenue minus holding and expired costs
            total_profit += revenue - holding_cost - expired_cost

            # Constrain inventory + new orders to not exceed adaptive max
            prob += current_inventory + order_vars[product] <= adaptive_max_inventory

        # Objective to maximize profit
        prob += total_profit, "Total_Profit"
        
        # Solve LP problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract the optimal order quantities for each product
        order_quantities = {product: pulp.value(order_vars[product]) for product in model.product_names}

        # Return the objective value and order quantities
        return {
            "objective_value": pulp.value(prob.objective),
            "order_quantities": order_quantities
        }
