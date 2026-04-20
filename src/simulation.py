import numpy as np
import random

class WarehouseSimulator:
    """
    Simulates a picker's movement in a grid-based warehouse.
    """
    def __init__(self, product_ids, grid_size=(100, 100), start_pos=(0, 0)):
        self.product_ids = list(product_ids)
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.n_products = len(product_ids)
        
        # Check if grid can accommodate products
        if self.n_products > grid_size[0] * grid_size[1]:
            # Scale grid if necessary
            side = int(np.ceil(np.sqrt(self.n_products))) + 5
            self.grid_size = (side, side)

    def create_baseline_layout(self):
        """
        Assigns products to random locations (Baseline).
        """
        locations = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        random.shuffle(locations)
        
        layout = {pid: locations[i] for i, pid in enumerate(self.product_ids)}
        return layout

    def create_optimized_layout(self, product_zones):
        """
        Assigns products to locations grouped by zone (Optimized).
        - product_zones: dict {product_id: zone_id}
        """
        # Sort products by zone
        sorted_zones = sorted(product_zones.items(), key=lambda x: x[1])
        
        # Fill the grid linearly with sorted products
        # This ensures products in the same zone are physically close
        layout = {}
        for i, (pid, zone) in enumerate(sorted_zones):
            x = i % self.grid_size[0]
            y = i // self.grid_size[0]
            layout[pid] = (x, y)
            
        return layout

    def calculate_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def simulate_order_picking(self, order_products, layout):
        """
        Calculates total distance to pick all products in one order.
        Assumes the picker visits each product and returns to start.
        """
        total_dist = 0
        current_pos = self.start_pos
        
        # Filter products present in layout
        valid_products = [p for p in order_products if p in layout]
        
        if not valid_products:
            return 0
            
        # Simplification: Picker visits them in order of occurrence
        # In a real warehouse, they'd use a pathfinding algo (S-Shape or Largest Gap)
        # But for comparison, consistency is key.
        for pid in valid_products:
            target_pos = layout[pid]
            total_dist += self.calculate_manhattan_distance(current_pos, target_pos)
            current_pos = target_pos
            
        # Return to start
        total_dist += self.calculate_manhattan_distance(current_pos, self.start_pos)
        return total_dist

    def run_simulation(self, orders, layout, sample_size=1000):
        """
        Runs simulation for a set of orders.
        """
        if len(orders) > sample_size:
            sim_orders = random.sample(list(orders), sample_size)
        else:
            sim_orders = orders
            
        distances = []
        for order in sim_orders:
            dist = self.simulate_order_picking(order, layout)
            distances.append(dist)
            
        return np.mean(distances), np.sum(distances)

if __name__ == "__main__":
    # Small test
    products = [f"P{i}" for i in range(50)]
    zones = {f"P{i}": i // 10 for i in range(50)}
    sim = WarehouseSimulator(products, grid_size=(10, 10))
    
    orders = [[f"P{random.randint(0,49)}" for _ in range(3)] for _ in range(100)]
    
    base_layout = sim.create_baseline_layout()
    opt_layout = sim.create_optimized_layout(zones)
    
    m1, s1 = sim.run_simulation(orders, base_layout)
    m2, s2 = sim.run_simulation(orders, opt_layout)
    
    print(f"Baseline Avg Distance: {m1:.2f}")
    print(f"Optimized Avg Distance: {m2:.2f}")
    print(f"Improvement: {((m1-m2)/m1)*100:.1f}%")
