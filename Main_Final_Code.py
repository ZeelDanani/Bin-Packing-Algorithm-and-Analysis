import time
import random
import math
import matplotlib.pyplot as plt
from math import ceil, log2
import numpy as np
from collections import defaultdict
import pulp
import numpy as np
import heapq
from collections import defaultdict, Counter
import time

def grouped_lp_bin_packing(items, num_groups=10, max_iterations=100, tolerance=1e-4):
    """
    Advanced bin packing algorithm using:
    1. Item grouping and rounding
    2. Gilmore-Gomory LP with column generation on simplified items
    3. Karmarkar-Karp rounding for the final solution
    
    Parameters:
    - items: List of item sizes (each between 0 and 1)
    - num_groups: Number of groups to divide items into
    - max_iterations: Maximum iterations for column generation
    - tolerance: Tolerance for column generation termination
    
    Returns:
    - List of bin fill levels
    """
    if not items:
        return []
    
    # Step 1: Group items by size and round up within groups
    grouped_items, original_to_rounded, rounded_to_originals = group_and_round_items(items, num_groups)
    
    # Step 2: Solve LP using Gilmore-Gomory on the rounded items
    rounded_solution = solve_gilmore_gomory(grouped_items, max_iterations, tolerance)
    
    # Step 3: Map the rounded solution back to original items and use KK for rounding
    final_bins = map_and_round_solution(items, rounded_solution, original_to_rounded, rounded_to_originals)
    
    return final_bins

def group_and_round_items(items, num_groups):
    """
    Group items by size and round up to largest size in the group.
    
    Parameters:
    - items: Original items
    - num_groups: Number of size groups
    
    Returns:
    - grouped_items: List of (size, count) for rounded items
    - original_to_rounded: Mapping from original sizes to rounded sizes
    - rounded_to_originals: Mapping from rounded sizes to lists of original items
    """
    if not items:
        return [], {}, {}
    
    # Determine group boundaries
    min_size = min(items)
    max_size = max(items)
    group_width = (max_size - min_size) / num_groups if num_groups > 1 and max_size > min_size else 0.1
    
    # Initialize mappings
    original_to_rounded = {}
    rounded_to_originals = defaultdict(list)
    
    # First pass: Group items and determine the max size in each group
    groups = defaultdict(list)
    for item in items:
        # Determine which group this item belongs to
        if group_width > 0:
            group_idx = min(num_groups - 1, int((item - min_size) / group_width))
        else:
            group_idx = 0
        groups[group_idx].append(item)
    
    # Determine the representative (maximum) size for each group
    group_representatives = {}
    for group_idx, group_items in groups.items():
        if group_items:
            group_representatives[group_idx] = max(group_items)
    
    # Second pass: Create the mappings
    for group_idx, group_items in groups.items():
        rounded_size = group_representatives[group_idx]
        for original_item in group_items:
            original_to_rounded[original_item] = rounded_size
            rounded_to_originals[rounded_size].append(original_item)
    
    # Create the grouped items list: [(size, count), ...]
    grouped_items = [(size, len(items_list)) for size, items_list in rounded_to_originals.items()]
    
    return grouped_items, original_to_rounded, rounded_to_originals

def solve_gilmore_gomory(grouped_items, max_iterations=100, tolerance=1e-6):
    """
    Solve the bin packing problem using Gilmore-Gomory column generation.
    
    Parameters:
    - grouped_items: List of (size, count) pairs
    - max_iterations: Maximum column generation iterations
    - tolerance: Tolerance for terminating column generation
    
    Returns:
    - Solution dictionary mapping patterns to fractional counts
    """
    # Extract sizes and counts
    sizes = [item[0] for item in grouped_items]
    counts = [item[1] for item in grouped_items]
    
    n = len(grouped_items)
    if n == 0:
        return {}
    
    # Initialize with single-item patterns
    patterns = []
    for i in range(n):
        # Maximum number of items of this type that can fit in a bin
        max_count = min(counts[i], int(1.0 / sizes[i]))
        pattern = [0] * n
        pattern[i] = max_count
        patterns.append(pattern)
    
    # Column generation loop
    iteration = 0
    while iteration < max_iterations:
        # Set up the restricted master problem
        master = pulp.LpProblem("BinPackingMaster", pulp.LpMinimize)
        
        # Variables: How many times to use each pattern
        lambda_vars = [pulp.LpVariable(f"lambda_{j}", lowBound=0, cat=pulp.LpContinuous) 
                      for j in range(len(patterns))]
        
        # Objective: Minimize number of bins
        master += pulp.lpSum(lambda_vars)
        
        # Constraints: Must satisfy demand for each item type
        for i in range(n):
            master += pulp.lpSum(patterns[j][i] * lambda_vars[j] for j in range(len(patterns))) >= counts[i]
        
        # Solve the master problem
        master.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Get dual values
        dual_values = []
        for i in range(n):
            constraint_name = f"_C{i+1}"
            if constraint_name in master.constraints:
                dual_values.append(master.constraints[constraint_name].pi)
            else:
                dual_values.append(0)
        
        # Solve the pricing problem (knapsack)
        new_pattern, reduced_cost = solve_knapsack_pricing(sizes, counts, dual_values)
        
        # Check if we found a pattern with negative reduced cost
        if reduced_cost < -tolerance:
            patterns.append(new_pattern)
        else:
            # No pattern with negative reduced cost, we're done
            break
            
        iteration += 1
    
    # Extract the solution
    solution = {}
    for j in range(len(patterns)):
        pattern_value = lambda_vars[j].value()
        if pattern_value is not None and pattern_value > tolerance:
            # Convert the pattern to a hashable format (tuple)
            pattern_tuple = tuple(patterns[j])
            solution[pattern_tuple] = pattern_value
    
    return solution

def solve_knapsack_pricing(sizes, counts, dual_values):
    """
    Solve the knapsack pricing problem to find a new pattern.
    
    Parameters:
    - sizes: Item sizes
    - counts: Item counts (demand)
    - dual_values: Dual values from the master problem
    
    Returns:
    - new_pattern: A new pattern
    - reduced_cost: The reduced cost of the pattern
    """
    n = len(sizes)
    if n == 0:
        return [], 1.0
    
    # Set up the pricing problem
    pricing = pulp.LpProblem("KnapsackPricing", pulp.LpMaximize)
    
    # Variables: How many of each item to include in the pattern
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=counts[i], cat=pulp.LpInteger) 
        for i in range(n)]
    
    # Objective: Maximize sum of dual values * number of items
    pricing += pulp.lpSum(dual_values[i] * x[i] for i in range(n))
    
    # Constraint: Total size cannot exceed bin capacity
    pricing += pulp.lpSum(sizes[i] * x[i] for i in range(n)) <= 1
    
    # Solve the pricing problem
    pricing.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract the solution
    new_pattern = [int(round(x[i].value() or 0)) for i in range(n)]
    
    # Calculate the reduced cost
    reduced_cost = 1 - sum(dual_values[i] * new_pattern[i] for i in range(n))
    
    return new_pattern, reduced_cost

def map_and_round_solution(original_items, lp_solution, original_to_rounded, rounded_to_originals):
    """
    Map the LP solution back to original items and use Karmarkar-Karp to round.
    
    Parameters:
    - original_items: Original list of item sizes
    - lp_solution: Solution from the Gilmore-Gomory algorithm
    - original_to_rounded: Mapping from original sizes to rounded sizes
    - rounded_to_originals: Mapping from rounded sizes to original items
    
    Returns:
    - List of bin fill levels
    """
    if not lp_solution:
        return []
    
    # Create a copy of the rounded_to_originals mapping to work with
    item_pool = {size: items[:] for size, items in rounded_to_originals.items()}
    
    # Step 1: Convert the LP solution to a list of patterns with integer counts
    bins = []
    fractional_patterns = []
    
    for pattern_tuple, count in lp_solution.items():
        # Integer part of the count
        int_count = int(count)
        
        # Process integer counts
        for _ in range(int_count):
            bin_items = []
            
            # For each item type in the pattern
            for i, item_count in enumerate(pattern_tuple):
                if i >= len(list(item_pool.keys())):
                    continue
                
                # Get the corresponding rounded size
                rounded_size = list(item_pool.keys())[i]
                
                # Take up to 'item_count' items of this size
                available_items = item_pool[rounded_size]
                for _ in range(min(item_count, len(available_items))):
                    if available_items:
                        bin_items.append(available_items.pop(0))
            
            # Add this bin if it has items
            if bin_items:
                bins.append(sum(bin_items))
        
        # Store fractional part for later
        fractional = count - int_count
        if fractional > 1e-6:
            fractional_patterns.append((pattern_tuple, fractional))
    
    # Step 2: Collect remaining items
    remaining_items = []
    for items_list in item_pool.values():
        remaining_items.extend(items_list)
    
    # Step 3: Handle fractional patterns using a simple approach
    # (Process in order of decreasing fractional value)
    fractional_patterns.sort(key=lambda x: x[1], reverse=True)
    
    # Try to form additional bins from fractional patterns
    for pattern_tuple, _ in fractional_patterns:
        if not remaining_items:
            break
            
        bin_items = []
        pattern_fits = True
        
        # Check if we can form this pattern with remaining items
        temp_remaining = remaining_items[:]
        temp_bin = []
        
        # For simplicity, just try to fill one more bin with each fractional pattern
        current_size = 0
        for i, item_count in enumerate(pattern_tuple):
            if i >= len(list(rounded_to_originals.keys())):
                continue
                
            rounded_size = list(rounded_to_originals.keys())[i]
            size_count = 0
            
            for j, item in enumerate(temp_remaining):
                if original_to_rounded.get(item) == rounded_size and size_count < item_count:
                    if current_size + item <= 1.0:
                        temp_bin.append(item)
                        current_size += item
                        size_count += 1
                        temp_remaining[j] = None  # Mark as used
            
            if size_count < item_count:
                pattern_fits = False
                break
        
        if pattern_fits and temp_bin:
            bins.append(sum(temp_bin))
            remaining_items = [item for item in temp_remaining if item is not None]
    
    # Step 4: Use Karmarkar-Karp for any remaining items
    if remaining_items:
        kk_bins = karmarkar_karp_packing(remaining_items)
        bins.extend(kk_bins)
    
    return bins

def karmarkar_karp_packing(items):
    """
    Implementation of Karmarkar-Karp bin packing algorithm.
    """
    if not items:
        return []
    
    # Sort items in non-increasing order
    sorted_items = sorted(items, reverse=True)
    
    # Use Best Fit Decreasing (which is similar to KK's approach)
    bins = []
    
    for item in sorted_items:
        # Find the bin with the most remaining space that can fit this item
        best_fit = -1
        best_space = float('inf')
        
        for i, bin_level in enumerate(bins):
            space = 1.0 - bin_level
            if space >= item and space < best_space:
                best_fit = i
                best_space = space
        
        if best_fit >= 0:
            # Add item to the best bin
            bins[best_fit] += item
        else:
            # Create a new bin
            bins.append(item)
    
    return bins

# Main wrapper function
def lp_bin_packing_with_grouping_and_kk(items, num_groups=10):
    """
    Main wrapper function for the advanced LP bin packing algorithm.
    
    Parameters:
    - items: List of item sizes
    - num_groups: Number of groups for item rounding
    
    Returns:
    - List of bin fill levels
    """
    return grouped_lp_bin_packing(items, num_groups)
class SegmentTree:
    def __init__(self, capacity, size):
        # Initialize segment tree with height = log₂(n) + 1
        self.size = size
        self.capacity = capacity
        self.tree_size = 2 * (2 ** ceil(log2(size))) - 1
        self.tree = [capacity] * self.tree_size
        self.real_size = size
        
    def update(self, pos, val):
        """Update the capacity at position pos by subtracting val"""
        # Find position in the leaf nodes
        idx = pos + self.size - 1
        
        # Update the capacity
        self.tree[idx] -= val
        
        # Update parent nodes
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] = max(self.tree[2 * idx + 1], self.tree[2 * idx + 2])
    
    def query(self, val):
        """Find the first bin that has capacity >= val"""
        # If the root node doesn't have enough capacity, return -1
        if self.tree[0] < val:
            return -1
        
        idx = 0
        while idx < self.size - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            # Check left child first
            if self.tree[left] >= val:
                idx = left
            else:
                idx = right
        
        # Convert tree index to bin index
        bin_idx = idx - (self.size - 1)
        
        # Check if it's within our real size
        if bin_idx < self.real_size:
            return bin_idx
        return -1
def harmonic_m(items, M=8):
    """
    Implementation of the HARMONIC_M bin packing algorithm as described in 
    "A Simple On-Line Bin-Packing Algorithm" by C. C. Lee and D. T. Lee.
    
    Parameters:
    - items: List of item sizes (each between 0 and 1)
    - M: Partition number (default: 12 as recommended in the paper)
    
    Returns:
    - List of bin fill levels
    """
    if not items:
        return []
    
    # Initialize bins for each type (1 to M)
    type_bins = [[] for _ in range(M)]
    
    # Count of filled bins for each type
    filled_bins_count = [0] * M
    
    # Function to determine item type based on size
    def get_type(size):
        if size > 1/2:  # Type 1: (1/2, 1]
            return 0
        
        for k in range(2, M):
            if size > 1/(k+1):  # Type k: (1/(k+1), 1/k]
                return k-1
        
        return M-1  # Type M: (0, 1/M]
    
    # Process each item
    bins = []  # Final bins to return (will contain fill levels)
    
    for item in items:
        item_type = get_type(item)
        
        # Add item to appropriate type bin
        type_bins[item_type].append(item)
        
        # Check if this type bin is filled
        if item_type < M-1:  # Types 1 to M-1
            k = item_type + 1  # k is 1-indexed in the paper
            if len(type_bins[item_type]) == k:
                # Add this filled bin to our result
                bins.append(sum(type_bins[item_type]))
                filled_bins_count[item_type] += 1
                # Reset this type bin
                type_bins[item_type] = []
    
    # Process any remaining partially filled bins
    for i in range(M):
        if type_bins[i]:
            bins.append(sum(type_bins[i]))
    
    return bins
def next_fit(items):
    bins = []
    current_bin = 0
    bins.append(0)
    for item in items:
        if bins[current_bin] + item <= 1:
            bins[current_bin] += item
        else:
            bins.append(item)
            current_bin += 1
    return bins

def first_fit(items):
    if not items:
        return []
    
    n = len(items)
    bins = [0]  # Start with one bin
    
    # Determine the initial segment tree size (power of 2)
    tree_size = 1
    while tree_size < n:
        tree_size *= 2
    
    # Initialize segment tree with remaining capacities (1.0 for each bin)
    seg_tree = SegmentTree(1.0, tree_size)
    remaining_capacities = [1.0]  # Track remaining capacity of each bin
    
    for item in items:
        if item > 1.0:  # Skip items larger than bin capacity
            continue
            
        # Find the first bin with enough capacity
        bin_idx = seg_tree.query(item)
        
        if bin_idx != -1 and bin_idx < len(bins):
            # Found a suitable bin
            bins[bin_idx] += item
            remaining_capacities[bin_idx] -= item
            seg_tree.update(bin_idx, item)
        else:
            # Create a new bin
            bins.append(item)
            remaining_capacities.append(1.0 - item)
            
            # If we've exceeded the segment tree size, we need to resize
            if len(bins) > tree_size:
                new_tree_size = tree_size * 2
                new_seg_tree = SegmentTree(1.0, new_tree_size)
                for i, cap in enumerate(remaining_capacities):
                    new_seg_tree.update(i, 1.0 - cap)
                seg_tree = new_seg_tree
                tree_size = new_tree_size
            else:
                seg_tree.update(len(bins) - 1, item)
    
    return bins

def first_fit_decreasing(items):
    items_sorted = sorted(items, reverse=True)
    return first_fit(items_sorted)

def evaluate_algorithm(algorithm, items):
    start_time = time.time()
    bins = algorithm(items)
    end_time = time.time()

    num_bins = len(bins)
    opt_lb = math.ceil(sum(items))  # Lower bound on optimal solution
    approx_ratio = num_bins / opt_lb if opt_lb > 0 else 0
    
    # Calculate waste and utilization statistics
    waste_per_bin = [1 - b for b in bins]
    avg_waste = sum(waste_per_bin) / num_bins if num_bins > 0 else 0
    std_waste = np.std(waste_per_bin) if num_bins > 0 else 0
    
    avg_util = sum(bins) / num_bins if num_bins > 0 else 0
    std_util = np.std(bins) if num_bins > 0 else 0
    
    # Calculate bin fill distribution (percentage of bins in each fill range)
    fill_ranges = defaultdict(int)
    for b in bins:
        if b < 0.6:
            fill_ranges["<60%"] += 1
        elif b < 0.8:
            fill_ranges["60-80%"] += 1
        elif b < 0.9:
            fill_ranges["80-90%"] += 1
        elif b < 0.95:
            fill_ranges["90-95%"] += 1
        else:
            fill_ranges["95-100%"] += 1
    
    fill_distribution = {k: (v/num_bins)*100 if num_bins > 0 else 0 for k, v in fill_ranges.items()}
    
    time_taken = end_time - start_time

    return {
        'num_bins': num_bins,
        'approx_ratio': approx_ratio,
        'avg_waste': avg_waste,
        'std_waste': std_waste,
        'avg_util': avg_util, 
        'std_util': std_util,
        'fill_distribution': fill_distribution,
        'time_taken': time_taken
    }

# Parameters
input_sizes = [2**i for i in range(0, 16)]
algorithms = {
    'Next Fit': next_fit,
    'First Fit': first_fit,
    'First Fit Decreasing': first_fit_decreasing,
    'Harmonic_8': harmonic_m,
    'LP-GroupKK': lp_bin_packing_with_grouping_and_kk  # Add the new algorithm
}
k = 10  # Monte Carlo runs per input size

# Initialize metrics dictionary
metrics = {
    'num_bins': {name: [] for name in algorithms},
    'approx_ratio': {name: [] for name in algorithms},
    'avg_waste': {name: [] for name in algorithms},
    'std_waste': {name: [] for name in algorithms},
    'avg_util': {name: [] for name in algorithms},
    'std_util': {name: [] for name in algorithms},
    'fill_distribution': {name: [{} for _ in range(len(input_sizes))] for name in algorithms},
    'time_taken': {name: [] for name in algorithms}
}

# Main loop with Monte Carlo simulation
for i, n in enumerate(input_sizes):
    print(f"Evaluating input size: {n}")
    for name, algo in algorithms.items():
        results = defaultdict(float)
        fill_dist_sum = defaultdict(float)
        
        for _ in range(k):
            items = [round(random.uniform(0.0001, 1.0), 4) for _ in range(n)]
            eval_results = evaluate_algorithm(algo, items)
            
            # Accumulate regular metrics
            for key in ['num_bins', 'approx_ratio', 'avg_waste', 'std_waste', 'avg_util', 'std_util', 'time_taken']:
                results[key] += eval_results[key]
            
            # Accumulate fill distribution
            for fill_range, percentage in eval_results['fill_distribution'].items():
                fill_dist_sum[fill_range] += percentage

        # Store averages for regular metrics
        for key in ['num_bins', 'approx_ratio', 'avg_waste', 'std_waste', 'avg_util', 'std_util', 'time_taken']:
            metrics[key][name].append(results[key] / k)
        
        # Store average fill distribution
        metrics['fill_distribution'][name][i] = {
            fill_range: percentage / k 
            for fill_range, percentage in fill_dist_sum.items()
        }

# Plotting functions
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(10, 6))

    markers = {
    'Next Fit': 's',        # square
    'First Fit': 'o',       # circle
    'First Fit Decreasing': 'D',  # diamond
    'Harmonic_8': '^'  ,
    'LP-GroupKK' : '*'# triangle up
    }

    colors = {
    'Next Fit': 'blue',
    'First Fit': 'green',
    'First Fit Decreasing': 'red',
    'Harmonic_8': 'purple',
    'LP-GroupKK' : 'black' 
    }

    for name in algorithms:
        plt.plot(
            input_sizes,
            metrics[metric_name][name],
            label=name,
            marker=markers[name],
            markersize=6,
            linewidth=2,
            color=colors[name]
        )

    plt.xscale("log", base=2)
    plt.xlabel("Input Size (log scale)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Input Size (Monte Carlo avg over {k} runs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric_name}_plot.png")
    plt.show()

def plot_fill_distribution(algorithm_name, size_idx):
    """Plot the bin fill distribution for a specific algorithm and input size"""
    plt.figure(figsize=(10, 6))
    
    fill_dist = metrics['fill_distribution'][algorithm_name][size_idx]
    fill_ranges = ["<60%", "60-80%", "80-90%", "90-95%", "95-100%"]
    
    # Ensure all ranges are present (even with zero values)
    values = [fill_dist.get(fill_range, 0) for fill_range in fill_ranges]
    
    plt.bar(fill_ranges, values, color='skyblue')
    plt.xlabel("Bin Fill Range")
    plt.ylabel("Percentage of Bins")
    plt.title(f"Bin Fill Distribution for {algorithm_name} (Input Size: {input_sizes[size_idx]})")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"fill_dist_{algorithm_name}_{input_sizes[size_idx]}.png")
    plt.show()

# Plot selected metrics (excluding std_util and std_waste)
plot_metric('num_bins', 'Average Number of Bins')
plot_metric('approx_ratio', 'Average Approximation Ratio')
plot_metric('avg_waste', 'Average Waste per Bin')
plot_metric('avg_util', 'Average Bin Utilization')
plot_metric('time_taken', 'Average Time Taken (s)')

# Plot fill distribution for each algorithm with the largest input size
for name in algorithms:
    plot_fill_distribution(name, -1)  # -1 to get the largest input size

# End of Code 