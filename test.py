import math

from matplotlib import pyplot as plt
import numpy as np

def speedup(k, p, L=12, Ld=12, n=1500, m=100):
    """
    Approximate theoretical speedup using attention-only compute (ignores feed-forward layers).

    Inputs:
        k  = number of encoder layers before downsampling (0 to L inclusive)
        p  = sequence length fraction after downsampling

    Returns:
        speedup factor
    """

    base_cost = L * n**2 + Ld * m * n
    new_cost = k * n**2 + (L - k) * (p * n)**2 + Ld * m * (p * n)

    return base_cost / new_cost

def find_p_for_speedup(target_speedup, k, L=12, Ld=12, n=1500, m=100):
    """
    Calculates the sequence length proportion 'p' required to achieve a given speedup.

    Inputs:
        target_speedup = desired speedup factor
        k              = number of encoder layers before downsampling (0 to L inclusive)
        L              = total number of encoder layers (default 12)
        Ld             = number of decoder layers (default 12)
        n              = original sequence length (default 1500)
        m              = decoder query length (default 100)

    Returns:
        The proportion p (0 < p <= 1), or None if the target speedup is unachievable,
        if inputs are invalid, or if no valid p is found.
    """
    epsilon = 1e-9  # Small tolerance for floating point comparisons

    # Validate inputs
    if not (isinstance(k, int) and 0 <= k <= L):
        return None
    if not (L > 0 and Ld > 0 and n > 0 and m > 0): # Parameters must be positive
        return None
    if not (isinstance(target_speedup, (int, float)) and target_speedup > 0):
        return None

    # Speedup generally must be >= 1.0.
    # p=1.0 results in speedup=1.0. Smaller p (if effective) increases speedup.
    if target_speedup < 1.0 - epsilon:
        return None # Speedup cannot be less than 1

    if abs(target_speedup - 1.0) < epsilon:
        return 1.0

    base_cost = float(L * n**2 + Ld * m * n)
    if abs(base_cost) < epsilon: # Avoid division by zero if base_cost is zero
        return None 

    # From speedup = base_cost / new_cost, target_new_cost = base_cost / target_speedup
    target_new_cost = base_cost / target_speedup

    # The new_cost is: k*n^2 + (L-k)*(p*n)^2 + Ld*m*(p*n)
    # Let x = p*n. The equation for x is:
    # (L-k)*x^2 + (Ld*m)*x + (k*n^2 - target_new_cost) = 0
    # This is a quadratic equation: A_quad*x^2 + B_quad*x + C_quad = 0
    A_quad = float(L - k)
    B_quad = float(Ld * m)
    C_quad = float(k * n**2 - target_new_cost)

    x_sol = None # This will store the solution for x = p*n

    if abs(A_quad) < epsilon:  # Linear case (k == L, so A_quad is 0)
        if abs(B_quad) < epsilon:
            # If B_quad is also 0 (Ld*m = 0), new_cost = k*n^2.
            # This implies target_new_cost = k*n^2.
            # If Ld*m=0, base_cost = L*n^2. If k=L, target_new_cost = L*n^2.
            # So target_speedup = (L*n^2)/(L*n^2) = 1.0. This is handled above.
            # If it reaches here with B_quad=0, it's an inconsistent state or Ld/m are zero.
            return None 
        
        # B_quad*x + C_quad = 0  => x = -C_quad / B_quad
        x_candidate = -C_quad / B_quad
        if x_candidate > epsilon:  # x = p*n must be positive
            x_sol = x_candidate
    else:  # Quadratic case (A_quad != 0)
        # Since 0 <= k <= L, L-k >= 0. As A_quad !=0, A_quad > 0 (k < L).
        discriminant = B_quad**2 - 4 * A_quad * C_quad
        
        if discriminant >= -epsilon:  # Check if discriminant is non-negative (within tolerance)
            if discriminant < 0: discriminant = 0.0 # Clamp for sqrt

            sqrt_discriminant = math.sqrt(discriminant)
            
            # We need the positive root for x = p*n.
            # With A_quad > 0 and B_quad >= 0:
            # Root 1: x1 = (-B_quad + sqrt_discriminant) / (2 * A_quad)
            # Root 2: x2 = (-B_quad - sqrt_discriminant) / (2 * A_quad) (this one is <=0)
            # For x1 to be positive, C_quad must be negative.
            if C_quad < -epsilon: 
                x_candidate = (-B_quad + sqrt_discriminant) / (2 * A_quad)
                if x_candidate > epsilon: # Ensure x is positive
                     x_sol = x_candidate
            # If C_quad >= 0 (and A_quad > 0), a positive x solution for p*n does not exist
            # (or x=0 if C_quad=0, leading to p=0, which we exclude by x_candidate > epsilon).

    if x_sol is not None:
        p = x_sol / n
        # We expect 0 < p <= 1.
        # target_speedup == 1.0 case returns p=1.0.
        # target_speedup > 1.0 should yield p < 1.0.
        if epsilon < p <= 1.0 + epsilon: # p must be positive and at most 1 (with tolerance)
            return min(p, 1.0)  # Clamp to 1.0 if slightly over due to precision
        else:
            # Calculated p is outside the valid (0, 1] range
            return None 
    else:
        # No valid positive x_sol (p*n) found
        return None

# print(speedup(4,0.67))

# speedup_values_to_plot = np.arange(1.1, 2.01, 0.1) # From 1.1 to 2.0, with a step of 0.1
speedup_values_to_plot = [1.575]
k_values = list(range(13)) # k from 0 to 12

plt.figure(figsize=(12, 8)) # Create a single figure
overall_max_y_val = 0 # To keep track of the max y value across all lines

for target_speedup_val in speedup_values_to_plot:
    discarded_proportions = []
    valid_k_for_plot = []

    for k_val in k_values:
        # Using default L, Ld, n, m from your function
        p_val = find_p_for_speedup(target_speedup_val, k_val)
        if p_val is not None:
            proportion_discarded = p_val
            discarded_proportions.append(proportion_discarded)
            valid_k_for_plot.append(k_val)
            if proportion_discarded > overall_max_y_val:
                overall_max_y_val = proportion_discarded
        else:
            # Optionally print if a specific k is not achievable for this speedup
            # print(f"Note: Target speedup of {target_speedup_val:.1f} not achievable for k={k_val}")
            pass

    if valid_k_for_plot: # Check if there are any points to plot for this speedup
        label = f"{target_speedup_val*100-100:.0f}% speedup"
        plt.plot(valid_k_for_plot, discarded_proportions, marker='o', linestyle='-', label=label)
    else:
        print(f"Could not find any valid 'p' for k in {k_values} to achieve {target_speedup_val:.1f} speedup.")

plt.xlabel("k (Number of encoder layers before downsampling)")
plt.ylabel("Proportion of sequence discarded (1 - p)")
plt.title("Sequence discard proportion vs. k for various speedups")
plt.xticks(k_values) # Ensure all k values are shown as ticks

# Adjust y-axis limit dynamically based on all plotted data
top_y_limit = max(0.1, overall_max_y_val * 1.1) if overall_max_y_val > 0 else 1.0
plt.ylim(bottom=0, top=top_y_limit)

plt.grid(True)
plt.legend() # Add legend to display labels
plt.show()