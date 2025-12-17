import torch
import torch.nn as nn
from gfn.model import SimpleMLP, get_input_tensor, sample_action
from gfn.constraints import parse_target, get_valid_mask
from oracle import RNAOracle
import numpy as np

# --- Week 4 Requirements ---

# --- Day 1: Trajectory Collection ---
def collect_trajectory(model, max_length, target_string, pair_map):
    """
    Runs the agent in a loop to generate a single trajectory.
    Stores states, actions, masks, and logits for every step.
    """
    trajectory = {
        "states": [],
        "actions": [],
        "masks": [],
        "logits": [],
        "sequence": ""
    }
    
    current_sequence = ""
    
    for t in range(max_length):
        # State is the current sequence
        trajectory["states"].append(current_sequence)
        
        # Get input for the model
        input_tensor = get_input_tensor(current_sequence, target_string, max_length, t)
        input_batch = input_tensor.unsqueeze(0)
        
        # Get logits from the model
        logits = model(input_batch)
        trajectory["logits"].append(logits)
        
        # --- Apply constraints ---
        mask_list = get_valid_mask(t, current_sequence, pair_map)
        mask = torch.tensor(mask_list).unsqueeze(0)
        trajectory["masks"].append(mask)

        # Sample action
        action = sample_action(logits, mask)
        action_idx = action.item()
        trajectory["actions"].append(action_idx)

        # Update sequence
        bases = ["A", "U", "C", "G"]
        current_sequence += bases[action_idx]

    trajectory["sequence"] = current_sequence
    return trajectory

# --- Day 2: Loss Function (Trajectory Balance) ---
def calculate_loss(trajectory, log_Z, target_string, beta):
    """
    Implements the Trajectory Balance loss function in log-space.
    loss = (log_Z + sum(log_Pf) - log_R - sum(log_Pb))**2
    """
    log_Pf = 0
    for i in range(len(trajectory["actions"])):
        logits = trajectory["logits"][i]
        mask = trajectory["masks"][i]
        action = trajectory["actions"][i]
        
        masked_logits = logits + (mask - 1) * 1e9
        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
        log_Pf += log_probs[0, action]

    # --- Week 6 Integration ---
    # Convert sequence to one-hot JAX array
    onehot_seq_jax = RNAOracle.seq_to_onehot(trajectory["sequence"])
    
    # Reshape for batch dimension
    onehot_seq_jax_batch = onehot_seq_jax.reshape(1, *onehot_seq_jax.shape)
    
    # Calculate defect using RNAOracle
    defect_jax = RNAOracle.compute_defect(onehot_seq_jax_batch, [target_string])
    
    # Convert defect from JAX array to PyTorch tensor
    defect_numpy = np.array(defect_jax)
    defect_tensor = torch.from_numpy(defect_numpy)
    
    # New log_R calculation
    log_R = -beta * defect_tensor
        
    # Assume P_B is uniform (log_Pb = 0)
    log_Pb = 0
    
    loss = (log_Z + log_Pf - log_R.sum() - log_Pb)**2
    return loss

# --- Day 3: Dummy Training ---
def train():
    """
    Runs the training loop with RNAOracle integration and constraints.
    """
    max_length = 8
    target_string = "((....))"
    pair_map = parse_target(target_string)
    input_dim = max_length * 7 + 1
    output_dim = 4 # 4 bases
    beta = 10.0
    
    model = SimpleMLP(input_dim, output_dim)
    log_Z = nn.Parameter(torch.tensor(0.0))
    
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_Z], lr=0.0001)
    
    print("Starting training with log-space stability...")
    for epoch in range(2000):
        trajectory = collect_trajectory(model, max_length, target_string, pair_map)
        loss = calculate_loss(trajectory, log_Z, target_string, beta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Re-calculate defect for logging
            onehot_seq_jax = RNAOracle.seq_to_onehot(trajectory["sequence"])
            onehot_seq_jax_batch = onehot_seq_jax.reshape(1, *onehot_seq_jax.shape)
            defect_jax = RNAOracle.compute_defect(onehot_seq_jax_batch, [target_string])
            defect = np.array(defect_jax)[0]
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Seq: {trajectory['sequence']}, Defect: {defect:.4f}, log_Z: {log_Z.item():.4f}")
            
    print("\nFinished training.")
    
    # --- Gatekeeper Test ---
    print("\n--- Gatekeeper Test ---")
    defects = []
    n_samples = 100
    for _ in range(n_samples):
        trajectory = collect_trajectory(model, max_length, target_string, pair_map)
        onehot_seq_jax = RNAOracle.seq_to_onehot(trajectory["sequence"])
        onehot_seq_jax_batch = onehot_seq_jax.reshape(1, *onehot_seq_jax.shape)
        defect_jax = RNAOracle.compute_defect(onehot_seq_jax_batch, [target_string])
        defects.append(np.array(defect_jax)[0])

    avg_defect = np.mean(defects)
    print(f"Average defect over {n_samples} samples: {avg_defect:.4f}")
    if avg_defect < 1.0: # Arbitrary threshold for success
        print("Gatekeeper Test Passed!")
    else:
        print("Gatekeeper Test Failed.")


if __name__ == "__main__":
    train()
