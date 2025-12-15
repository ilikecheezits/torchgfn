import torch
import torch.nn as nn
from week3 import SimpleMLP, get_input_tensor, sample_action

# --- Week 4 Requirements ---

# --- Day 1: Trajectory Collection ---
def collect_trajectory(model, max_length, target_string):
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
        input_tensor = get_input_tensor(current_sequence, target_string, max_length)
        input_batch = input_tensor.flatten().unsqueeze(0)
        
        # Get logits from the model
        logits = model(input_batch)
        trajectory["logits"].append(logits)
        
        # All actions are allowed
        mask = torch.ones(1, 4) # 4 bases
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
def calculate_loss(trajectory, log_Z):
    """
    Implements the Trajectory Balance loss function.
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

    # Reward function for "AAAA..." task
    if trajectory["sequence"] == "AAAA":
        log_R = torch.log(torch.tensor(10.0))
    else:
        log_R = torch.log(torch.tensor(1.0))
        
    # Assume P_B is uniform (log_Pb = 0)
    log_Pb = 0
    
    loss = (log_Z + log_Pf - log_R - log_Pb)**2
    return loss

# --- Day 3: Dummy Training ---
def train():
    """
    Runs the dummy training loop.
    """
    max_length = 4
    target_string = "...." # Not really used for this dummy task
    input_dim = max_length * 7
    output_dim = 4 # 4 bases
    
    model = SimpleMLP(input_dim, output_dim)
    log_Z = nn.Parameter(torch.tensor(0.0))
    
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_Z], lr=0.001)
    
    print("Starting training...")
    for epoch in range(2000):
        trajectory = collect_trajectory(model, max_length, target_string)
        loss = calculate_loss(trajectory, log_Z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Seq: {trajectory['sequence']}, log_Z: {log_Z.item():.4f}")
            
    print("\nFinished training.")
    
    # --- Gatekeeper Test ---
    print("\n--- Gatekeeper Test ---")
    aaaa_count = 0
    n_samples = 100
    for _ in range(n_samples):
        trajectory = collect_trajectory(model, max_length, target_string)
        if trajectory["sequence"] == "AAAA":
            aaaa_count += 1
            
    print(f"Agent generated 'AAAA' in {aaaa_count}/{n_samples} of cases.")
    if loss.item() < 1.0 and (aaaa_count / n_samples) > 0.9:
        print("Gatekeeper Test Passed!")
    else:
        print("Gatekeeper Test Failed.")


if __name__ == "__main__":
    train()
