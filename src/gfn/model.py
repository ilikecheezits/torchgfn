import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def sample_action(logits, mask, temperature=1.0):
    masked_logits = logits + (mask - 1) * 1e9
    probs = torch.nn.functional.softmax(masked_logits / temperature, dim=-1)
    action = torch.multinomial(probs, 1)
    return action

BASE_TO_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3, "": -1}
TARGET_TO_INDEX = {"(": 0, ")": 1, ".": 2, "": -1}

def seq_to_onehot(sequence_string, max_length):
    one_hot = torch.zeros(max_length, 4)
    for i, base in enumerate(sequence_string):
        if base in BASE_TO_INDEX and BASE_TO_INDEX[base] != -1:
            one_hot[i, BASE_TO_INDEX[base]] = 1
    return one_hot

def target_to_onehot(target_string, max_length):
    one_hot = torch.zeros(max_length, 3)
    for i, char in enumerate(target_string):
        if char in TARGET_TO_INDEX and TARGET_TO_INDEX[char] != -1:
            one_hot[i, TARGET_TO_INDEX[char]] = 1
    return one_hot

def get_input_tensor(sequence_string, target_string, max_length, t):
    seq_one_hot = seq_to_onehot(sequence_string, max_length)
    target_one_hot = target_to_onehot(target_string, max_length)
    t_tensor = torch.tensor([t], dtype=torch.float32)
    return torch.cat([seq_one_hot.flatten(), target_one_hot.flatten(), t_tensor])