import torch
from gfn.model import (
    SimpleMLP,
    get_input_tensor,
    sample_action,
    seq_to_onehot,
    target_to_onehot,
)

def test_seq_to_onehot():
    seq = "AG"
    max_length = 4
    one_hot = seq_to_onehot(seq, max_length)
    expected = torch.zeros(max_length, 4)
    expected[0, 0] = 1
    expected[1, 3] = 1
    assert torch.equal(one_hot, expected)

def test_target_to_onehot():
    target = "(.))"
    max_length = 4
    one_hot = target_to_onehot(target, max_length)
    expected = torch.zeros(max_length, 3)
    expected[0, 0] = 1
    expected[1, 2] = 1
    expected[2, 1] = 1
    expected[3, 1] = 1
    assert torch.equal(one_hot, expected)

def test_get_input_tensor():
    seq = "A"
    target = "(."
    max_length = 3
    input_tensor = get_input_tensor(seq, target, max_length)
    assert input_tensor.shape == (max_length, 7)
    
    seq_one_hot = seq_to_onehot(seq, max_length)
    target_one_hot = target_to_onehot(target, max_length)
    expected = torch.cat([seq_one_hot, target_one_hot], dim=1)
    
    assert torch.equal(input_tensor, expected)

def test_simple_mlp():
    max_length = 10
    input_dim = max_length * 7
    output_dim = 5  # 4 bases + 1 exit action
    model = SimpleMLP(input_dim, output_dim)
    
    seq = "ACGU"
    target = "((..))"
    input_tensor = get_input_tensor(seq, target, max_length)
    
    # The model expects a batch
    input_batch = input_tensor.flatten().unsqueeze(0)
    
    logits = model(input_batch)
    assert logits.shape == (1, output_dim)

def test_sample_action():
    logits = torch.tensor([[1.0, 1.0, 10.0, 1.0, 1.0]])
    mask = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32)
    
    action = sample_action(logits, mask)
    assert action.item() == 2

    mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.float32)
    action = sample_action(logits, mask)
    assert action.item() != 2
