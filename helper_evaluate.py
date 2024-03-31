import torch
import torch.nn.functional as F


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features in data_loader:
           features = features.to(device)
           encoded, z_mean, z_log_var, decoded = model(features)
    # Assuming the loss_fn requires features and decoded tensors as inputs
           loss = loss_fn(decoded, features)
           num_examples += features.size(0)
           curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
