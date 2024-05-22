import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  random_split, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import TransformerRegressor

now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")

x_input = np.load("binned_matrix/remap_binned_combined.npy")
x_input = torch.tensor(x_input).float()
y_input = np.load("y_input_normed.npy")

assert(x_input.shape[0] == y_input.shape[0])

y_input_cut_tensor = torch.tensor(y_input)
y = y_input_cut_tensor.unsqueeze(1).float()
src = x_input.permute(0, 2, 1)

pre_mask = y_input_cut_tensor != 0
total_y = torch.log(y_input_cut_tensor[pre_mask])

# ChatGPT helped + custom revised section A. "START"
'''
This part is to assign weights on loss calculations to balance the different density of samples
Weigh more loss from lowly populated values    
'''
num_bins=25
bins = np.linspace(min(total_y) - 1, max(total_y) + 1, num_bins)
bin_indices = np.digitize(total_y, bins, right=True)
bin_counts = pd.Series(bin_indices).value_counts().sort_index()
bin_counts = bin_counts + 1
bin_weights = 1 / bin_counts

def compute_continuous_sample_weights(y, bin_weights, num_bins=num_bins):
    _bin_indices = np.digitize(y, bins, right=True)
    sample_weights = bin_weights[_bin_indices].values
    return sample_weights
# ChatGPT helped + custom revised section B. "END"

# Hyperparameters
input_dim = src.shape[2]  # Example input dimension
model_dim = 64  # Dimension of the transformer model before feeding into positional embedding
num_heads = 8  # Number of attention heads
num_layers = 3  # Number of transformer encoder and decoder layers
ff_dim = 256  # Dimension of the feedforward fully connected neural network
output_dim = 1  # Single float output

seq_length = src.shape[1]
dataset = TensorDataset(src, y)

# Note
# The backbone script about the batch training through dataloader has been cited from
# https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/


for i in [0, 1, 2]: # different sub sample seeds
    # PRE-PROCESSING PART: model and data set up
    model = TransformerRegressor(input_dim, model_dim, num_heads, num_layers, ff_dim, output_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    criterion = nn.MSELoss(reduction='none') # this is for obtaining individual loss values with rediction = 'none' for sample weight loss.
    optimizer = optim.Adam(model.parameters(), lr=0.00002) # tesed with 0.0001, 0.00001, 0.0002
    writer = SummaryWriter(log_dir='./tensorboard_logs')

    train_size = int(0.8 * len(dataset)) 
    val_size = int(0.1 * len(dataset)) 
    test_size = len(dataset) - train_size - val_size  
    generator1 = torch.Generator().manual_seed(i)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    patience = 10 # max epoch rounds with no improvement
    epochs_no_improve = 0

    num_epochs = 200
    global_step = 0
    best_train_loss = float("Inf")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        # PART A: Train
        for batch_src, batch_y in train_loader:
            output_mask = batch_y.squeeze() != 0  # Create a mask for non-zero output entries
            input_mask = batch_src.sum(dim=[1, 2]) != 0 # Create a mask for non-zero x entries
            combined_mask = output_mask & input_mask

            batch_src = batch_src[combined_mask]
            batch_y = torch.log(batch_y[combined_mask])
            batch_src, batch_y = batch_src.to(device), batch_y.to(device)

            if batch_src.size(0) == 0:
                continue
            
            # ChatGPT helped + custom revised section B. "START" 
            # Calculating weighted loss
            weights = torch.tensor(compute_continuous_sample_weights(batch_y[0], bin_weights))
            optimizer.zero_grad()
            outputs = model(batch_src)
            loss = criterion(outputs, batch_y)
            weighted_loss = loss * weights
            weighted_loss = weighted_loss.mean() 
            weighted_loss.backward() 
            # ChatGPT helped + custom revised section B. "END"
            optimizer.step()
            
            writer.add_scalar('Training Loss', weighted_loss.item(), global_step)
            global_step += 1

            if weighted_loss.item() < best_train_loss:
                best_train_loss = weighted_loss.item()
                print(f'Detected best model with training loss: {best_train_loss:.4f} at epoch {epoch + 1}, step {global_step}')
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {weighted_loss.item():.4f}')

        # PART B: VALIDATION + EARLY STOPPING
        model.eval()
        val_loss = 0
        with torch.no_grad():
            total_valid_val = 0
            for batch_src, batch_y in val_loader:
                target_mask = batch_y.squeeze() != 0
                input_mask = batch_src.sum(dim=[1, 2]) != 0
                combined_mask = target_mask & input_mask
                
                batch_src = batch_src[combined_mask]
                batch_y = torch.log(batch_y[combined_mask])
                batch_src, batch_y = batch_src.to(device), batch_y.to(device)
                
                if batch_src.size(0) == 0:
                    continue
            
                outputs = model(batch_src)
                loss = criterion(outputs, batch_y)
                val_loss += loss.sum() # becuase I set criterion (none), which gives individual output
                total_valid_val += combined_mask.sum()
        
        val_loss /= total_valid_val
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_transformer_regressor_condition5_{dt_string}_cv{i}.pth')
            print(f'Saved best model with validation loss: {best_val_loss:.4f} at epoch {epoch + 1}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    model.load_state_dict(torch.load(f'best_transformer_regressor_condition5_{dt_string}_cv{i}.pth'))

    # PART C: TEST ON HELD OUT DATASET AND SAVE
    model.eval()
    test_loss = 0
    predictions = []
    real_values = []
    x_values = []

    with torch.no_grad():
        total_valid = 0
        for batch_src, batch_y in test_loader:
            output_mask = batch_y.squeeze() != 0  # Create a mask for non-zero output entries
            input_mask = batch_src.sum(dim=[1, 2]) != 0 # Create a mask for non-zero x entries
            combined_mask = output_mask & input_mask

            batch_src = batch_src[combined_mask]
            batch_y = torch.log(batch_y[combined_mask])
            batch_src, batch_y = batch_src.to(device), batch_y.to(device)
            
            outputs = model(batch_src)
            loss = criterion(outputs, batch_y)
            test_loss += loss.sum()
            total_valid += combined_mask.sum()

            predictions.extend(outputs.cpu().numpy())
            real_values.extend(batch_y.cpu().numpy())
            x_values.extend(batch_src.cpu().numpy())
            
    test_loss /= total_valid
    print(f'Test Loss: {test_loss:.4f}')
    current_time = datetime.now()

    writer.close()

    predictions = np.array(predictions)
    real_values = np.array(real_values)
    x_values = np.array(x_values)

    np.save(f"y_val_prediction_{current_time}.npy", predictions)
    np.save(f"y_val_real_{current_time}.npy", real_values)
    np.save(f"x_val_{current_time}.npy", x_values)