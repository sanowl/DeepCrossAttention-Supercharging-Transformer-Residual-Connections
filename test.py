import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Assuming the DCA implementation is in a file called dca_model.py
from dca_model import DCATransformerPreTraining, TextDataset

def create_synthetic_data(vocab_size=1000, seq_length=128, num_sequences=1000):
    """Create synthetic data for testing"""
    # Create random sequences
    data = np.random.randint(0, vocab_size, size=num_sequences * seq_length)
    
    # Split into train and validation
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    return train_data, val_data

def plot_training_progress(losses, perplexities):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    
    ax2.plot(perplexities)
    ax2.set_title('Validation Perplexity')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def test_dca():
    # Model hyperparameters
    vocab_size = 1000
    dim = 256
    depth = 6
    num_heads = 8
    seq_length = 128
    batch_size = 32
    
    # Create synthetic data
    print("Creating synthetic data...")
    train_data, val_data = create_synthetic_data(
        vocab_size=vocab_size,
        seq_length=seq_length
    )
    
    # Create datasets
    train_dataset = TextDataset(train_data, seq_length)
    val_dataset = TextDataset(val_data, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = DCATransformerPreTraining(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        num_heads=num_heads
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Training setup
    epochs = 5
    total_steps = len(train_loader) * epochs
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        warmup_steps=100,
        total_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_perplexities = []
    
    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += torch.nn.functional.cross_entropy(
                        output.view(-1, output.size(-1)),
                        target.view(-1)
                    ).item()
            
            val_loss /= len(val_loader)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            val_perplexities.append(val_ppl)
            
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
                  f"Val PPL: {val_ppl:.2f}")
        
        # Plot training progress
        plot_training_progress(train_losses, val_perplexities)
        
        # Save model
        torch.save(model.state_dict(), 'dca_test_model.pt')
        print("Training completed successfully!")
        
        # Test model generation
        print("\nTesting model generation:")
        model.eval()
        with torch.no_grad():
            # Generate starting sequence
            start_seq = torch.randint(0, vocab_size, (1, 10)).to(device)
            output = model(start_seq)
            next_token = torch.argmax(output[:, -1, :], dim=-1)
            print(f"Generated token: {next_token.item()}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    test_dca