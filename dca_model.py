import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with caching"""
    def __init__(self, dim, max_seq_length=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.register_buffer(
            'div_term',
            torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        )
        # Cache for different sequence lengths
        self.pe_cache = {}
        
    def _get_pe(self, seq_length):
        if seq_length in self.pe_cache:
            return self.pe_cache[seq_length]
            
        position = torch.arange(seq_length, device=self.div_term.device).unsqueeze(1)
        pe = torch.zeros(1, seq_length, self.dim, device=self.div_term.device)
        pe[0, :, 0::2] = torch.sin(position * self.div_term)
        pe[0, :, 1::2] = torch.cos(position * self.div_term)
        
        # Cache if sequence length is common
        if seq_length in [32, 64, 128, 256, 512, 1024, 2048]:
            self.pe_cache[seq_length] = pe
            
        return pe

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """Token embedding with weight tying support"""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = math.sqrt(dim)
        
        # Initialize using a normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
    def forward(self, x):
        return self.embedding(x) * self.scale

class DCATransformerPreTraining(nn.Module):
    """Complete DCA Transformer with embeddings and weight tying"""
    def __init__(self, vocab_size, dim, depth, num_heads=8, mlp_ratio=4, last_k=2, 
                 max_seq_length=2048, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, dim)
        self.pos_encoding = PositionalEncoding(dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            DCADecoderBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection (weight tied with embedding)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        self.out_proj.weight = self.token_embedding.embedding.weight
        
        # Layer output manager for memory efficiency
        self.layer_manager = LayerOutputManager(last_k)
        
        # Initialize using paper's strategy
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use scaled initialization for better gradient flow
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward(self, x):
        # Apply token embedding and positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Reset layer manager
        self.layer_manager = LayerOutputManager(self.layer_manager.k)
        self.layer_manager.add(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, self.layer_manager.get_outputs())
            self.layer_manager.add(x)
            
        # Project to vocabulary
        return self.out_proj(x)

class TextDataset(Dataset):
    """Dataset for language modeling with proper bounds checking and padding"""
    def __init__(self, data, seq_length, pad_token_id=0):
        self.data = data
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        # Ensure data length is multiple of sequence length
        self.effective_length = (len(data) // seq_length) * seq_length
        
    def __len__(self):
        return max(0, self.effective_length - 1)  # -1 for target shift
        
    def pad_sequence(self, seq, target_len):
        if len(seq) < target_len:
            padding = [self.pad_token_id] * (target_len - len(seq))
            seq = seq + padding
        return seq[:target_len]
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Get sequences with proper bounds checking
        x_start = idx
        x_end = min(idx + self.seq_length, len(self.data))
        y_start = idx + 1
        y_end = min(idx + self.seq_length + 1, len(self.data))
        
        x = self.pad_sequence(self.data[x_start:x_end], self.seq_length)
        y = self.pad_sequence(self.data[y_start:y_end], self.seq_length)
        
        return torch.LongTensor(x), torch.LongTensor(y)

def setup_distributed():
    """Setup for distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu)
        return True, rank, world_size, gpu
    return False, 0, 1, 0

def train_epoch(model, loader, optimizer, scheduler, grad_clip=1.0, device='cuda'):
    """Single training epoch with gradient clipping"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def train_model(model, train_data, val_data, batch_size=2048, epochs=10, 
                seq_length=128, lr=0.0016, warmup_steps=1000, grad_clip=1.0,
                device='cuda', fp16=True, gradient_accumulation_steps=1,
                checkpoint_dir='checkpoints'):
    """Complete training pipeline"""
    
    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()
    if is_distributed:
        model = DDP(model.to(gpu), device_ids=[gpu])
    else:
        model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_data, seq_length)
    val_dataset = TextDataset(val_data, seq_length)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // world_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        warmup_steps=warmup_steps,
        total_steps=len(train_loader) * epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            grad_clip=grad_clip, device=device
        )
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output.view(-1, output.size(-1)), 
                                          target.view(-1)).item()
            val_loss /= len(val_loader)
            val_ppl = math.exp(val_loss)
        
        if rank == 0:  # Only print on main process
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                torch.save(model.state_dict(), 'best_model.pt')
    
    return model
    """Dimension-independent weights version"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x, layer_outputs):
        # Simple learnable scalar weight for all dimensions
        stacked = torch.stack(layer_outputs, dim=1)
        weights = F.softmax(self.alpha * torch.ones(len(layer_outputs)), dim=0)
        output = torch.sum(stacked * weights.view(1, -1, 1), dim=1)
        return output

class GRNv2(nn.Module):
    """Dimension-dependent weights version"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Parameter(torch.ones(dim))
        
    def forward(self, x, layer_outputs):
        # Different weight for each dimension
        stacked = torch.stack(layer_outputs, dim=1)
        weights = F.softmax(self.weights.unsqueeze(0).expand(len(layer_outputs), -1), dim=0)
        output = torch.sum(stacked * weights.unsqueeze(0), dim=1)
        return output

class GRNv3(nn.Module):
    """Generalized Residual Network v3 component implementing input-dependent weights"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        
    def forward(self, x, layer_outputs):
        # Stack previous layer outputs
        stacked = torch.stack(layer_outputs, dim=1)  # [batch, num_layers, dim]
        
        # Generate input-dependent weights
        w = self.linear(stacked)  # [batch, num_layers, dim]
        w = self.act(w)
        
        # Apply learnable bias
        b = self.norm(stacked)
        
        # Combine with input-dependent weights
        combined = w + b
        
        # Weighted sum across layers
        output = torch.sum(combined, dim=1)  # [batch, dim]
        return output

class DCADecoderBlock(nn.Module):
    """Decoder block with DeepCrossAttention"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Three GRN instances for queries, keys, values
        self.grn_q = GRNv3(dim)
        self.grn_k = GRNv3(dim)
        self.grn_v = GRNv3(dim)
        
        # Attention components
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_norm = nn.LayerNorm(dim)
        self.attn_out = nn.Linear(dim, dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)
        
    def forward(self, x, layer_outputs):
        # Generate Q, K, V using GRNs
        q = self.grn_q(x, layer_outputs)
        k = self.grn_k(x, layer_outputs)
        v = self.grn_v(x, layer_outputs)
        
        # Reshape for multi-head attention
        B = x.shape[0]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        
        # Process through layers
        out = self.attn_out(out)
        out = x + self.attn_norm(out)
        out = out + self.ffn_norm(self.ffn(out))
        
        return out

class LayerOutputManager:
    """Efficient management of layer outputs maintaining only first and last-k layers"""
    def __init__(self, k):
        self.k = k
        self.outputs = []
        self.intermediate_sum = None
        self._max_stored = k + 2  # first + intermediate + k last layers
        
    def reset(self):
        """Reset state without reallocating"""
        self.outputs.clear()
        self.intermediate_sum = None
        
    def add(self, output):
        if len(self.outputs) <= self.k + 1:  # Keep all initial layers up to k+1
            self.outputs.append(output)
        else:
            # Keep first layer, sum of intermediate layers, and last k layers
            if len(self.outputs) == self.k + 2:  # First time exceeding k+1
                intermediate_sum = self.outputs[1]
            else:
                intermediate_sum = self.outputs[1] + output
            
            self.outputs = [
                self.outputs[0],  # First layer
                intermediate_sum,  # Sum of intermediate layers
                *self.outputs[-self.k:]  # Last k layers
            ]
    
    def get_outputs(self):
        return self.outputs

class DCATransformer(nn.Module):
    """Complete transformer architecture with DeepCrossAttention"""
    def __init__(self, dim, depth, vocab_size, num_heads=8, mlp_ratio=4, last_k=2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DCADecoderBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        layer_outputs = [x]
        
        for layer in self.layers:
            x = layer(x, layer_outputs)
            layer_outputs.append(x)
            
        return x

def create_optimizer_and_scheduler(model, warmup_steps=1000, total_steps=500000):
    """Creates optimizer and learning rate scheduler as specified in the paper"""
    optimizer = AdamW(
        model.parameters(),
        lr=0.0016,  # Base learning rate from paper
        betas=(0.9, 0.98),
        weight_decay=0.1
    )
    
    # Learning rate scheduler with warmup and inverse square root decay
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return (warmup_steps / float(max(1, step))) ** 0.5
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def calculate_perplexity(logits, targets, ignore_index=-100):
    """Calculates perplexity for language modeling"""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                          targets.view(-1), 
                          ignore_index=ignore_index,
                          reduction='mean')
    return torch.exp(loss)