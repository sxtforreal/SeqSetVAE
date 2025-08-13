#!/usr/bin/env python3
"""
Optimized classifier head design for SeqSetVAE fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedClassifierHead(nn.Module):
    """
    Optimized classification head with configurable architecture
    Designed specifically for fine-tuning scenarios
    """
    
    def __init__(self, input_dim, num_classes, layers=None, dropout=0.1):
        super().__init__()
        
        if layers is None:
            # Default: simpler architecture for fine-tuning
            layers = [input_dim // 2, input_dim // 4]
        
        # Build classification head dynamically
        modules = []
        prev_dim = input_dim
        
        for layer_dim in layers:
            modules.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # Use BatchNorm for better training stability
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = layer_dim
        
        # Final classification layer
        modules.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*modules)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.classifier(x)


class SimpleFeatureExtractor(nn.Module):
    """
    Simplified feature extractor without complex fusion
    """
    
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.Tanh(),
            nn.Linear(latent_dim // 4, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, h_seq):
        """
        Extract features using simple attention pooling
        
        Args:
            h_seq: [B, S, D] sequence of hidden states
            
        Returns:
            [B, D] pooled features
        """
        B, S, D = h_seq.shape
        
        # Compute attention weights
        attention_weights = self.attention(h_seq)  # [B, S, 1]
        
        # Weighted average pooling
        pooled_features = torch.sum(h_seq * attention_weights, dim=1)  # [B, D]
        
        return pooled_features


class LightweightClassifierHead(nn.Module):
    """
    Ultra-lightweight classifier head for fine-tuning
    """
    
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.xavier_uniform_(self.classifier[3].weight)
    
    def forward(self, x):
        return self.classifier(x)


# Configuration for different classifier head types
CLASSIFIER_CONFIGS = {
    "lightweight": {
        "type": "lightweight",
        "layers": None,
        "dropout": 0.1,
        "description": "2-layer network, minimal parameters"
    },
    "standard": {
        "type": "standard", 
        "layers": [128, 64],
        "dropout": 0.15,
        "description": "3-layer network, balanced complexity"
    },
    "deep": {
        "type": "deep",
        "layers": [256, 128, 64],
        "dropout": 0.2,
        "description": "4-layer network, higher capacity"
    }
}


def create_classifier_head(config_name, input_dim, num_classes):
    """
    Factory function to create classifier head based on configuration
    
    Args:
        config_name: One of "lightweight", "standard", "deep"
        input_dim: Input feature dimension
        num_classes: Number of output classes
        
    Returns:
        Classifier head module
    """
    if config_name not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CLASSIFIER_CONFIGS.keys())}")
    
    config = CLASSIFIER_CONFIGS[config_name]
    
    if config["type"] == "lightweight":
        return LightweightClassifierHead(input_dim, num_classes, config["dropout"])
    else:
        return OptimizedClassifierHead(input_dim, num_classes, config["layers"], config["dropout"])


# Example usage and testing
if __name__ == "__main__":
    # Test different classifier heads
    input_dim = 256
    num_classes = 2
    batch_size = 8
    seq_len = 10
    
    print("Testing different classifier head configurations:")
    print("=" * 50)
    
    for config_name in CLASSIFIER_CONFIGS:
        print(f"\n{config_name.upper()} configuration:")
        print(f"Description: {CLASSIFIER_CONFIGS[config_name]['description']}")
        
        # Create classifier
        classifier = create_classifier_head(config_name, input_dim, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in classifier.parameters())
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        test_input = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            output = classifier(test_input)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")