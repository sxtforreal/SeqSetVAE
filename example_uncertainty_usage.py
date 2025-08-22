#!/usr/bin/env python3
"""
Example: Using SeqSetVAE with Advanced VAE Feature Fusion and Uncertainty Quantification
"""

import torch
import torch.nn.functional as F
from model import SeqSetVAE
import numpy as np

def example_uncertainty_aware_inference():
    """
    Demonstrate how to use the uncertainty-aware SeqSetVAE for robust predictions
    """
    print("🚀 SeqSetVAE Advanced VAE Feature Fusion Example")
    
    # Model configuration
    model_config = {
        'input_dim': 100,
        'reduced_dim': 64,
        'latent_dim': 32,
        'levels': 2,
        'heads': 4,
        'm': 8,
        'beta': 1.0,
        'lr': 1e-4,
        'num_classes': 2,
        'ff_dim': 128,
        'transformer_heads': 4,
        'transformer_layers': 2,
        'vae_fusion_method': 'enhanced_concat',  # Try: attention, gated, uncertainty_weighted
        'estimate_uncertainty': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
    
    # Create model with different fusion methods
    fusion_methods = ['simple_concat', 'enhanced_concat', 'attention', 'gated', 'uncertainty_weighted']
    
    print("\n📊 Comparing Different VAE Fusion Methods:")
    print("=" * 60)
    
    for method in fusion_methods:
        print(f"\n🔧 Testing {method} method:")
        
        # Create model
        model_config['vae_fusion_method'] = method
        model = SeqSetVAE(**model_config)
        model.eval()
        
        # Create dummy batch data
        batch_size = 4
        seq_len = 10
        batch = {
            'var': torch.randn(batch_size, seq_len, model_config['input_dim']),
            'val': torch.ones(batch_size, seq_len, model_config['input_dim']),
            'minute': torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            'set_id': torch.zeros(batch_size, seq_len, dtype=torch.long),
            'label': torch.randint(0, 2, (batch_size,))
        }
        
        # Standard forward pass
        with torch.no_grad():
            logits, recon_loss, kl_loss = model(batch)
            
        # Get predictions and probabilities
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        print(f"   - Feature dimension: {model.vae_feature_dim}")
        print(f"   - Predictions: {predictions.tolist()}")
        print(f"   - Max probability: {torch.max(probs, dim=-1)[0].mean().item():.3f}")
        
        # Uncertainty-aware inference (if supported)
        if hasattr(model, 'forward_with_uncertainty'):
            try:
                logits_unc, uncertainties, recon_loss, kl_loss = model.forward_with_uncertainty(batch)
                
                print(f"   - Uncertainty estimation available:")
                for unc_type, unc_values in uncertainties.items():
                    if unc_values is not None:
                        print(f"     * {unc_type}: {unc_values.mean().item():.4f}")
                        
                # Calibrated predictions
                if hasattr(model, 'get_calibrated_predictions'):
                    calibrated_logits = model.get_calibrated_predictions(logits_unc)
                    calibrated_probs = F.softmax(calibrated_logits, dim=-1)
                    print(f"     * Temperature: {model.temperature.item():.3f}")
                    print(f"     * Calibrated max prob: {torch.max(calibrated_probs, dim=-1)[0].mean().item():.3f}")
                    
            except Exception as e:
                print(f"   - Uncertainty estimation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Comparison complete!")

def example_feature_analysis():
    """
    Analyze the different types of features extracted by each fusion method
    """
    print("\n🔍 VAE Feature Analysis:")
    print("=" * 40)
    
    # Create a model with enhanced concatenation
    model = SeqSetVAE(
        input_dim=50, reduced_dim=32, latent_dim=16, levels=1, heads=2, m=4,
        beta=1.0, lr=1e-4, num_classes=2, ff_dim=64, transformer_heads=2, transformer_layers=1,
        vae_fusion_method='enhanced_concat', estimate_uncertainty=True
    )
    
    # Create dummy latent variables
    batch_size = 2
    latent_dim = 16
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim) * 0.5  # Smaller variance
    
    print(f"Input shapes: mu={mu.shape}, logvar={logvar.shape}")
    
    # Test different fusion methods
    for method in ['simple_concat', 'enhanced_concat']:
        model.vae_fusion_method = method
        model._init_vae_fusion_components(latent_dim, method)
        
        fused_features = model._fuse_vae_features(mu, logvar)
        print(f"{method}: {fused_features.shape} -> {fused_features.shape[1]} features")
        
        if method == 'enhanced_concat':
            # Show uncertainty features
            uncertainty_features = model._compute_uncertainty_features(mu, logvar)
            print(f"  - Basic features (mu+std): {latent_dim*2}")
            print(f"  - Uncertainty features: {uncertainty_features.shape[1]}")
            print(f"    * Total variance: {uncertainty_features[0, 0].item():.4f}")
            print(f"    * Mean magnitude: {uncertainty_features[0, 1].item():.4f}")
            print(f"    * KL divergence: {uncertainty_features[0, 2].item():.4f}")
            print(f"    * Coefficient of variation: {uncertainty_features[0, 3].item():.4f}")
            print(f"    * Entropy approximation: {uncertainty_features[0, 4].item():.4f}")

def main():
    """Run all examples"""
    print("🎯 SeqSetVAE Advanced Feature Fusion Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_uncertainty_aware_inference()
    example_feature_analysis()
    
    print("\n🎉 All examples completed successfully!")
    print("\n💡 Usage Tips:")
    print("1. Use 'enhanced_concat' for best balance of performance and interpretability")
    print("2. Use 'attention' for adaptive feature weighting")
    print("3. Use 'gated' for maximum flexibility (but more parameters)")
    print("4. Always enable uncertainty estimation for robust predictions")
    print("5. Monitor temperature scaling for better calibration")

if __name__ == "__main__":
    main()