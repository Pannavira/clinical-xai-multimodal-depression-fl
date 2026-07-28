import os
import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .visual_encoder import VisualEncoder


class MultimodalDepressionClassifier(nn.Module):
    """
    PyTorch Multimodal Depression Classifier supporting 7 Baseline Strategies:
    - Unimodal: 'text_only', 'audio_only', 'visual_only'
    - Bimodal: 'text_audio', 'text_visual', 'audio_visual'
    - Full Multimodal: 'late_fusion' (Text + Audio + Visual)
    """

    VALID_STRATEGIES = [
        'text_only', 'audio_only', 'visual_only',
        'text_audio', 'text_visual', 'audio_visual',
        'late_fusion'
    ]

    def __init__(self, config=None, **kwargs):
        super(MultimodalDepressionClassifier, self).__init__()
        
        # Parse parameters from config dictionary or kwargs
        if config is not None:
            feat_cfg = config.get('features', {})
            model_cfg = config.get('model', {})
            
            text_dim = feat_cfg.get('text_dim', 768)
            audio_dim = feat_cfg.get('audio_dim', 128)
            visual_dim = feat_cfg.get('visual_dim', 178)
            
            hidden_dim = model_cfg.get('hidden_dim', 128)
            fusion_dim = model_cfg.get('fusion_dim', 128)
            dropout = model_cfg.get('dropout', 0.3)
            fusion_strategy = model_cfg.get('fusion_strategy', 'late_fusion')
        else:
            text_dim = kwargs.get('text_dim', 768)
            audio_dim = kwargs.get('audio_dim', 128)
            visual_dim = kwargs.get('visual_dim', 178)
            hidden_dim = kwargs.get('hidden_dim', 128)
            fusion_dim = kwargs.get('fusion_dim', 128)
            dropout = kwargs.get('dropout', 0.3)
            fusion_strategy = kwargs.get('fusion_strategy', 'late_fusion')

        if fusion_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid fusion_strategy '{fusion_strategy}'. Must be one of {self.VALID_STRATEGIES}")

        self.fusion_strategy = fusion_strategy
        self.hidden_dim = hidden_dim

        # Instantiate modal encoders conditionally
        self.text_encoder = None
        self.audio_encoder = None
        self.visual_encoder = None

        active_modalities_count = 0

        if 'text' in fusion_strategy or fusion_strategy == 'late_fusion':
            self.text_encoder = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
            active_modalities_count += 1

        if 'audio' in fusion_strategy or fusion_strategy == 'late_fusion':
            self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
            active_modalities_count += 1

        if 'visual' in fusion_strategy or fusion_strategy == 'late_fusion':
            self.visual_encoder = VisualEncoder(input_dim=visual_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
            active_modalities_count += 1

        concat_dim = hidden_dim * active_modalities_count

        # Late Fusion & Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, text=None, audio=None, visual=None):
        """
        Forward pass. Accepts individual modality tensors or a batch dictionary.
        
        Args:
            text (torch.Tensor or dict): (B, 768) tensor or batch dictionary containing keys 'text', 'audio', 'visual'.
            audio (torch.Tensor, optional): (B, 128) tensor.
            visual (torch.Tensor, optional): (B, 178) tensor.
            
        Returns:
            torch.Tensor: Binary logits of shape (B, 1).
        """
        # Support passing a dictionary batch
        if isinstance(text, dict):
            batch_dict = text
            text_tensor = batch_dict.get('text', None)
            audio_tensor = batch_dict.get('audio', None)
            visual_tensor = batch_dict.get('visual', None)
        else:
            text_tensor = text
            audio_tensor = audio
            visual_tensor = visual

        representations = []

        if self.text_encoder is not None:
            if text_tensor is None:
                raise ValueError(f"Strategy '{self.fusion_strategy}' requires 'text' tensor input.")
            representations.append(self.text_encoder(text_tensor))

        if self.audio_encoder is not None:
            if audio_tensor is None:
                raise ValueError(f"Strategy '{self.fusion_strategy}' requires 'audio' tensor input.")
            representations.append(self.audio_encoder(audio_tensor))

        if self.visual_encoder is not None:
            if visual_tensor is None:
                raise ValueError(f"Strategy '{self.fusion_strategy}' requires 'visual' tensor input.")
            representations.append(self.visual_encoder(visual_tensor))

        # Concatenate latent representations along feature dimension
        if len(representations) == 1:
            fused_representation = representations[0]
        else:
            fused_representation = torch.cat(representations, dim=1)

        # Output raw logits (B, 1) for BCEWithLogitsLoss
        logits = self.classifier(fused_representation)
        return logits


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from modeling.dataset_loader import get_multimodal_dataloaders, load_config

    print("=" * 60)
    print("   TESTING MULTIMODAL DEPRESSION CLASSIFIER (7 STRATEGIES)")
    print("=" * 60)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg_file = os.path.join(base_dir, 'modeling', 'configs', 'centralized_baseline_config.yaml')
    config = load_config(cfg_file)

    train_loader, _, _ = get_multimodal_dataloaders(config, base_dir=base_dir)
    batch = next(iter(train_loader))

    for strategy in MultimodalDepressionClassifier.VALID_STRATEGIES:
        test_cfg = config.copy()
        test_cfg['model']['fusion_strategy'] = strategy
        
        model = MultimodalDepressionClassifier(config=test_cfg)
        model.eval()
        
        with torch.no_grad():
            out_logits = model(batch)
            
        print(f"Strategy: {strategy:15s} | Output Logits Shape: {out_logits.shape} | Sample Logit: {out_logits[0].item():.4f}")

    print("\nAll 7 strategy architectures forward pass test PASSED!")
