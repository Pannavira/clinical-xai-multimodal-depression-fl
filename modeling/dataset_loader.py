import os
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_config(config_path_or_dict):
    """Utility function to load configuration dictionary from file or dict."""
    if isinstance(config_path_or_dict, str):
        with open(config_path_or_dict, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    else:
        raise ValueError("config_path_or_dict must be a file path string or a dictionary.")


class MultimodalDepressionDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Depression Detection (Text, Audio, Visual).
    
    Reads participant split indices and loads features from preprocessing_output.
    - Text: 768-d BERT embedding array (.npy)
    - Audio: 128-d MFCC feature array (.npy)
    - Visual: 179-d tabular visual features (from visual_features.csv) or 31-d sequence pooled (.npy)
    - Label: Binary classification target (0 = non-depressed, 1 = depressed)
    """

    def __init__(self, index_df, config, base_dir=".", scaler_stats=None):
        """
        Args:
            index_df (pd.DataFrame): DataFrame containing participant split index.
            config (dict): Configuration dictionary loaded from centralized_baseline_config.yaml.
            base_dir (str): Base root directory of the workspace.
            scaler_stats (dict, optional): Mean and Std dictionary for feature standardization.
        """
        self.config = config
        self.base_dir = base_dir
        self.df = index_df.copy().reset_index(drop=True)
        self.scaler_stats = scaler_stats
        
        # Standardize participant_id to int
        self.df['participant_id'] = self.df['participant_id'].astype(int)
        
        # Load binary labels safely via participant mapping dictionary
        labels_path = os.path.join(self.base_dir, config['data']['detailed_labels_path'])
        if os.path.exists(labels_path):
            df_labels = pd.read_csv(labels_path)
            df_labels['Participant'] = df_labels['Participant'].astype(int)
            label_dict = df_labels.set_index('Participant')['Depression_label'].to_dict()
            
            mapped_labels = self.df['participant_id'].map(label_dict)
            if mapped_labels.isnull().any():
                self.df['binary_label'] = mapped_labels.fillna(
                    (self.df['depression_label'] >= 10).astype(int)
                ).astype(int)
            else:
                self.df['binary_label'] = mapped_labels.astype(int)
        else:
            # Fallback threshold >= 10
            self.df['binary_label'] = (self.df['depression_label'] >= 10).astype(int)

        # Pre-load visual tabular features if using tabular visual source
        self.visual_source = config['features'].get('visual_source', 'tabular')
        self.visual_lookup = {}
        
        if self.visual_source == 'tabular':
            visual_csv_path = os.path.join(self.base_dir, config['data']['visual_features_csv'])
            if os.path.exists(visual_csv_path):
                df_vis = pd.read_csv(visual_csv_path)
                df_vis['participant_id'] = df_vis['participant_id'].astype(int)
                
                # Exclude participant_id and non-numeric metadata columns (e.g. quality_flag)
                numeric_cols = df_vis.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if c != 'participant_id']
                for _, row in df_vis.iterrows():
                    pid = int(row['participant_id'])
                    feat_vector = row[feature_cols].values.astype(np.float32)
                    # Replace NaNs/Infs if any with 0.0
                    feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)
                    self.visual_lookup[pid] = feat_vector
            else:
                raise FileNotFoundError(f"Visual tabular CSV not found at {visual_csv_path}")
                
        elif self.visual_source == 'sequence_pooled':
            seq_npy_path = os.path.join(self.base_dir, config['data']['visual_sequence_npy'])
            seq_idx_path = os.path.join(self.base_dir, config['data']['visual_sequence_index_csv'])
            
            if os.path.exists(seq_npy_path) and os.path.exists(seq_idx_path):
                seq_data = np.load(seq_npy_path) # Shape: (275, 300, 31)
                seq_idx_df = pd.read_csv(seq_idx_path)
                seq_idx_df['participant_id'] = seq_idx_df['participant_id'].astype(int)
                
                for _, row in seq_idx_df.iterrows():
                    row_i = int(row['row_index'])
                    pid = int(row['participant_id'])
                    # Mean pool along sequence dimension (300, 31) -> (31,)
                    pooled_feat = np.mean(seq_data[row_i], axis=0).astype(np.float32)
                    pooled_feat = np.nan_to_num(pooled_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    self.visual_lookup[pid] = pooled_feat
            else:
                raise FileNotFoundError(f"Visual sequence files not found at {seq_npy_path} or {seq_idx_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = int(row['participant_id'])
        label = float(row['binary_label'])

        prep_output_dir = os.path.join(self.base_dir, self.config['data']['preprocessing_output_dir'])

        # 1. Text Feature Loading (768-d)
        text_rel_path = row['text_feature_path']
        text_full_path = os.path.join(prep_output_dir, text_rel_path)
        if os.path.exists(text_full_path):
            text_feat = np.load(text_full_path).astype(np.float32)
            text_feat = np.nan_to_num(text_feat, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            text_dim = self.config['features']['text_dim']
            text_feat = np.zeros(text_dim, dtype=np.float32)

        # 2. Audio Feature Loading (128-d)
        audio_rel_path = row['audio_feature_path']
        audio_full_path = os.path.join(prep_output_dir, audio_rel_path)
        if os.path.exists(audio_full_path):
            audio_feat = np.load(audio_full_path).astype(np.float32)
            audio_feat = np.nan_to_num(audio_feat, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio_dim = self.config['features']['audio_dim']
            audio_feat = np.zeros(audio_dim, dtype=np.float32)

        # 3. Visual Feature Loading (179-d or 31-d)
        if pid in self.visual_lookup:
            visual_feat = self.visual_lookup[pid]
        else:
            vis_dim = self.config['features']['visual_dim']
            visual_feat = np.zeros(vis_dim, dtype=np.float32)

        # Apply z-score normalization if scaler statistics are provided
        if self.scaler_stats is not None:
            if 'visual_mean' in self.scaler_stats and 'visual_std' in self.scaler_stats:
                visual_feat = (visual_feat - self.scaler_stats['visual_mean']) / (self.scaler_stats['visual_std'] + 1e-8)
            if 'audio_mean' in self.scaler_stats and 'audio_std' in self.scaler_stats:
                audio_feat = (audio_feat - self.scaler_stats['audio_mean']) / (self.scaler_stats['audio_std'] + 1e-8)
            if 'text_mean' in self.scaler_stats and 'text_std' in self.scaler_stats:
                text_feat = (text_feat - self.scaler_stats['text_mean']) / (self.scaler_stats['text_std'] + 1e-8)

        return {
            'participant_id': torch.tensor(pid, dtype=torch.long),
            'text': torch.tensor(text_feat, dtype=torch.float32),
            'audio': torch.tensor(audio_feat, dtype=torch.float32),
            'visual': torch.tensor(visual_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def compute_scaler_stats(train_dataset):
    """Computes mean and std across the training dataset for feature scaling."""
    all_visual, all_audio, all_text = [], [], []
    for item in train_dataset:
        all_visual.append(item['visual'].numpy())
        all_audio.append(item['audio'].numpy())
        all_text.append(item['text'].numpy())

    all_visual = np.array(all_visual)
    all_audio = np.array(all_audio)
    all_text = np.array(all_text)

    return {
        'visual_mean': np.mean(all_visual, axis=0),
        'visual_std': np.std(all_visual, axis=0),
        'audio_mean': np.mean(all_audio, axis=0),
        'audio_std': np.std(all_audio, axis=0),
        'text_mean': np.mean(all_text, axis=0),
        'text_std': np.std(all_text, axis=0),
    }


def get_pos_weight(train_dataset):
    """Calculates class balance ratio pos_weight = num_negatives / num_positives for BCE loss."""
    labels = [item['label'].item() for item in train_dataset]
    n_pos = sum(1 for l in labels if l == 1.0)
    n_neg = sum(1 for l in labels if l == 0.0)
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def get_multimodal_dataloaders(config_path_or_dict, base_dir=".", normalize=True):
    """
    Constructs PyTorch DataLoaders for Train, Validation, and Test splits.
    
    Args:
        config_path_or_dict: File path string or config dictionary.
        base_dir (str): Base root directory of the workspace.
        normalize (bool): Whether to perform feature z-score normalization using training stats.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    config = load_config(config_path_or_dict)
    
    train_csv = os.path.join(base_dir, config['data']['train_index'])
    val_csv = os.path.join(base_dir, config['data']['validation_index'])
    test_csv = os.path.join(base_dir, config['data']['test_index'])

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train index CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation index CSV not found: {val_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test index CSV not found: {test_csv}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    raw_train_dataset = MultimodalDepressionDataset(df_train, config, base_dir=base_dir)
    
    scaler_stats = None
    if normalize:
        scaler_stats = compute_scaler_stats(raw_train_dataset)

    train_dataset = MultimodalDepressionDataset(df_train, config, base_dir=base_dir, scaler_stats=scaler_stats)
    val_dataset = MultimodalDepressionDataset(df_val, config, base_dir=base_dir, scaler_stats=scaler_stats)
    test_dataset = MultimodalDepressionDataset(df_test, config, base_dir=base_dir, scaler_stats=scaler_stats)

    batch_size = config['training']['batch_size']
    num_workers = 0  # Safe default for Windows cross-process loader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Simple self-test when run directly
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'centralized_baseline_config.yaml')
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("Testing Multimodal Dataset Loader...")
    train_ld, val_ld, test_ld = get_multimodal_dataloaders(config_file, base_dir=base_path)
    
    print(f"Dataset sizes -> Train: {len(train_ld.dataset)}, Val: {len(val_ld.dataset)}, Test: {len(test_ld.dataset)}")
    
    sample_batch = next(iter(train_ld))
    print("Sample Batch Shapes:")
    print("  Participant IDs:", sample_batch['participant_id'].shape)
    print("  Text Tensor:    ", sample_batch['text'].shape)
    print("  Audio Tensor:   ", sample_batch['audio'].shape)
    print("  Visual Tensor:  ", sample_batch['visual'].shape)
    print("  Label Tensor:   ", sample_batch['label'].shape)
    print("Self-test completed successfully!")
