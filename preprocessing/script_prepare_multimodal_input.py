import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Alignment Script for Phase 3")
    parser.add_argument('--text_index', type=str, default='../preprocessing_output/text/text_embedding_index.csv', help='Path to text embedding index')
    parser.add_argument('--audio_index', type=str, default='../preprocessing_output/audio/audio_feature_index.csv', help='Path to audio feature index')
    parser.add_argument('--visual_index', type=str, default='../preprocessing_output/visual/visual_features.csv', help='Path to visual features index/data')
    parser.add_argument('--labels', type=str, default='../data/detailed_lables.csv', help='Path to ground truth labels')
    parser.add_argument('--output_dataset', type=str, default='../preprocessing_output/processed_multimodal_dataset.csv', help='Path to output processed dataset')
    parser.add_argument('--output_index', type=str, default='../preprocessing_output/multimodal_feature_index.csv', help='Path to output feature index')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define paths based on script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    text_path = os.path.join(base_dir, 'preprocessing_output', 'text', 'text_embedding_index.csv')
    audio_path = os.path.join(base_dir, 'preprocessing_output', 'audio', 'audio_feature_index.csv')
    visual_path = os.path.join(base_dir, 'preprocessing_output', 'visual', 'visual_features.csv')
    labels_path = os.path.join(base_dir, 'data', 'detailed_lables.csv')
    
    out_dataset = os.path.join(base_dir, 'preprocessing_output', 'processed_multimodal_dataset.csv')
    out_index = os.path.join(base_dir, 'preprocessing_output', 'multimodal_feature_index.csv')

    print("Loading indices and labels...")
    try:
        df_text = pd.read_csv(text_path)
        df_audio = pd.read_csv(audio_path)
        df_visual = pd.read_csv(visual_path)
        df_labels = pd.read_csv(labels_path)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    print("Standardizing participant IDs...")
    # Standardize participant_id to string without 'P' prefix, then to integer for reliable merging
    df_text['participant_id'] = df_text['participant_id'].astype(str).str.replace('P', '', regex=False).astype(int)
    df_audio['participant_id'] = df_audio['participant_id'].astype(str).str.replace('P', '', regex=False).astype(int)
    df_visual['participant_id'] = df_visual['participant_id'].astype(int)
    
    df_labels = df_labels.rename(columns={'Participant': 'participant_id', 'Depression_severity': 'depression_label'})
    df_labels['participant_id'] = df_labels['participant_id'].astype(int)

    print("Extracting required columns...")
    # Rename columns to match requirements
    df_text = df_text[['participant_id', 'embedding_file']].rename(columns={'embedding_file': 'text_feature_path'})
    
    # For audio, using mfcc_file as the primary feature path (or you could combine mfcc and spectrogram)
    df_audio = df_audio[['participant_id', 'mfcc_file']].rename(columns={'mfcc_file': 'audio_feature_path'})
    
    # For visual, create a reference to the sequence or features file. 
    # Since visual features are in a unified file, we point to the sequence file.
    df_visual = df_visual[['participant_id']].copy()
    df_visual['visual_feature_path'] = 'visual/visual_features_sequence.npy'

    # Optional: Prepend modalities output directory to text and audio paths 
    # if they are relative to their modality subfolders.
    df_text['text_feature_path'] = 'text/' + df_text['text_feature_path']
    df_audio['audio_feature_path'] = 'audio/' + df_audio['audio_feature_path']

    print("Performing inner joins across all modalities...")
    # Merge all 
    df_merged = pd.merge(df_text, df_audio, on='participant_id', how='inner')
    df_merged = pd.merge(df_merged, df_visual, on='participant_id', how='inner')
    
    # Merge with labels
    df_merged = pd.merge(df_merged, df_labels[['participant_id', 'depression_label']], on='participant_id', how='inner')

    # Ensure required columns
    final_cols = ['participant_id', 'text_feature_path', 'audio_feature_path', 'visual_feature_path', 'depression_label']
    df_final = df_merged[final_cols]

    print(f"Total synchronized subjects: {len(df_final)}")
    
    # Save the files
    try:
        df_final.to_csv(out_dataset, index=False)
        df_final.to_csv(out_index, index=False)
        print(f"Successfully exported final indices to:")
        print(f"- {out_dataset}")
        print(f"- {out_index}")
    except Exception as e:
        print(f"Error saving output files: {e}")

if __name__ == '__main__':
    main()
