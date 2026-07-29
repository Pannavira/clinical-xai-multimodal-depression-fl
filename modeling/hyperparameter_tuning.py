import os
import sys
import argparse
import itertools
import pandas as pd

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config
from modeling.train_centralized import train_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.14 - Hyperparameter Tuning Terpusat")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML config file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='late_fusion',
        help='Model strategy to tune'
    )
    parser.add_argument(
        '--max_trials',
        type=int,
        default=None,
        help='Maximum number of hyperparameter combinations to try'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Max epochs per trial during tuning'
    )
    return parser.parse_args()


def run_hyperparameter_tuning(config_path, strategy='late_fusion', max_trials=None, epochs=25):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
    
    out_metrics_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    os.makedirs(out_metrics_dir, exist_ok=True)

    print("=" * 70)
    print(f" HYPERPARAMETER TUNING UNTUK STRATEGI: {strategy.upper()}")
    print("=" * 70)

    # Candidate hyperparameter grid targeting overfitting reduction
    param_grid = {
        'learning_rate': [0.0001, 0.0003, 0.0005],
        'batch_size': [16, 32],
        'dropout': [0.3, 0.4, 0.5],
        'hidden_dim': [32, 64, 128],
        'weight_decay': [0.0005, 0.001, 0.005]
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if max_trials is not None and max_trials < len(all_combinations):
        trials_to_run = all_combinations[:max_trials]
    else:
        trials_to_run = all_combinations

    print(f"Total kombinasi hyperparameter yang akan diuji: {len(trials_to_run)} trials.\n")

    tuning_records = []

    for i, params in enumerate(trials_to_run, start=1):
        exp_id = f"EXP{i:03d}"
        print(f">>> Running Trial [{i}/{len(trials_to_run)}] {exp_id} | LR: {params['learning_rate']} | Batch: {params['batch_size']} | Dropout: {params['dropout']} | HiddenDim: {params['hidden_dim']} | L2: {params['weight_decay']}")

        trial_config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
        trial_config['model']['hidden_dim'] = params['hidden_dim']
        trial_config['model']['fusion_dim'] = params['hidden_dim']
        trial_config['model']['dropout'] = params['dropout']

        try:
            _, val_m, test_m, _ = train_single_model(
                trial_config,
                strategy=strategy,
                epochs=epochs,
                lr=params['learning_rate'],
                batch_size=params['batch_size'],
                weight_decay=params['weight_decay'],
                verbose=False
            )

            record = {
                'experiment_id': exp_id,
                'strategy': strategy,
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'dropout': params['dropout'],
                'hidden_dim': params['hidden_dim'],
                'weight_decay': params['weight_decay'],
                'val_f1': val_m['f1_score'],
                'val_auc': val_m['auc_roc'],
                'val_accuracy': val_m['accuracy'],
                'val_recall': val_m['recall'],
                'val_loss': val_m['loss'],
                'test_f1': test_m['f1_score'],
                'test_auc': test_m['auc_roc'],
                'status': 'completed'
            }
        except Exception as e:
            print(f"    Trial {exp_id} failed with error: {e}")
            record = {
                'experiment_id': exp_id,
                'strategy': strategy,
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'dropout': params['dropout'],
                'hidden_dim': params['hidden_dim'],
                'weight_decay': params['weight_decay'],
                'val_f1': 0.0,
                'val_auc': 0.0,
                'val_accuracy': 0.0,
                'val_recall': 0.0,
                'val_loss': 999.0,
                'test_f1': 0.0,
                'test_auc': 0.0,
                'status': f'failed: {str(e)}'
            }

        tuning_records.append(record)
        print(f"    Results -> Val F1: {record['val_f1']:.4f} | Val AUC: {record['val_auc']:.4f} | Test F1: {record['test_f1']:.4f}\n")

    tuning_df = pd.DataFrame(tuning_records)

    # Sort candidates by Val F1 primary, Val AUC secondary
    best_df = tuning_df.sort_values(by=['val_f1', 'val_auc'], ascending=[False, False]).reset_index(drop=True)

    log_xlsx_path = os.path.join(out_metrics_dir, 'hyperparameter_tuning_log.xlsx')
    selection_xlsx_path = os.path.join(out_metrics_dir, 'best_model_selection.xlsx')

    with pd.ExcelWriter(log_xlsx_path, engine='openpyxl') as writer:
        tuning_df.to_excel(writer, sheet_name='all_trials', index=False)

    with pd.ExcelWriter(selection_xlsx_path, engine='openpyxl') as writer:
        best_df.to_excel(writer, sheet_name='ranked_trials', index=False)

    print("=" * 70)
    print(" TUNING SELESAI!")
    print(f" Best Trial: {best_df.iloc[0]['experiment_id']} | Val F1: {best_df.iloc[0]['val_f1']:.4f} | Val AUC: {best_df.iloc[0]['val_auc']:.4f}")
    print(f" Log tuning disimpan ke: {log_xlsx_path}")
    print(f" Rangking model terbaik disimpan ke: {selection_xlsx_path}")
    print("=" * 70)

    return best_df


if __name__ == '__main__':
    args = parse_args()
    run_hyperparameter_tuning(
        config_path=args.config,
        strategy=args.strategy,
        max_trials=args.max_trials,
        epochs=args.epochs
    )
