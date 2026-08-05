import os
import sys
import argparse
import itertools
import random
import pandas as pd

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config
from modeling.train_centralized import train_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.14 - Advanced Hyperparameter Tuning Terpusat")
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
        default=40,
        help='Maximum number of hyperparameter combinations to try (default: 40)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Max epochs per trial during tuning (default: 30)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for trial shuffling reproducibility'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='advanced',
        choices=['baseline', 'advanced'],
        help='"baseline" = original 3-dim grid, "advanced" = full 8-dim extended grid (default: advanced)'
    )
    return parser.parse_args()


def run_hyperparameter_tuning(config_path, strategy='late_fusion', max_trials=40,
                               epochs=30, seed=42, mode='advanced'):
    """
    Run hyperparameter tuning via grid search over an expanded search space.

    Two modes:
    - 'baseline': Original 3-dim grid (lr, batch_size, dropout, hidden_dim, weight_decay)
    - 'advanced': Extended 8-dim grid including focal loss, noise augmentation, 
                  gated fusion, and LR scheduler options

    Ranking criterion: primary = val_recall, secondary = val_auc (clinical priority).
    This prioritizes maximizing sensitivity (detecting depressed patients) over accuracy.

    Args:
        config_path (str): Path to YAML config file.
        strategy (str): Fusion strategy to tune.
        max_trials (int): Maximum number of trials.
        epochs (int): Max training epochs per trial.
        seed (int): Seed for shuffling trial order.
        mode (str): 'baseline' or 'advanced'.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)

    out_metrics_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    os.makedirs(out_metrics_dir, exist_ok=True)

    print("=" * 70)
    print(f" ADVANCED HYPERPARAMETER TUNING - STRATEGI: {strategy.upper()}")
    print(f" Mode: {mode.upper()} | Max Trials: {max_trials} | Epochs per Trial: {epochs}")
    print("=" * 70)

    if mode == 'baseline':
        # Original grid (backward compatible)
        param_grid = {
            'learning_rate':  [0.0001, 0.0003, 0.0005],
            'batch_size':     [16, 32],
            'dropout':        [0.3, 0.4, 0.5],
            'hidden_dim':     [32, 64, 128],
            'weight_decay':   [0.0005, 0.001, 0.005],
            'use_focal_loss': [False],
            'augment_noise_std': [0.0],
            'use_gated_fusion':  [False],
            'use_lr_scheduler':  [False],
        }
    else:
        # Extended advanced grid targeting clinical performance (Recall, AUC)
        param_grid = {
            # Core training hyperparameters
            'learning_rate':  [0.00005, 0.0001, 0.0002, 0.0005],
            'batch_size':     [8, 16, 32],
            'dropout':        [0.4, 0.5, 0.6],
            'hidden_dim':     [64, 96, 128],
            'weight_decay':   [0.0005, 0.001, 0.003],

            # Advanced techniques
            'use_focal_loss':    [True, False],
            'augment_noise_std': [0.0, 0.01, 0.02],
            'use_gated_fusion':  [True, False],
            'use_lr_scheduler':  [True, False],
        }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Shuffle to get diverse coverage within max_trials budget
    random.seed(seed)
    random.shuffle(all_combinations)

    if max_trials is not None and max_trials < len(all_combinations):
        trials_to_run = all_combinations[:max_trials]
    else:
        trials_to_run = all_combinations

    print(f"Search space size: {len(all_combinations):,} combinations")
    print(f"Trials to run: {len(trials_to_run)} (randomly sampled)\n")

    tuning_records = []

    for i, params in enumerate(trials_to_run, start=1):
        exp_id = f"EXP{i:03d}"
        print(
            f">>> [{i:2d}/{len(trials_to_run)}] {exp_id} | "
            f"LR: {params['learning_rate']:.5f} | Batch: {params['batch_size']:2d} | "
            f"Drop: {params['dropout']} | H: {params['hidden_dim']:3d} | "
            f"L2: {params['weight_decay']:.4f} | "
            f"Focal: {str(params['use_focal_loss']):5s} | "
            f"Noise: {params['augment_noise_std']:.2f} | "
            f"Gated: {str(params['use_gated_fusion']):5s} | "
            f"Sched: {str(params['use_lr_scheduler']):5s}"
        )

        trial_config = load_config(
            os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
        )
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
                verbose=False,
                use_focal_loss=params['use_focal_loss'],
                augment_noise_std=params['augment_noise_std'],
                use_gated_fusion=params['use_gated_fusion'],
                use_lr_scheduler=params['use_lr_scheduler'],
            )

            record = {
                'experiment_id': exp_id,
                'strategy': strategy,
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'dropout': params['dropout'],
                'hidden_dim': params['hidden_dim'],
                'weight_decay': params['weight_decay'],
                'use_focal_loss': params['use_focal_loss'],
                'augment_noise_std': params['augment_noise_std'],
                'use_gated_fusion': params['use_gated_fusion'],
                'use_lr_scheduler': params['use_lr_scheduler'],
                # Validation metrics
                'val_f1': val_m['f1_score'],
                'val_recall': val_m['recall'],
                'val_precision': val_m['precision'],
                'val_auc': val_m['auc_roc'],
                'val_accuracy': val_m['accuracy'],
                'val_loss': val_m['loss'],
                # Test metrics (for reference, NOT used for selection)
                'test_f1': test_m['f1_score'],
                'test_recall': test_m['recall'],
                'test_precision': test_m['precision'],
                'test_auc': test_m['auc_roc'],
                'test_accuracy': test_m['accuracy'],
                'status': 'completed'
            }
        except Exception as e:
            print("    [FAIL] Trial {exp_id} failed: {e}".format(exp_id=exp_id, e=e))
            record = {
                'experiment_id': exp_id,
                'strategy': strategy,
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'dropout': params['dropout'],
                'hidden_dim': params['hidden_dim'],
                'weight_decay': params['weight_decay'],
                'use_focal_loss': params['use_focal_loss'],
                'augment_noise_std': params['augment_noise_std'],
                'use_gated_fusion': params['use_gated_fusion'],
                'use_lr_scheduler': params['use_lr_scheduler'],
                'val_f1': 0.0, 'val_recall': 0.0, 'val_precision': 0.0,
                'val_auc': 0.0, 'val_accuracy': 0.0, 'val_loss': 999.0,
                'test_f1': 0.0, 'test_recall': 0.0, 'test_precision': 0.0,
                'test_auc': 0.0, 'test_accuracy': 0.0,
                'status': f'failed: {str(e)}'
            }

        tuning_records.append(record)
        print(
            f"    [OK] Val  -> F1: {record['val_f1']:.4f} | Rec: {record['val_recall']:.4f} | AUC: {record['val_auc']:.4f}"
            f"\n    [OK] Test -> F1: {record['test_f1']:.4f} | Rec: {record['test_recall']:.4f} | AUC: {record['test_auc']:.4f}\n"
        )

    tuning_df = pd.DataFrame(tuning_records)

    # -----------------------------------------------------------------------
    # RANKING: Primary = val_recall (clinical priority), Secondary = val_auc
    # -----------------------------------------------------------------------
    # Rationale: In clinical depression screening, maximizing sensitivity
    # (Recall) is paramount to avoid missed diagnoses (False Negatives).
    # val_auc is secondary to ensure probabilistic discrimination is also good.
    best_df = tuning_df.sort_values(
        by=['val_recall', 'val_auc'],
        ascending=[False, False]
    ).reset_index(drop=True)

    log_xlsx_path = os.path.join(out_metrics_dir, 'hyperparameter_tuning_log.xlsx')
    selection_xlsx_path = os.path.join(out_metrics_dir, 'best_model_selection.xlsx')

    with pd.ExcelWriter(log_xlsx_path, engine='openpyxl') as writer:
        tuning_df.to_excel(writer, sheet_name='all_trials', index=False)

    with pd.ExcelWriter(selection_xlsx_path, engine='openpyxl') as writer:
        best_df.to_excel(writer, sheet_name='ranked_trials', index=False)

    print("=" * 70)
    print(" TUNING SELESAI!")
    print(f" Best Trial: {best_df.iloc[0]['experiment_id']}")
    print(f"   Val  Recall: {best_df.iloc[0]['val_recall']:.4f} | F1: {best_df.iloc[0]['val_f1']:.4f} | AUC: {best_df.iloc[0]['val_auc']:.4f}")
    print(f"   Test Recall: {best_df.iloc[0]['test_recall']:.4f} | F1: {best_df.iloc[0]['test_f1']:.4f} | AUC: {best_df.iloc[0]['test_auc']:.4f}")
    print(f" Best Config:")
    print(f"   LR={best_df.iloc[0]['learning_rate']} | Batch={best_df.iloc[0]['batch_size']} | "
          f"Dropout={best_df.iloc[0]['dropout']} | HiddenDim={best_df.iloc[0]['hidden_dim']} | "
          f"L2={best_df.iloc[0]['weight_decay']}")
    print(f"   FocalLoss={best_df.iloc[0]['use_focal_loss']} | Noise={best_df.iloc[0]['augment_noise_std']} | "
          f"GatedFusion={best_df.iloc[0]['use_gated_fusion']} | LRSched={best_df.iloc[0]['use_lr_scheduler']}")
    print(f" Log tuning disimpan ke: {log_xlsx_path}")
    print(f" Ranking model terbaik disimpan ke: {selection_xlsx_path}")
    print("=" * 70)

    return best_df


if __name__ == '__main__':
    args = parse_args()
    run_hyperparameter_tuning(
        config_path=args.config,
        strategy=args.strategy,
        max_trials=args.max_trials,
        epochs=args.epochs,
        seed=args.seed,
        mode=args.mode
    )
