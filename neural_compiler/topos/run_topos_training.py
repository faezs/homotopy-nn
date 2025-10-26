"""
Unified Runner for Topos-Based ARC Training

Modes:
1. fractal    - Fractal learning with Kan extension transfer
2. benchmark  - Compare Kan extension vs gradient descent
3. production - Original geometric topos training
4. unified    - Full Gros topos (Grid × Prompt × Graph)

Usage:
    python run_topos_training.py fractal
    python run_topos_training.py benchmark
    python run_topos_training.py production --epochs 100
    python run_topos_training.py unified --curriculum
"""

import argparse
import sys
from pathlib import Path


def run_fractal_training(args):
    """Run fractal learning + Kan extension."""
    print("\n🚀 Starting Fractal + Derivator Training")
    print("=" * 70)

    from fractal_derivator_training import FractalDerivatorTrainer, FractalDerivatorConfig

    config = FractalDerivatorConfig(
        start_scale=args.start_scale,
        end_scale=args.end_scale,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        feature_dim=args.feature_dim,
        use_kan_transfer=True,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        verbose=args.verbose
    )

    trainer = FractalDerivatorTrainer(config)
    metrics = trainer.run_full_pipeline()

    print("\n✅ Fractal training complete!")
    return metrics


def run_benchmark(args):
    """Run Kan extension vs gradient descent benchmark."""
    print("\n⚡ Starting Kan Extension vs Gradient Descent Benchmark")
    print("=" * 70)

    from benchmark_kan_vs_gradient import KanVsGradientBenchmark

    benchmark = KanVsGradientBenchmark(device=args.device)
    results = benchmark.run_full_benchmark()

    print("\n✅ Benchmark complete!")
    return results


def run_production_training(args):
    """Run original geometric topos training."""
    print("\n🏭 Starting Production Geometric Topos Training")
    print("=" * 70)

    from train_arc_geometric_production import train_on_multiple_tasks_tqdm
    import torch

    device = torch.device(args.device)

    results = train_on_multiple_tasks_tqdm(
        task_ids=args.task_ids or None,  # None = all tasks
        max_tasks=args.max_tasks,
        epochs=args.epochs,
        early_stop_patience=args.patience,
        lr=args.lr,
        device=device,
        log_verbose=args.log_verbose
    )

    print("\n✅ Production training complete!")
    return results


def run_unified_training(args):
    """Run full Gros topos (Grid × Prompt × Graph)."""
    print("\n🌌 Starting Unified Gros Topos Training")
    print("=" * 70)

    from unified_gros_topos import UnifiedGrosToposTrainer, TripleProductTopos
    import torch

    # Create triple topos
    topos = TripleProductTopos(
        grid_dim=args.grid_dim,
        prompt_dim=args.prompt_dim,
        graph_dim=args.graph_dim
    ).to(args.device)

    # Trainer config
    config = {
        'epochs_per_level': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'device': args.device,
        'use_curriculum': args.curriculum,
        'generate_synthetic': args.synthetic,
        'coherence_weight': args.coherence_weight,
        'adjunction_weight': args.adjunction_weight
    }

    trainer = UnifiedGrosToposTrainer(topos, config)

    # Load datasets
    from gros_topos_curriculum import GrosToposCurriculum

    curriculum = GrosToposCurriculum()
    curriculum.load_all_datasets()
    curriculum.organize_by_complexity()

    # Train
    metrics = trainer.train_full_curriculum(curriculum)

    print("\n✅ Unified training complete!")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Topos-Based ARC Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fractal learning (fast!)
  python run_topos_training.py fractal --start-scale 0 --end-scale 2

  # Benchmark Kan vs gradient descent
  python run_topos_training.py benchmark

  # Production training (100 epochs)
  python run_topos_training.py production --epochs 100 --max-tasks 50

  # Full Gros topos with curriculum
  python run_topos_training.py unified --curriculum --synthetic
        """
    )

    parser.add_argument(
        'mode',
        choices=['fractal', 'benchmark', 'production', 'unified'],
        help='Training mode'
    )

    # Common arguments
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    # Fractal arguments
    parser.add_argument('--start-scale', type=int, default=0, help='Starting scale level')
    parser.add_argument('--end-scale', type=int, default=2, help='Ending scale level')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warm-up epochs')
    parser.add_argument('--finetune-epochs', type=int, default=5, help='Fine-tune epochs')
    parser.add_argument('--feature-dim', type=int, default=64, help='Feature dimension')

    # Production arguments
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--max-tasks', type=int, default=None, help='Max tasks to train')
    parser.add_argument('--task-ids', nargs='+', help='Specific task IDs')
    parser.add_argument('--log-verbose', action='store_true', help='Verbose TensorBoard logs')

    # Unified arguments
    parser.add_argument('--grid-dim', type=int, default=512, help='Grid topos dimension')
    parser.add_argument('--prompt-dim', type=int, default=768, help='Prompt topos dimension')
    parser.add_argument('--graph-dim', type=int, default=256, help='Graph topos dimension')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic tasks')
    parser.add_argument('--coherence-weight', type=float, default=0.1, help='Coherence loss weight')
    parser.add_argument('--adjunction-weight', type=float, default=0.1, help='Adjunction loss weight')

    args = parser.parse_args()

    # Run selected mode
    try:
        if args.mode == 'fractal':
            results = run_fractal_training(args)
        elif args.mode == 'benchmark':
            results = run_benchmark(args)
        elif args.mode == 'production':
            results = run_production_training(args)
        elif args.mode == 'unified':
            results = run_unified_training(args)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("🎉 ALL TRAINING COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
