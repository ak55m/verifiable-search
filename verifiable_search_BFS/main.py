#!/usr/bin/env python3
"""
Main entry point for verifiable search experiments.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eval.run_expts import ExperimentRunner


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verifiable Search: Compare rule-based and LLM planners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode quick                    # Quick test (5 episodes each)
  python main.py --mode full                     # Full evaluation (100 episodes each)
  python main.py --planner bfs --scenario safe_microwave --episodes 50
        """
    )
    
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Experiment mode: quick test or full evaluation")
    parser.add_argument("--planner", 
                       choices=["bfs", "astar", "llm_stub", "random", "all"],
                       default="all",
                       help="Planner to test (or 'all' for comparison)")
    parser.add_argument("--scenario",
                       choices=["safe_microwave", "tidy_desk", "make_tea", "all"],
                       default="all", 
                       help="Scenario to test (or 'all' for all scenarios)")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes per experiment")
    parser.add_argument("--results-dir", default="results",
                       help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    if args.mode == "quick":
        print("🚀 Running quick test...")
        runner.run_quick_test()
        
    elif args.mode == "full":
        print("🔬 Running full evaluation...")
        runner.run_full_evaluation()
        
    else:
        # Custom experiment
        planners = [args.planner] if args.planner != "all" else ["bfs", "astar", "llm_stub", "random"]
        scenarios = [args.scenario] if args.scenario != "all" else ["safe_microwave", "tidy_desk", "make_tea"]
        
        print(f"🎯 Running custom experiment: {planners} on {scenarios}")
        results = runner.run_comparison_experiment(planners, scenarios, args.episodes)
        
        # Print summary
        print("\n=== RESULTS SUMMARY ===")
        for experiment in results["experiments"]:
            print(f"{experiment.planner_name} on {experiment.scenario_name}:")
            print(f"  Success rate: {experiment.success_rate:.2%}")
            print(f"  Avg steps: {experiment.avg_steps:.1f}")
            print(f"  Avg violations: {experiment.avg_violations:.1f}")
            print()
        
        # Save results
        runner.save_results_to_csv()
        runner.save_summary_to_json()
    
    print("✅ Experiment completed!")


if __name__ == "__main__":
    main()
