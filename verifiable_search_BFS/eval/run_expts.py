"""
Main experiment runner for verifiable search evaluation.
Runs experiments comparing different planners on various scenarios.
"""

import csv
import json
import time
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

from env.backend import Environment
from planner.search import BFSSearchPlanner, AStarSearchPlanner
from planner.llm_baseline import LLMPlanner, RandomPlanner
from rules.verify import RuleVerifier
from eval.scenarios import ScenarioGenerator
from eval.metrics import MetricsCollector, EpisodeResult, ExperimentResult


class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize experiment runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.env = Environment(width=5, height=5)
        self.scenario_generator = ScenarioGenerator(self.env)
        self.metrics_collector = MetricsCollector()
        
        # Initialize planners
        self.planners = {
            "bfs": BFSSearchPlanner(self.env),
            "astar": AStarSearchPlanner(self.env),
            "llm_stub": LLMPlanner(self.env),
            "random": RandomPlanner(self.env)
        }
    
    def run_single_episode(self, planner_name: str, scenario: Dict[str, Any], 
                          episode_id: int) -> EpisodeResult:
        """Run a single episode with given planner and scenario."""
        planner = self.planners[planner_name]
        initial_state = scenario["initial_state"]
        goal = scenario["goal"]
        
        start_time = time.time()
        
        # Generate plan
        if planner_name in ["bfs", "astar"]:
            # Use higher depth limit for complex scenarios
            max_depth = 100 if goal in ["tidy_desk", "make_tea"] else 50
            plan, metadata = planner.plan(initial_state, goal, max_depth=max_depth)
            success = metadata["success"]
            steps = len(plan)
            violations = 0  # Search planners don't generate violations
            violation_types = {}
        else:
            plan, metadata = planner.plan(initial_state, goal)
            success = metadata["success"]
            steps = metadata["steps"]
            violations = metadata["violations"]
            violation_types = {"precondition": violations}  # Simplified
        
        runtime = time.time() - start_time
        
        # Execute plan to get final state
        current_state = initial_state.copy()
        verifier = RuleVerifier()
        
        for action in plan:
            if not verifier.violates_rules(current_state, action):
                try:
                    current_state = self.env.apply_action(current_state, action)
                except ValueError:
                    pass  # Action failed, continue
        
        return EpisodeResult(
            episode_id=episode_id,
            planner_name=planner_name,
            scenario_name=scenario["name"],
            success=success,
            steps=steps,
            runtime_s=runtime,
            violations=violations,
            violation_types=violation_types,
            plan=plan,
            final_state=current_state,
            metadata=metadata
        )
    
    def run_experiment(self, planner_name: str, scenario_name: str, 
                      num_episodes: int = 100) -> ExperimentResult:
        """Run a complete experiment with multiple episodes."""
        print(f"Running experiment: {planner_name} on {scenario_name} ({num_episodes} episodes)")
        
        # Get scenario
        scenarios = self.scenario_generator.get_all_scenarios()
        scenario = next((s for s in scenarios if s["name"] == scenario_name), None)
        
        if not scenario:
            raise ValueError(f"Scenario {scenario_name} not found")
        
        episode_results = []
        
        for episode_id in range(num_episodes):
            result = self.run_single_episode(planner_name, scenario, episode_id)
            episode_results.append(result)
            self.metrics_collector.record_episode(result)
            
            if (episode_id + 1) % 10 == 0:
                print(f"  Completed {episode_id + 1}/{num_episodes} episodes")
        
        # Calculate experiment statistics
        success_rate = sum(1 for r in episode_results if r.success) / len(episode_results)
        avg_steps = sum(r.steps for r in episode_results) / len(episode_results)
        avg_runtime = sum(r.runtime_s for r in episode_results) / len(episode_results)
        avg_violations = sum(r.violations for r in episode_results) / len(episode_results)
        violation_rate = sum(r.violations for r in episode_results) / sum(r.steps for r in episode_results) if sum(r.steps for r in episode_results) > 0 else 0
        
        # Calculate confidence interval for success rate
        success_values = [1 if r.success else 0 for r in episode_results]
        confidence_interval = self.metrics_collector.calculate_confidence_interval(success_values)
        
        experiment_result = ExperimentResult(
            experiment_name=f"{planner_name}_{scenario_name}",
            planner_name=planner_name,
            scenario_name=scenario_name,
            num_episodes=num_episodes,
            success_rate=success_rate,
            avg_steps=avg_steps,
            avg_runtime=avg_runtime,
            avg_violations=avg_violations,
            violation_rate=violation_rate,
            episode_results=episode_results,
            confidence_interval=confidence_interval,
            statistical_significance={}
        )
        
        self.metrics_collector.record_experiment(experiment_result)
        return experiment_result
    
    def run_comparison_experiment(self, planners: List[str], scenarios: List[str], 
                                num_episodes: int = 100) -> Dict[str, Any]:
        """Run comparison experiment across multiple planners and scenarios."""
        print(f"Running comparison experiment: {planners} on {scenarios}")
        
        all_results = []
        
        for planner in planners:
            for scenario in scenarios:
                try:
                    result = self.run_experiment(planner, scenario, num_episodes)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error running {planner} on {scenario}: {e}")
                    continue
        
        return {
            "experiments": all_results,
            "summary": self.metrics_collector.generate_summary_report()
        }
    
    def save_results_to_csv(self, filename: str = None) -> str:
        """Save episode results to CSV file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"experiment_results_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'episode_id', 'planner_name', 'scenario_name', 'success', 
                'steps', 'runtime_s', 'violations'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for episode in self.metrics_collector.episodes:
                writer.writerow({
                    'episode_id': episode.episode_id,
                    'planner_name': episode.planner_name,
                    'scenario_name': episode.scenario_name,
                    'success': episode.success,
                    'steps': episode.steps,
                    'runtime_s': episode.runtime_s,
                    'violations': episode.violations
                })
        
        print(f"Results saved to {filepath}")
        return str(filepath)
    
    def save_summary_to_json(self, filename: str = None) -> str:
        """Save summary report to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"experiment_summary_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        summary = self.metrics_collector.generate_summary_report()
        
        with open(filepath, 'w') as jsonfile:
            json.dump(summary, jsonfile, indent=2, default=str)
        
        print(f"Summary saved to {filepath}")
        return str(filepath)
    
    def run_quick_test(self) -> None:
        """Run a quick test with minimal episodes."""
        print("Running quick test...")
        
        planners = ["bfs", "llm_stub"]
        scenarios = ["safe_microwave", "tidy_desk"]
        
        results = self.run_comparison_experiment(planners, scenarios, num_episodes=5)
        
        # Print results
        print("\n=== QUICK TEST RESULTS ===")
        for experiment in results["experiments"]:
            print(f"{experiment.planner_name} on {experiment.scenario_name}:")
            print(f"  Success rate: {experiment.success_rate:.2%}")
            print(f"  Avg steps: {experiment.avg_steps:.1f}")
            print(f"  Avg violations: {experiment.avg_violations:.1f}")
            print()
        
        # Save results
        self.save_results_to_csv("quick_test_results.csv")
        self.save_summary_to_json("quick_test_summary.json")
    
    def run_full_evaluation(self) -> None:
        """Run full evaluation with all planners and scenarios."""
        print("Running full evaluation...")
        
        planners = ["bfs", "astar", "llm_stub", "random"]
        scenarios = ["safe_microwave", "tidy_desk", "make_tea"]
        
        results = self.run_comparison_experiment(planners, scenarios, num_episodes=100)
        
        # Print results
        print("\n=== FULL EVALUATION RESULTS ===")
        summary = results["summary"]
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.2%}")
        print(f"Overall violation rate: {summary['overall_violation_rate']:.3f}")
        print()
        
        print("By planner:")
        for planner, stats in summary["planner_statistics"].items():
            print(f"  {planner}: {stats['success_rate']:.2%} success, {stats['violation_rate']:.3f} violations")
        
        # Save results
        self.save_results_to_csv("full_evaluation_results.csv")
        self.save_summary_to_json("full_evaluation_summary.json")


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run verifiable search experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Experiment mode: quick test or full evaluation")
    parser.add_argument("--results-dir", default="results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    if args.mode == "quick":
        runner.run_quick_test()
    else:
        runner.run_full_evaluation()


if __name__ == "__main__":
    main()
