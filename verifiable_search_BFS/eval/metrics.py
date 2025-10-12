"""
Metrics and evaluation utilities for verifiable search.
Tracks success rates, violations, and performance statistics.
"""

from typing import Dict, List, Any, Tuple
import statistics
import time
from dataclasses import dataclass
from env.state import State
from rules.verify import RuleVerifier


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    episode_id: int
    planner_name: str
    scenario_name: str
    success: bool
    steps: int
    runtime_s: float
    violations: int
    violation_types: Dict[str, int]
    plan: List[Dict[str, Any]]
    final_state: State
    metadata: Dict[str, Any]


@dataclass
class ExperimentResult:
    """Result of a complete experiment."""
    experiment_name: str
    planner_name: str
    scenario_name: str
    num_episodes: int
    success_rate: float
    avg_steps: float
    avg_runtime: float
    avg_violations: float
    violation_rate: float
    episode_results: List[EpisodeResult]
    confidence_interval: Tuple[float, float]
    statistical_significance: Dict[str, Any]


class MetricsCollector:
    """Collects and analyzes metrics from experiments."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.episodes: List[EpisodeResult] = []
        self.experiments: List[ExperimentResult] = []
    
    def record_episode(self, episode_result: EpisodeResult) -> None:
        """Record a single episode result."""
        self.episodes.append(episode_result)
    
    def record_experiment(self, experiment_result: ExperimentResult) -> None:
        """Record a complete experiment result."""
        self.experiments.append(experiment_result)
    
    def get_episode_results(self, planner_name: str = None, scenario_name: str = None) -> List[EpisodeResult]:
        """Get episode results, optionally filtered."""
        results = self.episodes
        
        if planner_name:
            results = [r for r in results if r.planner_name == planner_name]
        
        if scenario_name:
            results = [r for r in results if r.scenario_name == scenario_name]
        
        return results
    
    def get_experiment_results(self, planner_name: str = None, scenario_name: str = None) -> List[ExperimentResult]:
        """Get experiment results, optionally filtered."""
        results = self.experiments
        
        if planner_name:
            results = [r for r in results if r.planner_name == planner_name]
        
        if scenario_name:
            results = [r for r in results if r.scenario_name == scenario_name]
        
        return results
    
    def calculate_success_rate(self, planner_name: str = None, scenario_name: str = None) -> float:
        """Calculate success rate for given filters."""
        results = self.get_episode_results(planner_name, scenario_name)
        
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.success)
        return successful / len(results)
    
    def calculate_average_steps(self, planner_name: str = None, scenario_name: str = None) -> float:
        """Calculate average steps for given filters."""
        results = self.get_episode_results(planner_name, scenario_name)
        
        if not results:
            return 0.0
        
        steps = [r.steps for r in results]
        return statistics.mean(steps)
    
    def calculate_average_runtime(self, planner_name: str = None, scenario_name: str = None) -> float:
        """Calculate average runtime for given filters."""
        results = self.get_episode_results(planner_name, scenario_name)
        
        if not results:
            return 0.0
        
        runtimes = [r.runtime_s for r in results]
        return statistics.mean(runtimes)
    
    def calculate_violation_rate(self, planner_name: str = None, scenario_name: str = None) -> float:
        """Calculate violation rate for given filters."""
        results = self.get_episode_results(planner_name, scenario_name)
        
        if not results:
            return 0.0
        
        total_violations = sum(r.violations for r in results)
        total_steps = sum(r.steps for r in results)
        
        if total_steps == 0:
            return 0.0
        
        return total_violations / total_steps
    
    def calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        n = len(values)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        
        # Z-score for given confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin_error = z * (stdev / (n ** 0.5))
        
        return (mean - margin_error, mean + margin_error)
    
    def compare_planners(self, planner1: str, planner2: str, scenario_name: str = None) -> Dict[str, Any]:
        """Compare two planners statistically."""
        results1 = self.get_episode_results(planner1, scenario_name)
        results2 = self.get_episode_results(planner2, scenario_name)
        
        if not results1 or not results2:
            return {"error": "Insufficient data for comparison"}
        
        # Success rates
        success1 = self.calculate_success_rate(planner1, scenario_name)
        success2 = self.calculate_success_rate(planner2, scenario_name)
        
        # Violation rates
        violations1 = self.calculate_violation_rate(planner1, scenario_name)
        violations2 = self.calculate_violation_rate(planner2, scenario_name)
        
        # Statistical significance (simple t-test approximation)
        success_diff = abs(success1 - success2)
        violation_diff = abs(violations1 - violations2)
        
        return {
            "planner1": planner1,
            "planner2": planner2,
            "scenario": scenario_name,
            "success_rate_1": success1,
            "success_rate_2": success2,
            "success_difference": success_diff,
            "violation_rate_1": violations1,
            "violation_rate_2": violations2,
            "violation_difference": violation_diff,
            "sample_size_1": len(results1),
            "sample_size_2": len(results2)
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        if not self.episodes:
            return {"error": "No episodes recorded"}
        
        # Overall statistics
        total_episodes = len(self.episodes)
        overall_success_rate = self.calculate_success_rate()
        overall_avg_steps = self.calculate_average_steps()
        overall_avg_runtime = self.calculate_average_runtime()
        overall_violation_rate = self.calculate_violation_rate()
        
        # By planner
        planners = list(set(r.planner_name for r in self.episodes))
        planner_stats = {}
        for planner in planners:
            planner_stats[planner] = {
                "success_rate": self.calculate_success_rate(planner),
                "avg_steps": self.calculate_average_steps(planner),
                "avg_runtime": self.calculate_average_runtime(planner),
                "violation_rate": self.calculate_violation_rate(planner),
                "episodes": len(self.get_episode_results(planner))
            }
        
        # By scenario
        scenarios = list(set(r.scenario_name for r in self.episodes))
        scenario_stats = {}
        for scenario in scenarios:
            scenario_stats[scenario] = {
                "success_rate": self.calculate_success_rate(scenario_name=scenario),
                "avg_steps": self.calculate_average_steps(scenario_name=scenario),
                "avg_runtime": self.calculate_average_runtime(scenario_name=scenario),
                "violation_rate": self.calculate_violation_rate(scenario_name=scenario),
                "episodes": len(self.get_episode_results(scenario_name=scenario))
            }
        
        return {
            "total_episodes": total_episodes,
            "overall_success_rate": overall_success_rate,
            "overall_avg_steps": overall_avg_steps,
            "overall_avg_runtime": overall_avg_runtime,
            "overall_violation_rate": overall_violation_rate,
            "planner_statistics": planner_stats,
            "scenario_statistics": scenario_stats,
            "experiments": len(self.experiments)
        }


class ViolationAnalyzer:
    """Analyzes rule violations in detail."""
    
    def __init__(self):
        """Initialize violation analyzer."""
        self.violation_types: Dict[str, int] = {}
        self.violation_contexts: List[Dict[str, Any]] = []
    
    def analyze_violations(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        """Analyze violations across episode results."""
        total_violations = 0
        violation_by_type = {}
        violation_by_planner = {}
        violation_by_scenario = {}
        
        for result in episode_results:
            # Count violations by type
            for violation_type, count in result.violation_types.items():
                violation_by_type[violation_type] = violation_by_type.get(violation_type, 0) + count
                total_violations += count
            
            # Count violations by planner
            planner = result.planner_name
            violation_by_planner[planner] = violation_by_planner.get(planner, 0) + result.violations
            
            # Count violations by scenario
            scenario = result.scenario_name
            violation_by_scenario[scenario] = violation_by_scenario.get(scenario, 0) + result.violations
        
        return {
            "total_violations": total_violations,
            "violations_by_type": violation_by_type,
            "violations_by_planner": violation_by_planner,
            "violations_by_scenario": violation_by_scenario,
            "most_common_violation": max(violation_by_type.items(), key=lambda x: x[1]) if violation_by_type else None
        }
    
    def get_violation_trends(self, episode_results: List[EpisodeResult]) -> Dict[str, List[float]]:
        """Analyze violation trends over time."""
        # Group by episode order
        sorted_results = sorted(episode_results, key=lambda x: x.episode_id)
        
        trends = {
            "episode_ids": [r.episode_id for r in sorted_results],
            "violations_per_episode": [r.violations for r in sorted_results],
            "success_per_episode": [1 if r.success else 0 for r in sorted_results]
        }
        
        return trends
