"""
Optimal Transport for State Distribution Matching
================================================

Implements Wasserstein distance for comparing state distributions between
novel actions and known safe actions.

Mathematical Foundation:
W₁(P, Q) = inf_γ ∫∫ d(x,y) γ(dx, dy)
where γ is a coupling of P and Q

In our case:
P = State distribution after novel action
Q = State distribution after known safe action
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from env.state import State
from env.backend import Environment


class OptimalTransport:
    """Implements optimal transport for state distribution matching."""
    
    def __init__(self, distance_metric: str = "euclidean"):
        """Initialize with distance metric."""
        self.distance_metric = distance_metric
        self.coupling_cache = {}  # Cache for computed couplings
    
    def wasserstein_distance(self, dist1: np.ndarray, dist2: np.ndarray, 
                           cost_matrix: Optional[np.ndarray] = None) -> float:
        """
        Compute Wasserstein-1 distance between two distributions.
        
        W₁(P, Q) = inf_γ ∫∫ d(x,y) γ(dx, dy)
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
            cost_matrix: Precomputed cost matrix (optional)
            
        Returns:
            Wasserstein-1 distance
        """
        # Ensure distributions are normalized
        dist1 = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
        dist2 = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
        
        # Compute cost matrix if not provided
        if cost_matrix is None:
            n1, n2 = len(dist1), len(dist2)
            cost_matrix = np.ones((n1, n2))  # Default uniform cost
        
        # Solve optimal transport problem
        # For discrete distributions, this becomes assignment problem
        if len(dist1) == len(dist2):
            # Use Hungarian algorithm for square cost matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            optimal_cost = cost_matrix[row_indices, col_indices].sum()
        else:
            # Use linear programming for general case
            optimal_cost = self._solve_optimal_transport(dist1, dist2, cost_matrix)
        
        return optimal_cost
    
    def state_distribution_distance(self, states1: List[State], states2: List[State]) -> float:
        """
        Compute Wasserstein distance between two sets of states.
        
        Args:
            states1: First set of states
            states2: Second set of states
            
        Returns:
            Wasserstein distance
        """
        if not states1 or not states2:
            return float('inf')
        
        # Convert states to feature vectors
        features1 = [self._state_to_features(s) for s in states1]
        features2 = [self._state_to_features(s) for s in states2]
        
        # Compute cost matrix
        cost_matrix = cdist(features1, features2, metric=self.distance_metric)
        
        # Convert to probability distributions
        dist1 = np.ones(len(features1)) / len(features1)
        dist2 = np.ones(len(features2)) / len(features2)
        
        # Compute Wasserstein distance
        return self.wasserstein_distance(dist1, dist2, cost_matrix)
    
    def compare_action_outcomes(self, state: State, action1: Dict[str, Any], 
                              action2: Dict[str, Any], env: Environment, 
                              n_samples: int = 100) -> float:
        """
        Compare outcomes of two actions using optimal transport.
        
        Args:
            state: Initial state
            action1: First action
            action2: Second action
            env: Environment
            n_samples: Number of samples for each action
            
        Returns:
            Wasserstein distance between action outcomes
        """
        # Sample outcomes for both actions
        outcomes1 = self._sample_action_outcomes(state, action1, env, n_samples)
        outcomes2 = self._sample_action_outcomes(state, action2, env, n_samples)
        
        # Compute distance
        return self.state_distribution_distance(outcomes1, outcomes2)
    
    def is_action_similar_to_safe(self, state: State, novel_action: Dict[str, Any], 
                                 safe_actions: List[Dict[str, Any]], env: Environment,
                                 threshold: float = 0.1) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if novel action is similar to any safe action using optimal transport.
        
        Args:
            state: Current state
            novel_action: Novel action to verify
            safe_actions: List of known safe actions
            env: Environment
            threshold: Similarity threshold
            
        Returns:
            (is_similar, min_distance, most_similar_action)
        """
        min_distance = float('inf')
        most_similar_action = None
        
        for safe_action in safe_actions:
            distance = self.compare_action_outcomes(state, novel_action, safe_action, env)
            if distance < min_distance:
                min_distance = distance
                most_similar_action = safe_action
        
        is_similar = min_distance <= threshold
        
        return is_similar, min_distance, most_similar_action
    
    def compute_transport_plan(self, dist1: np.ndarray, dist2: np.ndarray, 
                             cost_matrix: np.ndarray) -> np.ndarray:
        """
        Compute optimal transport plan (coupling) between distributions.
        
        Args:
            dist1: Source distribution
            dist2: Target distribution
            cost_matrix: Cost matrix
            
        Returns:
            Optimal transport plan (coupling matrix)
        """
        # Normalize distributions
        dist1 = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
        dist2 = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
        
        # Solve optimal transport problem
        if len(dist1) == len(dist2):
            # Use Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            coupling = np.zeros((len(dist1), len(dist2)))
            for i, j in zip(row_indices, col_indices):
                coupling[i, j] = min(dist1[i], dist2[j])
        else:
            # Use linear programming approximation
            coupling = self._solve_optimal_transport(dist1, dist2, cost_matrix)
        
        return coupling
    
    def _solve_optimal_transport(self, dist1: np.ndarray, dist2: np.ndarray, 
                               cost_matrix: np.ndarray) -> float:
        """
        Solve optimal transport problem using linear programming.
        
        This is a simplified implementation. In practice, you would use
        a proper linear programming solver like scipy.optimize.linprog.
        """
        # Simplified greedy algorithm
        n1, n2 = len(dist1), len(dist2)
        total_cost = 0.0
        
        # Greedy assignment
        remaining1 = dist1.copy()
        remaining2 = dist2.copy()
        
        while np.sum(remaining1) > 1e-6 and np.sum(remaining2) > 1e-6:
            # Find minimum cost assignment
            min_cost = float('inf')
            min_i, min_j = 0, 0
            
            for i in range(n1):
                for j in range(n2):
                    if remaining1[i] > 1e-6 and remaining2[j] > 1e-6:
                        cost = cost_matrix[i, j]
                        if cost < min_cost:
                            min_cost = cost
                            min_i, min_j = i, j
            
            # Assign minimum possible mass
            mass = min(remaining1[min_i], remaining2[min_j])
            total_cost += mass * min_cost
            remaining1[min_i] -= mass
            remaining2[min_j] -= mass
        
        return total_cost
    
    def _state_to_features(self, state: State) -> np.ndarray:
        """Convert state to feature vector for distance computation."""
        features = []
        
        # Robot position (normalized)
        features.extend([state.robot[0] / state.width, state.robot[1] / state.height])
        
        # Gripper state
        features.append(1.0 if state.gripper is not None else 0.0)
        
        # Object positions (normalized)
        for obj_name in sorted(state.objects.keys()):
            obj = state.objects[obj_name]
            features.extend([obj.position[0] / state.width, obj.position[1] / state.height])
        
        # Object properties
        for obj_name in sorted(state.objects.keys()):
            obj = state.objects[obj_name]
            features.append(1.0 if obj.properties.get("is_microwave_safe", False) else 0.0)
            features.append(1.0 if obj.properties.get("is_heated", False) else 0.0)
            features.append(1.0 if obj.properties.get("is_container", False) else 0.0)
        
        # Open containers
        features.append(len(state.open_containers))
        
        return np.array(features)
    
    def _sample_action_outcomes(self, state: State, action: Dict[str, Any], 
                              env: Environment, n_samples: int) -> List[State]:
        """Sample outcome states for an action."""
        outcomes = []
        
        for _ in range(n_samples):
            try:
                new_state = env.apply_action(state, action)
                outcomes.append(new_state)
            except ValueError:
                # Invalid action, outcome is same state
                outcomes.append(state)
        
        return outcomes
    
    def compute_barycenter(self, distributions: List[np.ndarray], 
                          weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Compute Wasserstein barycenter of distributions.
        
        Args:
            distributions: List of distributions
            weights: Weights for each distribution (optional)
            
        Returns:
            Wasserstein barycenter
        """
        if not distributions:
            return np.array([])
        
        if weights is None:
            weights = [1.0 / len(distributions)] * len(distributions)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # For simplicity, use weighted average
        # In practice, this would solve the barycenter optimization problem
        barycenter = np.zeros_like(distributions[0])
        for dist, weight in zip(distributions, weights):
            barycenter += weight * dist
        
        return barycenter
    
    def compute_transport_cost(self, state1: State, state2: State) -> float:
        """Compute transport cost between two states."""
        features1 = self._state_to_features(state1)
        features2 = self._state_to_features(state2)
        
        if self.distance_metric == "euclidean":
            return np.linalg.norm(features1 - features2)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(features1 - features2))
        else:
            return np.linalg.norm(features1 - features2)
