"""
Bisimulation Metrics for Action Similarity
==========================================

Implements Wasserstein distance between actions for similarity assessment.
This is the core mathematical foundation for determining if a novel action
is similar enough to known safe actions.

Mathematical Foundation:
d(a₁, a₂) = maxₛ |R(s,a₁) - R(s,a₂)| + γ ⋅ W₁(P(⋅|s,a₁), P(⋅|s,a₂))

Where:
- R(s,a) = Expected reward in state s taking action a
- P(s′|s,a) = Transition probability to state s′
- W₁ = Wasserstein distance (Earth Mover's Distance)
- γ = Discount factor (0.9 in our case)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.optimize import linear_sum_assignment
from env.state import State
from env.backend import Environment


class BisimulationMetrics:
    """Computes bisimulation metrics for action similarity."""
    
    def __init__(self, gamma: float = 0.9):
        """Initialize with discount factor."""
        self.gamma = gamma
        self.action_cache = {}  # Cache for computed similarities
    
    def wasserstein_distance(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute Wasserstein-1 distance between two distributions.
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
            
        Returns:
            Wasserstein-1 distance
        """
        # Ensure distributions are normalized
        dist1 = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
        dist2 = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
        
        # Compute cumulative distributions
        cum1 = np.cumsum(dist1)
        cum2 = np.cumsum(dist2)
        
        # Wasserstein-1 distance is the L1 norm of cumulative differences
        return np.sum(np.abs(cum1 - cum2))
    
    def action_reward_difference(self, state: State, action1: Dict, action2: Dict, 
                                env: Environment) -> float:
        """
        Compute maximum reward difference between two actions across all states.
        
        Args:
            state: Current state
            action1: First action
            action2: Second action
            env: Environment for reward computation
            
        Returns:
            Maximum reward difference
        """
        # For now, use a simplified reward model
        # In practice, this would use learned reward functions
        
        # Reward based on goal progress and safety
        reward1 = self._compute_action_reward(state, action1, env)
        reward2 = self._compute_action_reward(state, action2, env)
        
        return abs(reward1 - reward2)
    
    def transition_probability_distance(self, state: State, action1: Dict, action2: Dict,
                                      env: Environment) -> float:
        """
        Compute Wasserstein distance between transition distributions.
        
        Args:
            state: Current state
            action1: First action
            action2: Second action
            env: Environment for transition simulation
            
        Returns:
            Wasserstein distance between transition distributions
        """
        # Sample transitions for both actions
        transitions1 = self._sample_transitions(state, action1, env, n_samples=100)
        transitions2 = self._sample_transitions(state, action2, env, n_samples=100)
        
        # Convert to probability distributions over state space
        dist1 = self._transitions_to_distribution(transitions1, state)
        dist2 = self._transitions_to_distribution(transitions2, state)
        
        return self.wasserstein_distance(dist1, dist2)
    
    def bisimulation_distance(self, state: State, action1: Dict, action2: Dict,
                             env: Environment) -> float:
        """
        Compute bisimulation distance between two actions.
        
        d(a₁, a₂) = maxₛ |R(s,a₁) - R(s,a₂)| + γ ⋅ W₁(P(⋅|s,a₁), P(⋅|s,a₂))
        
        Args:
            state: Current state
            action1: First action
            action2: Second action
            env: Environment
            
        Returns:
            Bisimulation distance
        """
        # Compute reward difference
        reward_diff = self.action_reward_difference(state, action1, action2, env)
        
        # Compute transition distribution distance
        transition_dist = self.transition_probability_distance(state, action1, action2, env)
        
        # Combine with discount factor
        bisim_distance = reward_diff + self.gamma * transition_dist
        
        return bisim_distance
    
    def is_action_similar(self, state: State, novel_action: Dict, known_safe_actions: List[Dict],
                          env: Environment, threshold: float = 0.1) -> Tuple[bool, float, Dict]:
        """
        Check if novel action is similar to any known safe action.
        
        Args:
            state: Current state
            novel_action: Novel action to verify
            known_safe_actions: List of known safe actions
            env: Environment
            threshold: Similarity threshold
            
        Returns:
            (is_similar, min_distance, most_similar_action)
        """
        min_distance = float('inf')
        most_similar_action = None
        
        for safe_action in known_safe_actions:
            distance = self.bisimulation_distance(state, novel_action, safe_action, env)
            if distance < min_distance:
                min_distance = distance
                most_similar_action = safe_action
        
        is_similar = min_distance <= threshold
        
        return is_similar, min_distance, most_similar_action
    
    def _compute_action_reward(self, state: State, action: Dict, env: Environment) -> float:
        """Compute reward for an action in a given state."""
        # Simplified reward model
        reward = 0.0
        
        # Goal progress reward
        if env.is_goal_state(state, "safe_microwave"):
            reward += 10.0
        
        # Safety penalty
        if self._is_unsafe_action(state, action):
            reward -= 5.0
        
        # Efficiency reward (shorter plans are better)
        reward += 1.0
        
        return reward
    
    def _is_unsafe_action(self, state: State, action: Dict) -> bool:
        """Check if action is unsafe."""
        # Simplified safety check
        if action["type"] == "heat":
            obj_name = action["object"]
            if obj_name in state.objects:
                obj = state.objects[obj_name]
                if not obj.properties.get("is_microwave_safe", False):
                    return True
        return False
    
    def _sample_transitions(self, state: State, action: Dict, env: Environment, 
                           n_samples: int = 100) -> List[State]:
        """Sample transition states for an action."""
        transitions = []
        
        for _ in range(n_samples):
            try:
                new_state = env.apply_action(state, action)
                transitions.append(new_state)
            except ValueError:
                # Invalid action, transition to same state
                transitions.append(state)
        
        return transitions
    
    def _transitions_to_distribution(self, transitions: List[State], original_state: State) -> np.ndarray:
        """Convert transition states to probability distribution."""
        # Create a simplified state space representation
        # In practice, this would be more sophisticated
        
        # For now, use a simple grid-based representation
        width = original_state.width
        height = original_state.height
        state_space_size = width * height
        
        distribution = np.zeros(state_space_size)
        
        for state in transitions:
            # Convert state to index
            robot_x, robot_y = state.robot
            state_index = robot_y * width + robot_x
            distribution[state_index] += 1
        
        # Normalize
        if np.sum(distribution) > 0:
            distribution = distribution / np.sum(distribution)
        
        return distribution
    
    def get_similarity_matrix(self, actions: List[Dict], state: State, env: Environment) -> np.ndarray:
        """Compute similarity matrix between all pairs of actions."""
        n_actions = len(actions)
        similarity_matrix = np.zeros((n_actions, n_actions))
        
        for i in range(n_actions):
            for j in range(n_actions):
                if i != j:
                    distance = self.bisimulation_distance(state, actions[i], actions[j], env)
                    # Convert distance to similarity (0 = identical, 1 = completely different)
                    similarity = min(distance, 1.0)
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
