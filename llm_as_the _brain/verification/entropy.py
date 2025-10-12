"""
Maximum Causal Entropy for Schema Learning
==========================================

Implements the mathematical framework for learning action schemas from demonstrations
using maximum causal entropy principles.

Mathematical Foundation:
max_π H(A|S) = -∑ π(a|s) log π(a|s)
subject to: E_π[fᵢ(s,a)] = E_data[fᵢ(s,a)] for all features i

Solution:
π*(a|s) = exp(∑ θᵢfᵢ(s,a)) / Z(s)

This learns the simplest rules that explain demonstrated behavior.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Set
from scipy.optimize import minimize
from env.state import State
from env.backend import Environment


class MaximumCausalEntropy:
    """Implements maximum causal entropy for schema learning."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        """Initialize with learning parameters."""
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.feature_weights = {}  # θᵢ weights for each feature
        self.feature_functions = {}  # fᵢ(s,a) functions
        self.normalization_constants = {}  # Z(s) for each state
    
    def learn_schema(self, demonstrations: List[Tuple[State, Dict, State]], 
                     features: List[str]) -> Dict[str, float]:
        """
        Learn action schema from demonstrations using maximum causal entropy.
        
        Args:
            demonstrations: List of (state, action, next_state) tuples
            features: List of feature names to consider
            
        Returns:
            Learned feature weights θᵢ
        """
        # Initialize feature functions
        self._initialize_feature_functions(features)
        
        # Compute empirical feature expectations
        empirical_expectations = self._compute_empirical_expectations(demonstrations, features)
        
        # Initialize weights
        initial_weights = np.zeros(len(features))
        
        # Optimize using gradient descent
        result = minimize(
            fun=self._entropy_objective,
            x0=initial_weights,
            args=(demonstrations, features, empirical_expectations),
            method='BFGS',
            options={'maxiter': self.max_iterations}
        )
        
        # Store learned weights
        for i, feature in enumerate(features):
            self.feature_weights[feature] = result.x[i]
        
        return dict(zip(features, result.x))
    
    def predict_action_probability(self, state: State, action: Dict, features: List[str]) -> float:
        """
        Predict probability of action in given state using learned schema.
        
        π*(a|s) = exp(∑ θᵢfᵢ(s,a)) / Z(s)
        
        Args:
            state: Current state
            action: Action to evaluate
            features: List of features
            
        Returns:
            Probability of action
        """
        # Compute feature values
        feature_values = []
        for feature in features:
            if feature in self.feature_functions:
                value = self.feature_functions[feature](state, action)
                feature_values.append(value)
            else:
                feature_values.append(0.0)
        
        # Compute weighted sum
        weighted_sum = sum(self.feature_weights.get(feature, 0.0) * value 
                          for feature, value in zip(features, feature_values))
        
        # Compute normalization constant Z(s)
        z_s = self._compute_normalization_constant(state, features)
        
        # Return probability
        if z_s > 0:
            return np.exp(weighted_sum) / z_s
        else:
            return 0.0
    
    def extract_preconditions(self, demonstrations: List[Tuple[State, Dict, State]], 
                             action_type: str) -> Dict[str, Any]:
        """
        Extract preconditions for a specific action type.
        
        Args:
            demonstrations: List of demonstrations
            action_type: Type of action to analyze
            
        Returns:
            Dictionary of preconditions
        """
        # Filter demonstrations for this action type
        relevant_demos = [(s, a, s_next) for s, a, s_next in demonstrations 
                         if a.get("type") == action_type]
        
        if not relevant_demos:
            return {}
        
        # Analyze common patterns in pre-states
        preconditions = {}
        
        # Check robot position patterns
        robot_positions = [s.robot for s, a, s_next in relevant_demos]
        if len(set(robot_positions)) == 1:
            preconditions["robot_position"] = robot_positions[0]
        
        # Check gripper patterns
        gripper_states = [s.gripper for s, a, s_next in relevant_demos]
        if len(set(gripper_states)) == 1:
            preconditions["gripper_state"] = gripper_states[0]
        
        # Check object properties
        for obj_name in relevant_demos[0][0].objects:
            properties = [s.objects[obj_name].properties for s, a, s_next in relevant_demos]
            if all(props == properties[0] for props in properties):
                preconditions[f"object_{obj_name}_properties"] = properties[0]
        
        return preconditions
    
    def _initialize_feature_functions(self, features: List[str]) -> None:
        """Initialize feature functions for the given features."""
        feature_functions = {
            "robot_at_position": self._feature_robot_at_position,
            "gripper_contains": self._feature_gripper_contains,
            "object_at_position": self._feature_object_at_position,
            "object_has_property": self._feature_object_has_property,
            "action_type": self._feature_action_type,
            "distance_to_goal": self._feature_distance_to_goal,
            "safety_score": self._feature_safety_score
        }
        
        for feature in features:
            if feature in feature_functions:
                self.feature_functions[feature] = feature_functions[feature]
            else:
                # Default feature function
                self.feature_functions[feature] = lambda s, a: 0.0
    
    def _compute_empirical_expectations(self, demonstrations: List[Tuple[State, Dict, State]], 
                                       features: List[str]) -> Dict[str, float]:
        """Compute empirical feature expectations from demonstrations."""
        expectations = {}
        
        for feature in features:
            if feature in self.feature_functions:
                values = []
                for state, action, next_state in demonstrations:
                    value = self.feature_functions[feature](state, action)
                    values.append(value)
                expectations[feature] = np.mean(values)
            else:
                expectations[feature] = 0.0
        
        return expectations
    
    def _entropy_objective(self, weights: np.ndarray, demonstrations: List[Tuple[State, Dict, State]], 
                          features: List[str], empirical_expectations: Dict[str, float]) -> float:
        """Objective function for maximum entropy optimization."""
        # Update weights
        for i, feature in enumerate(features):
            self.feature_weights[feature] = weights[i]
        
        # Compute model expectations
        model_expectations = {}
        for feature in features:
            values = []
            for state, action, next_state in demonstrations:
                prob = self.predict_action_probability(state, action, features)
                value = self.feature_functions[feature](state, action)
                values.append(prob * value)
            model_expectations[feature] = np.mean(values)
        
        # Compute KL divergence (entropy objective)
        kl_divergence = 0.0
        for feature in features:
            empirical = empirical_expectations[feature]
            model = model_expectations[feature]
            if empirical > 0 and model > 0:
                kl_divergence += empirical * np.log(empirical / model)
        
        return kl_divergence
    
    def _compute_normalization_constant(self, state: State, features: List[str]) -> float:
        """Compute normalization constant Z(s) for state."""
        # For simplicity, we'll use a fixed normalization
        # In practice, this would sum over all possible actions
        return 1.0
    
    # Feature functions
    def _feature_robot_at_position(self, state: State, action: Dict) -> float:
        """Feature: Robot at specific position."""
        if "target" in action:
            return 1.0 if state.robot == action["target"] else 0.0
        return 0.0
    
    def _feature_gripper_contains(self, state: State, action: Dict) -> float:
        """Feature: Gripper contains specific object."""
        if "object" in action:
            return 1.0 if state.gripper == action["object"] else 0.0
        return 0.0
    
    def _feature_object_at_position(self, state: State, action: Dict) -> float:
        """Feature: Object at specific position."""
        if "object" in action and action["object"] in state.objects:
            obj = state.objects[action["object"]]
            if "target" in action:
                return 1.0 if obj.position == action["target"] else 0.0
        return 0.0
    
    def _feature_object_has_property(self, state: State, action: Dict) -> float:
        """Feature: Object has specific property."""
        if "object" in action and action["object"] in state.objects:
            obj = state.objects[action["object"]]
            # Check for common properties
            if "is_microwave_safe" in obj.properties:
                return 1.0 if obj.properties["is_microwave_safe"] else 0.0
        return 0.0
    
    def _feature_action_type(self, state: State, action: Dict) -> float:
        """Feature: Action type."""
        return 1.0  # Binary feature for action type
    
    def _feature_distance_to_goal(self, state: State, action: Dict) -> float:
        """Feature: Distance to goal."""
        # Simplified distance calculation
        return 1.0 / (1.0 + state.robot[0] + state.robot[1])
    
    def _feature_safety_score(self, state: State, action: Dict) -> float:
        """Feature: Safety score."""
        # Simplified safety scoring
        if action["type"] == "heat":
            obj_name = action.get("object")
            if obj_name and obj_name in state.objects:
                obj = state.objects[obj_name]
                return 1.0 if obj.properties.get("is_microwave_safe", False) else 0.0
        return 1.0
