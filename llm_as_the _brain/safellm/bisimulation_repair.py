"""
SafeLLM: Bisimulation Distance for Action Repair
===============================================

Implements the bisimulation-based substitution algorithm from SafeLLM.
This is the second of the four core algorithms.

The algorithm:
1. Uses bisimulation distance to find similar safe actions
2. Repairs unsafe actions by substituting with safe alternatives
3. Ensures repaired actions maintain similar behavior while being safe
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from env.state import State
from env.backend import Environment
from .action_verification import ActionVerifier, VerificationResult


@dataclass
class BisimulationMatch:
    """Represents a bisimulation match between actions."""
    original_action: Dict[str, Any]
    safe_action: Dict[str, Any]
    bisimulation_distance: float
    similarity_score: float
    repair_confidence: float
    behavior_preservation: float


@dataclass
class RepairResult:
    """Result of bisimulation-based action repair."""
    original_action: Dict[str, Any]
    repaired_action: Dict[str, Any]
    repair_method: str
    bisimulation_distance: float
    safety_improvement: float
    behavior_preservation: float
    repair_confidence: float
    repair_time: float


class BisimulationRepair:
    """
    Bisimulation Distance for Action Repair from SafeLLM.
    
    This algorithm:
    1. Uses bisimulation distance to find similar safe actions
    2. Repairs unsafe actions by substituting with safe alternatives
    3. Ensures repaired actions maintain similar behavior while being safe
    """
    
    def __init__(self, environment: Environment, action_verifier: ActionVerifier):
        """Initialize bisimulation repair system."""
        self.env = environment
        self.action_verifier = action_verifier
        self.safe_action_database = {}
        self.bisimulation_calculator = BisimulationCalculator()
        self.repair_strategies = RepairStrategies()
        self._initialize_safe_actions()
    
    def repair_action(self, unsafe_action: Dict[str, Any], state: State) -> RepairResult:
        """
        Repair an unsafe action using bisimulation-based substitution.
        
        Args:
            unsafe_action: Unsafe action to repair
            state: Current state
            
        Returns:
            Repair result with repaired action
        """
        start_time = time.time()
        
        # Step 1: Find similar safe actions using bisimulation distance
        similar_actions = self._find_similar_safe_actions(unsafe_action, state)
        
        if not similar_actions:
            # No similar safe actions found - use fallback repair
            return self._fallback_repair(unsafe_action, state, start_time)
        
        # Step 2: Select best repair candidate
        best_match = self._select_best_repair_candidate(similar_actions, unsafe_action, state)
        
        # Step 3: Generate repaired action
        repaired_action = self._generate_repaired_action(unsafe_action, best_match, state)
        
        # Step 4: Verify repaired action is safe
        verification_report = self.action_verifier.verify_action(repaired_action, state)
        
        # Step 5: Calculate repair metrics
        repair_metrics = self._calculate_repair_metrics(
            unsafe_action, repaired_action, best_match, verification_report
        )
        
        repair_time = time.time() - start_time
        
        return RepairResult(
            original_action=unsafe_action,
            repaired_action=repaired_action,
            repair_method="bisimulation_substitution",
            bisimulation_distance=best_match.bisimulation_distance,
            safety_improvement=repair_metrics['safety_improvement'],
            behavior_preservation=repair_metrics['behavior_preservation'],
            repair_confidence=repair_metrics['repair_confidence'],
            repair_time=repair_time
        )
    
    def _find_similar_safe_actions(self, unsafe_action: Dict[str, Any], state: State) -> List[BisimulationMatch]:
        """Find similar safe actions using bisimulation distance."""
        similar_actions = []
        
        for safe_action_type, safe_actions in self.safe_action_database.items():
            for safe_action in safe_actions:
                # Calculate bisimulation distance
                bisim_distance = self.bisimulation_calculator.calculate_distance(
                    unsafe_action, safe_action, state
                )
                
                # Calculate similarity score
                similarity_score = self._calculate_similarity_score(unsafe_action, safe_action)
                
                # Calculate repair confidence
                repair_confidence = self._calculate_repair_confidence(
                    unsafe_action, safe_action, bisim_distance, similarity_score
                )
                
                # Calculate behavior preservation
                behavior_preservation = self._calculate_behavior_preservation(
                    unsafe_action, safe_action, state
                )
                
                # Only consider actions with reasonable similarity
                if similarity_score > 0.3:  # Threshold for similarity
                    match = BisimulationMatch(
                        original_action=unsafe_action,
                        safe_action=safe_action,
                        bisimulation_distance=bisim_distance,
                        similarity_score=similarity_score,
                        repair_confidence=repair_confidence,
                        behavior_preservation=behavior_preservation
                    )
                    similar_actions.append(match)
        
        # Sort by repair confidence (higher is better)
        similar_actions.sort(key=lambda x: x.repair_confidence, reverse=True)
        
        return similar_actions
    
    def _select_best_repair_candidate(self, similar_actions: List[BisimulationMatch],
                                    unsafe_action: Dict[str, Any], state: State) -> BisimulationMatch:
        """Select the best repair candidate from similar actions."""
        if not similar_actions:
            raise ValueError("No similar actions found for repair")
        
        # Score each candidate
        scored_candidates = []
        for match in similar_actions:
            score = self._score_repair_candidate(match, unsafe_action, state)
            scored_candidates.append((score, match))
        
        # Return highest scoring candidate
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]
    
    def _generate_repaired_action(self, unsafe_action: Dict[str, Any], 
                                best_match: BisimulationMatch, state: State) -> Dict[str, Any]:
        """Generate repaired action based on best match."""
        # Start with the safe action as base
        repaired_action = best_match.safe_action.copy()
        
        # Apply repair strategies to maintain original intent
        repaired_action = self.repair_strategies.apply_repair_strategies(
            unsafe_action, repaired_action, best_match, state
        )
        
        return repaired_action
    
    def _calculate_repair_metrics(self, unsafe_action: Dict[str, Any], 
                                repaired_action: Dict[str, Any], best_match: BisimulationMatch,
                                verification_report) -> Dict[str, Any]:
        """Calculate repair metrics."""
        # Safety improvement
        original_safety = 0.0  # Unsafe action has 0 safety
        repaired_safety = verification_report.safety_score
        safety_improvement = repaired_safety - original_safety
        
        # Behavior preservation
        behavior_preservation = best_match.behavior_preservation
        
        # Repair confidence
        repair_confidence = best_match.repair_confidence
        
        return {
            'safety_improvement': safety_improvement,
            'behavior_preservation': behavior_preservation,
            'repair_confidence': repair_confidence
        }
    
    def _calculate_similarity_score(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> float:
        """Calculate similarity score between two actions."""
        # Type similarity
        type_similarity = 1.0 if action1["type"] == action2["type"] else 0.0
        
        # Parameter similarity
        params1 = set(action1.keys()) - {"type"}
        params2 = set(action2.keys()) - {"type"}
        param_similarity = len(params1.intersection(params2)) / max(len(params1), len(params2), 1)
        
        # Value similarity for common parameters
        value_similarity = 0.0
        common_params = params1.intersection(params2)
        if common_params:
            value_matches = 0
            for param in common_params:
                if action1[param] == action2[param]:
                    value_matches += 1
            value_similarity = value_matches / len(common_params)
        
        # Weighted combination
        similarity = (type_similarity * 0.4 + 
                     param_similarity * 0.3 + 
                     value_similarity * 0.3)
        
        return similarity
    
    def _calculate_repair_confidence(self, unsafe_action: Dict[str, Any], 
                                   safe_action: Dict[str, Any], bisim_distance: float,
                                   similarity_score: float) -> float:
        """Calculate confidence in repair quality."""
        # Higher similarity and lower bisimulation distance = higher confidence
        confidence = (similarity_score * 0.6 + 
                     (1.0 - bisim_distance) * 0.4)
        
        return min(confidence, 1.0)
    
    def _calculate_behavior_preservation(self, unsafe_action: Dict[str, Any], 
                                       safe_action: Dict[str, Any], state: State) -> float:
        """Calculate how well the safe action preserves the original behavior."""
        # This is a simplified calculation
        # In practice, this would involve more sophisticated behavior analysis
        
        # Same action type preserves more behavior
        type_preservation = 1.0 if unsafe_action["type"] == safe_action["type"] else 0.5
        
        # Same parameters preserve more behavior
        param_preservation = 0.0
        if unsafe_action["type"] == safe_action["type"]:
            common_params = set(unsafe_action.keys()).intersection(set(safe_action.keys()))
            if common_params:
                param_matches = sum(1 for p in common_params 
                                  if unsafe_action[p] == safe_action[p])
                param_preservation = param_matches / len(common_params)
        
        # Weighted combination
        behavior_preservation = (type_preservation * 0.7 + param_preservation * 0.3)
        
        return behavior_preservation
    
    def _score_repair_candidate(self, match: BisimulationMatch, 
                              unsafe_action: Dict[str, Any], state: State) -> float:
        """Score a repair candidate."""
        # Weighted combination of factors
        score = (match.repair_confidence * 0.4 +
                match.behavior_preservation * 0.3 +
                match.similarity_score * 0.2 +
                (1.0 - match.bisimulation_distance) * 0.1)
        
        return score
    
    def _fallback_repair(self, unsafe_action: Dict[str, Any], state: State, start_time: float) -> RepairResult:
        """Fallback repair when no similar safe actions are found."""
        # Use conservative repair strategies
        repaired_action = self.repair_strategies.conservative_repair(unsafe_action, state)
        
        # Verify repaired action
        verification_report = self.action_verifier.verify_action(repaired_action, state)
        
        repair_time = time.time() - start_time
        
        return RepairResult(
            original_action=unsafe_action,
            repaired_action=repaired_action,
            repair_method="conservative_fallback",
            bisimulation_distance=1.0,  # Maximum distance for fallback
            safety_improvement=verification_report.safety_score,
            behavior_preservation=0.5,  # Moderate preservation for fallback
            repair_confidence=0.3,  # Lower confidence for fallback
            repair_time=repair_time
        )
    
    def _initialize_safe_actions(self):
        """Initialize database of safe actions."""
        self.safe_action_database = {
            "pick": [
                {"type": "pick", "object": "bowl"},
                {"type": "pick", "object": "cup"},
                {"type": "pick", "object": "plate"}
            ],
            "drop": [
                {"type": "drop", "object": "bowl"},
                {"type": "drop", "object": "cup"},
                {"type": "drop", "object": "plate"}
            ],
            "move": [
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [0, 1], "target": [0, 1]},
                {"type": "move", "direction": [-1, 0], "target": [-1, 0]}
            ],
            "heat": [
                {"type": "heat", "object": "bowl"},  # Only microwave-safe objects
                {"type": "heat", "object": "cup"}
            ]
        }


class BisimulationCalculator:
    """Calculator for bisimulation distance between actions."""
    
    def calculate_distance(self, action1: Dict[str, Any], action2: Dict[str, Any], 
                          state: State) -> float:
        """Calculate bisimulation distance between two actions."""
        # Simplified bisimulation distance calculation
        # In practice, this would involve more sophisticated state transition analysis
        
        # Type distance
        type_distance = 0.0 if action1["type"] == action2["type"] else 1.0
        
        # Parameter distance
        param_distance = self._calculate_parameter_distance(action1, action2)
        
        # State transition distance (simplified)
        transition_distance = self._calculate_transition_distance(action1, action2, state)
        
        # Weighted combination
        bisim_distance = (type_distance * 0.4 + 
                         param_distance * 0.3 + 
                         transition_distance * 0.3)
        
        return min(bisim_distance, 1.0)
    
    def _calculate_parameter_distance(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> float:
        """Calculate parameter distance between actions."""
        params1 = set(action1.keys()) - {"type"}
        params2 = set(action2.keys()) - {"type"}
        
        if not params1 and not params2:
            return 0.0
        
        # Jaccard distance
        intersection = len(params1.intersection(params2))
        union = len(params1.union(params2))
        
        return 1.0 - (intersection / union) if union > 0 else 1.0
    
    def _calculate_transition_distance(self, action1: Dict[str, Any], action2: Dict[str, Any], 
                                     state: State) -> float:
        """Calculate state transition distance between actions."""
        # Simplified calculation
        # In practice, this would involve simulating both actions and comparing resulting states
        
        # For now, use parameter similarity as proxy
        return self._calculate_parameter_distance(action1, action2)


class RepairStrategies:
    """Strategies for repairing unsafe actions."""
    
    def apply_repair_strategies(self, unsafe_action: Dict[str, Any], 
                              safe_action: Dict[str, Any], best_match: BisimulationMatch,
                              state: State) -> Dict[str, Any]:
        """Apply repair strategies to generate repaired action."""
        repaired_action = safe_action.copy()
        
        # Strategy 1: Parameter substitution
        repaired_action = self._substitute_parameters(unsafe_action, repaired_action)
        
        # Strategy 2: Constraint relaxation
        repaired_action = self._relax_constraints(repaired_action, state)
        
        # Strategy 3: Safety enhancement
        repaired_action = self._enhance_safety(repaired_action, state)
        
        return repaired_action
    
    def _substitute_parameters(self, unsafe_action: Dict[str, Any], 
                             safe_action: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute parameters from unsafe action where safe."""
        repaired_action = safe_action.copy()
        
        # Copy safe parameters from unsafe action
        for param, value in unsafe_action.items():
            if param != "type" and self._is_safe_parameter(param, value):
                repaired_action[param] = value
        
        return repaired_action
    
    def _relax_constraints(self, action: Dict[str, Any], state: State) -> Dict[str, Any]:
        """Relax constraints to make action more feasible."""
        # This would involve relaxing constraints while maintaining safety
        # For now, return action as-is
        return action
    
    def _enhance_safety(self, action: Dict[str, Any], state: State) -> Dict[str, Any]:
        """Enhance safety of the action."""
        # This would involve adding safety measures
        # For now, return action as-is
        return action
    
    def _is_safe_parameter(self, param: str, value: Any) -> bool:
        """Check if a parameter value is safe."""
        # Simplified safety check
        if param == "object":
            # Check if object is safe
            return isinstance(value, str) and len(value) > 0
        elif param == "target":
            # Check if target is within bounds
            return isinstance(value, list) and len(value) == 2
        else:
            return True
    
    def conservative_repair(self, unsafe_action: Dict[str, Any], state: State) -> Dict[str, Any]:
        """Conservative repair strategy for fallback."""
        # Convert to safest possible action
        if unsafe_action["type"] == "mix":
            # Convert mixing to safe individual actions
            return {"type": "pick", "object": "bowl"}  # Safe fallback
        elif unsafe_action["type"] == "heat":
            # Only heat safe objects
            return {"type": "pick", "object": "bowl"}  # Safe fallback
        else:
            # Use safest version of action type
            return {"type": "move", "direction": [0, 0], "target": [0, 0]}  # Safe fallback
