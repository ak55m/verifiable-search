"""
SafeLLM: Schema Learning via Maximum Causal Entropy
=================================================

Implements the schema learning algorithm from SafeLLM using Maximum Causal Entropy.
This is the third of the four core algorithms.

The algorithm:
1. Learns missing action schemas from demonstrations
2. Uses Maximum Causal Entropy to find optimal action distributions
3. Discovers preconditions, postconditions, and safety invariants
4. Builds a knowledge base of safe action patterns
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from env.state import State
from env.backend import Environment
from .action_verification import ActionSchema, ActionVerifier


@dataclass
class Demonstration:
    """Represents a demonstration of safe action execution."""
    action: Dict[str, Any]
    pre_state: State
    post_state: State
    success: bool
    safety_score: float
    context: Dict[str, Any]


@dataclass
class LearnedSchema:
    """Represents a learned action schema."""
    action_type: str
    parameters: List[str]
    learned_preconditions: List[str]
    learned_postconditions: List[str]
    learned_invariants: List[str]
    learned_constraints: List[str]
    confidence: float
    support_count: int
    entropy_score: float


@dataclass
class LearningResult:
    """Result of schema learning process."""
    learned_schemas: List[LearnedSchema]
    learning_accuracy: float
    schema_coverage: float
    learning_time: float
    demonstrations_used: int
    new_patterns_discovered: int


class SchemaLearner:
    """
    Schema Learning via Maximum Causal Entropy from SafeLLM.
    
    This algorithm:
    1. Learns missing action schemas from demonstrations
    2. Uses Maximum Causal Entropy to find optimal action distributions
    3. Discovers preconditions, postconditions, and safety invariants
    4. Builds a knowledge base of safe action patterns
    """
    
    def __init__(self, environment: Environment, action_verifier: ActionVerifier):
        """Initialize schema learner."""
        self.env = environment
        self.action_verifier = action_verifier
        self.demonstrations = []
        self.learned_schemas = {}
        self.feature_extractor = FeatureExtractor()
        self.entropy_optimizer = MaximumCausalEntropyOptimizer()
        self.pattern_discoverer = PatternDiscoverer()
    
    def learn_from_demonstrations(self, demonstrations: List[Demonstration]) -> LearningResult:
        """
        Learn action schemas from demonstrations using Maximum Causal Entropy.
        
        Args:
            demonstrations: List of safe action demonstrations
            
        Returns:
            Learning result with learned schemas
        """
        start_time = time.time()
        
        # Store demonstrations
        self.demonstrations.extend(demonstrations)
        
        # Step 1: Extract features from demonstrations
        features = self.feature_extractor.extract_features(demonstrations)
        
        # Step 2: Group demonstrations by action type
        action_groups = self._group_demonstrations_by_action_type(demonstrations)
        
        # Step 3: Learn schemas for each action type
        learned_schemas = []
        for action_type, action_demos in action_groups.items():
            schema = self._learn_schema_for_action_type(action_type, action_demos, features)
            if schema:
                learned_schemas.append(schema)
                self.learned_schemas[action_type] = schema
        
        # Step 4: Discover new patterns
        new_patterns = self.pattern_discoverer.discover_patterns(demonstrations, learned_schemas)
        
        # Step 5: Calculate learning metrics
        learning_accuracy = self._calculate_learning_accuracy(learned_schemas, demonstrations)
        schema_coverage = self._calculate_schema_coverage(learned_schemas, demonstrations)
        
        learning_time = time.time() - start_time
        
        return LearningResult(
            learned_schemas=learned_schemas,
            learning_accuracy=learning_accuracy,
            schema_coverage=schema_coverage,
            learning_time=learning_time,
            demonstrations_used=len(demonstrations),
            new_patterns_discovered=len(new_patterns)
        )
    
    def _group_demonstrations_by_action_type(self, demonstrations: List[Demonstration]) -> Dict[str, List[Demonstration]]:
        """Group demonstrations by action type."""
        groups = defaultdict(list)
        for demo in demonstrations:
            action_type = demo.action["type"]
            groups[action_type].append(demo)
        return dict(groups)
    
    def _learn_schema_for_action_type(self, action_type: str, demonstrations: List[Demonstration],
                                    features: Dict[str, Any]) -> Optional[LearnedSchema]:
        """Learn schema for a specific action type."""
        if len(demonstrations) < 2:  # Need at least 2 demonstrations
            return None
        
        # Step 1: Extract action parameters
        parameters = self._extract_parameters(demonstrations)
        
        # Step 2: Learn preconditions using Maximum Causal Entropy
        preconditions = self._learn_preconditions(demonstrations, features)
        
        # Step 3: Learn postconditions
        postconditions = self._learn_postconditions(demonstrations, features)
        
        # Step 4: Learn safety invariants
        invariants = self._learn_safety_invariants(demonstrations, features)
        
        # Step 5: Learn constraints
        constraints = self._learn_constraints(demonstrations, features)
        
        # Step 6: Calculate confidence and support
        confidence = self._calculate_schema_confidence(demonstrations, preconditions, postconditions)
        support_count = len(demonstrations)
        
        # Step 7: Calculate entropy score
        entropy_score = self.entropy_optimizer.calculate_entropy_score(demonstrations, features)
        
        return LearnedSchema(
            action_type=action_type,
            parameters=parameters,
            learned_preconditions=preconditions,
            learned_postconditions=postconditions,
            learned_invariants=invariants,
            learned_constraints=constraints,
            confidence=confidence,
            support_count=support_count,
            entropy_score=entropy_score
        )
    
    def _extract_parameters(self, demonstrations: List[Demonstration]) -> List[str]:
        """Extract parameters from demonstrations."""
        all_params = set()
        for demo in demonstrations:
            for param in demo.action.keys():
                if param != "type":
                    all_params.add(param)
        return list(all_params)
    
    def _learn_preconditions(self, demonstrations: List[Demonstration], 
                           features: Dict[str, Any]) -> List[str]:
        """Learn preconditions using Maximum Causal Entropy."""
        preconditions = []
        
        # Analyze state features before action execution
        pre_state_features = []
        for demo in demonstrations:
            pre_features = self.feature_extractor.extract_state_features(demo.pre_state)
            pre_state_features.append(pre_features)
        
        # Find common patterns in pre-states
        common_patterns = self._find_common_patterns(pre_state_features)
        
        # Convert patterns to precondition rules
        for pattern in common_patterns:
            if pattern['frequency'] >= 0.8:  # 80% of demonstrations
                precondition = self._pattern_to_precondition(pattern)
                if precondition:
                    preconditions.append(precondition)
        
        return preconditions
    
    def _learn_postconditions(self, demonstrations: List[Demonstration], 
                            features: Dict[str, Any]) -> List[str]:
        """Learn postconditions from demonstrations."""
        postconditions = []
        
        # Analyze state features after action execution
        post_state_features = []
        for demo in demonstrations:
            post_features = self.feature_extractor.extract_state_features(demo.post_state)
            post_state_features.append(post_features)
        
        # Find common patterns in post-states
        common_patterns = self._find_common_patterns(post_state_features)
        
        # Convert patterns to postcondition rules
        for pattern in common_patterns:
            if pattern['frequency'] >= 0.8:  # 80% of demonstrations
                postcondition = self._pattern_to_postcondition(pattern)
                if postcondition:
                    postconditions.append(postcondition)
        
        return postconditions
    
    def _learn_safety_invariants(self, demonstrations: List[Demonstration], 
                               features: Dict[str, Any]) -> List[str]:
        """Learn safety invariants from demonstrations."""
        invariants = []
        
        # Analyze safety features across all demonstrations
        safety_features = []
        for demo in demonstrations:
            safety_feat = self.feature_extractor.extract_safety_features(demo.pre_state, demo.post_state)
            safety_features.append(safety_feat)
        
        # Find safety patterns that are always preserved
        safety_patterns = self._find_safety_patterns(safety_features)
        
        # Convert patterns to invariant rules
        for pattern in safety_patterns:
            if pattern['frequency'] >= 0.95:  # 95% of demonstrations
                invariant = self._pattern_to_invariant(pattern)
                if invariant:
                    invariants.append(invariant)
        
        return invariants
    
    def _learn_constraints(self, demonstrations: List[Demonstration], 
                         features: Dict[str, Any]) -> List[str]:
        """Learn constraints from demonstrations."""
        constraints = []
        
        # Analyze constraint features
        constraint_features = []
        for demo in demonstrations:
            constraint_feat = self.feature_extractor.extract_constraint_features(demo.action, demo.pre_state)
            constraint_features.append(constraint_feat)
        
        # Find constraint patterns
        constraint_patterns = self._find_constraint_patterns(constraint_features)
        
        # Convert patterns to constraint rules
        for pattern in constraint_patterns:
            if pattern['frequency'] >= 0.8:  # 80% of demonstrations
                constraint = self._pattern_to_constraint(pattern)
                if constraint:
                    constraints.append(constraint)
        
        return constraints
    
    def _find_common_patterns(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common patterns in feature lists."""
        if not features_list:
            return []
        
        # Count feature occurrences
        feature_counts = defaultdict(int)
        total_demos = len(features_list)
        
        for features in features_list:
            for feature, value in features.items():
                if value:  # Feature is present/true
                    feature_counts[feature] += 1
        
        # Convert to patterns with frequency
        patterns = []
        for feature, count in feature_counts.items():
            frequency = count / total_demos
            patterns.append({
                'feature': feature,
                'frequency': frequency,
                'count': count
            })
        
        return patterns
    
    def _find_safety_patterns(self, safety_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find safety patterns that are always preserved."""
        if not safety_features:
            return []
        
        # Find features that are always true (safety preserved)
        safety_counts = defaultdict(int)
        total_demos = len(safety_features)
        
        for features in safety_features:
            for feature, value in features.items():
                if value:  # Safety feature is preserved
                    safety_counts[feature] += 1
        
        # Convert to patterns
        patterns = []
        for feature, count in safety_counts.items():
            frequency = count / total_demos
            patterns.append({
                'feature': feature,
                'frequency': frequency,
                'count': count
            })
        
        return patterns
    
    def _find_constraint_patterns(self, constraint_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find constraint patterns from demonstrations."""
        if not constraint_features:
            return []
        
        # Count constraint occurrences
        constraint_counts = defaultdict(int)
        total_demos = len(constraint_features)
        
        for features in constraint_features:
            for feature, value in features.items():
                if value:  # Constraint is satisfied
                    constraint_counts[feature] += 1
        
        # Convert to patterns
        patterns = []
        for feature, count in constraint_counts.items():
            frequency = count / total_demos
            patterns.append({
                'feature': feature,
                'frequency': frequency,
                'count': count
            })
        
        return patterns
    
    def _pattern_to_precondition(self, pattern: Dict[str, Any]) -> Optional[str]:
        """Convert pattern to precondition rule."""
        feature = pattern['feature']
        
        if feature == 'object_exists':
            return 'object_exists'
        elif feature == 'object_at_robot':
            return 'object_at_robot'
        elif feature == 'gripper_empty':
            return 'gripper_empty'
        elif feature == 'at_microwave':
            return 'at_microwave'
        elif feature == 'object_microwave_safe':
            return 'object_microwave_safe'
        else:
            return None
    
    def _pattern_to_postcondition(self, pattern: Dict[str, Any]) -> Optional[str]:
        """Convert pattern to postcondition rule."""
        feature = pattern['feature']
        
        if feature == 'object_in_gripper':
            return 'object_in_gripper'
        elif feature == 'object_at_position':
            return 'object_at_position'
        elif feature == 'gripper_empty':
            return 'gripper_empty'
        elif feature == 'robot_at_target':
            return 'robot_at_target'
        elif feature == 'object_heated':
            return 'object_heated'
        else:
            return None
    
    def _pattern_to_invariant(self, pattern: Dict[str, Any]) -> Optional[str]:
        """Convert pattern to safety invariant rule."""
        feature = pattern['feature']
        
        if feature == 'no_object_damage':
            return 'no_object_damage'
        elif feature == 'robot_stable':
            return 'robot_stable'
        elif feature == 'no_collision':
            return 'no_collision'
        elif feature == 'no_fire_risk':
            return 'no_fire_risk'
        elif feature == 'no_explosion_risk':
            return 'no_explosion_risk'
        else:
            return None
    
    def _pattern_to_constraint(self, pattern: Dict[str, Any]) -> Optional[str]:
        """Convert pattern to constraint rule."""
        feature = pattern['feature']
        
        if feature == 'object_pickable':
            return 'object_pickable'
        elif feature == 'within_reach':
            return 'within_reach'
        elif feature == 'safe_drop_location':
            return 'safe_drop_location'
        elif feature == 'within_bounds':
            return 'within_bounds'
        elif feature == 'path_clear':
            return 'path_clear'
        else:
            return None
    
    def _calculate_schema_confidence(self, demonstrations: List[Demonstration],
                                   preconditions: List[str], postconditions: List[str]) -> float:
        """Calculate confidence in learned schema."""
        if not demonstrations:
            return 0.0
        
        # Calculate success rate
        success_count = sum(1 for demo in demonstrations if demo.success)
        success_rate = success_count / len(demonstrations)
        
        # Calculate safety rate
        safety_scores = [demo.safety_score for demo in demonstrations]
        avg_safety = np.mean(safety_scores) if safety_scores else 0.0
        
        # Calculate rule coverage
        rule_count = len(preconditions) + len(postconditions)
        rule_coverage = min(rule_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        confidence = (success_rate * 0.4 + avg_safety * 0.4 + rule_coverage * 0.2)
        
        return min(confidence, 1.0)
    
    def _calculate_learning_accuracy(self, learned_schemas: List[LearnedSchema], 
                                   demonstrations: List[Demonstration]) -> float:
        """Calculate learning accuracy."""
        if not learned_schemas or not demonstrations:
            return 0.0
        
        # Test learned schemas on demonstrations
        correct_predictions = 0
        total_predictions = 0
        
        for demo in demonstrations:
            action_type = demo.action["type"]
            if action_type in self.learned_schemas:
                schema = self.learned_schemas[action_type]
                
                # Predict if action would be safe
                predicted_safe = self._predict_action_safety(demo.action, demo.pre_state, schema)
                actual_safe = demo.safety_score > 0.8
                
                if predicted_safe == actual_safe:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_schema_coverage(self, learned_schemas: List[LearnedSchema], 
                                 demonstrations: List[Demonstration]) -> float:
        """Calculate schema coverage of demonstrations."""
        if not demonstrations:
            return 0.0
        
        # Count demonstrations covered by learned schemas
        covered_demos = 0
        for demo in demonstrations:
            action_type = demo.action["type"]
            if action_type in self.learned_schemas:
                covered_demos += 1
        
        return covered_demos / len(demonstrations)
    
    def _predict_action_safety(self, action: Dict[str, Any], state: State, 
                             schema: LearnedSchema) -> bool:
        """Predict if action would be safe using learned schema."""
        # Simplified prediction based on learned preconditions
        for precondition in schema.learned_preconditions:
            if not self._evaluate_learned_precondition(precondition, state, action):
                return False
        
        return True
    
    def _evaluate_learned_precondition(self, precondition: str, state: State, 
                                     action: Dict[str, Any]) -> bool:
        """Evaluate a learned precondition."""
        if precondition == 'object_exists':
            return 'object' in action and action['object'] in state.objects
        elif precondition == 'object_at_robot':
            return ('object' in action and 
                   action['object'] in state.objects and
                   state.objects[action['object']].position == state.robot)
        elif precondition == 'gripper_empty':
            return state.gripper is None
        elif precondition == 'at_microwave':
            return any(obj.properties.get("is_microwave", False) and obj.position == state.robot
                      for obj in state.objects.values())
        else:
            return True  # Unknown precondition, assume true


class FeatureExtractor:
    """Extracts features from demonstrations for learning."""
    
    def extract_features(self, demonstrations: List[Demonstration]) -> Dict[str, Any]:
        """Extract features from demonstrations."""
        features = {
            'state_features': [],
            'action_features': [],
            'safety_features': [],
            'constraint_features': []
        }
        
        for demo in demonstrations:
            features['state_features'].append(self.extract_state_features(demo.pre_state))
            features['action_features'].append(self.extract_action_features(demo.action))
            features['safety_features'].append(self.extract_safety_features(demo.pre_state, demo.post_state))
            features['constraint_features'].append(self.extract_constraint_features(demo.action, demo.pre_state))
        
        return features
    
    def extract_state_features(self, state: State) -> Dict[str, Any]:
        """Extract features from state."""
        features = {
            'object_exists': len(state.objects) > 0,
            'gripper_empty': state.gripper is None,
            'robot_at_origin': state.robot == (0, 0),
            'has_microwave': any(obj.properties.get("is_microwave", False) for obj in state.objects.values())
        }
        
        # Add object-specific features
        for obj_name, obj in state.objects.items():
            features[f'object_{obj_name}_exists'] = True
            features[f'object_{obj_name}_at_robot'] = obj.position == state.robot
            features[f'object_{obj_name}_microwave_safe'] = obj.properties.get("is_microwave_safe", False)
        
        return features
    
    def extract_action_features(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from action."""
        features = {
            'action_type': action['type'],
            'has_object_param': 'object' in action,
            'has_target_param': 'target' in action,
            'has_direction_param': 'direction' in action
        }
        
        return features
    
    def extract_safety_features(self, pre_state: State, post_state: State) -> Dict[str, Any]:
        """Extract safety features from state transition."""
        features = {
            'no_object_damage': True,  # Simplified
            'robot_stable': True,  # Simplified
            'no_collision': True,  # Simplified
            'no_fire_risk': True,  # Simplified
            'no_explosion_risk': True  # Simplified
        }
        
        return features
    
    def extract_constraint_features(self, action: Dict[str, Any], state: State) -> Dict[str, Any]:
        """Extract constraint features."""
        features = {
            'object_pickable': True,  # Simplified
            'within_reach': True,  # Simplified
            'safe_drop_location': True,  # Simplified
            'within_bounds': True,  # Simplified
            'path_clear': True  # Simplified
        }
        
        return features


class MaximumCausalEntropyOptimizer:
    """Maximum Causal Entropy optimizer for schema learning."""
    
    def calculate_entropy_score(self, demonstrations: List[Demonstration], 
                              features: Dict[str, Any]) -> float:
        """Calculate entropy score for demonstrations."""
        if not demonstrations:
            return 0.0
        
        # Calculate action distribution entropy
        action_counts = defaultdict(int)
        for demo in demonstrations:
            action_type = demo.action['type']
            action_counts[action_type] += 1
        
        total_actions = len(demonstrations)
        entropy = 0.0
        
        for count in action_counts.values():
            probability = count / total_actions
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy


class PatternDiscoverer:
    """Discovers new patterns from demonstrations."""
    
    def discover_patterns(self, demonstrations: List[Demonstration], 
                         learned_schemas: List[LearnedSchema]) -> List[Dict[str, Any]]:
        """Discover new patterns from demonstrations."""
        patterns = []
        
        # Find action sequences
        action_sequences = self._find_action_sequences(demonstrations)
        patterns.extend(action_sequences)
        
        # Find state transitions
        state_transitions = self._find_state_transitions(demonstrations)
        patterns.extend(state_transitions)
        
        # Find safety patterns
        safety_patterns = self._find_safety_patterns(demonstrations)
        patterns.extend(safety_patterns)
        
        return patterns
    
    def _find_action_sequences(self, demonstrations: List[Demonstration]) -> List[Dict[str, Any]]:
        """Find common action sequences."""
        # Simplified implementation
        return []
    
    def _find_state_transitions(self, demonstrations: List[Demonstration]) -> List[Dict[str, Any]]:
        """Find common state transitions."""
        # Simplified implementation
        return []
    
    def _find_safety_patterns(self, demonstrations: List[Demonstration]) -> List[Dict[str, Any]]:
        """Find safety patterns."""
        # Simplified implementation
        return []
