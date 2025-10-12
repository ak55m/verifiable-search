"""
Paper 1: Symbolic Verifier - "Is this plan safe to start?"
========================================================

Implements high-level symbolic verification of complete LLM-generated plans.
This is the building inspector that checks architectural blueprints before construction.

The verifier:
1. Takes complete LLM-generated plans
2. Checks them against safety schemas and invariants
3. Returns "Safe to start" or "Not safe to start"
4. Works at blueprint level, not execution level
"""

import json
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from env.state import State
from env.backend import Environment


class SafetyLevel(Enum):
    """Safety levels for plan verification."""
    SAFE_TO_START = "safe_to_start"
    NOT_SAFE_TO_START = "not_safe_to_start"
    CRITICAL_VIOLATIONS = "critical_violations"
    REQUIRES_REVISION = "requires_revision"


@dataclass
class ActionSchema:
    """Symbolic action schema with pre/post-conditions and invariants."""
    action_type: str
    parameters: List[str]
    preconditions: List[str]
    postconditions: List[str]
    safety_invariants: List[str]
    constraints: List[str]
    risk_level: str  # "low", "medium", "high", "critical"


@dataclass
class PlanViolation:
    """Represents a safety violation in the plan."""
    action_index: int
    action: Dict[str, Any]
    violation_type: str
    violated_rule: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggested_fix: str


@dataclass
class PlanSafetyReport:
    """Complete safety report for a plan."""
    plan: List[Dict[str, Any]]
    safety_level: SafetyLevel
    is_safe_to_start: bool
    violations: List[PlanViolation]
    safety_score: float
    verification_time: float
    detailed_analysis: str
    recommendations: List[str]


class SymbolicVerifier:
    """
    Paper 1: Symbolic Verifier for "Is this plan safe to start?"
    
    This is the building inspector that checks architectural blueprints
    before construction begins. It works at the symbolic level with
    action schemas, pre/post-conditions, and safety invariants.
    """
    
    def __init__(self, environment: Environment):
        """Initialize symbolic verifier."""
        self.env = environment
        self.action_schemas = {}
        self.safety_rules = {}
        self._initialize_action_schemas()
        self._initialize_safety_rules()
    
    def verify_plan_safety(self, plan: List[Dict[str, Any]], 
                          initial_state: State) -> PlanSafetyReport:
        """
        Verify if a complete plan is safe to start.
        
        Args:
            plan: Complete LLM-generated plan
            initial_state: Initial state of the environment
            
        Returns:
            Complete safety report with "Safe to start" or "Not safe to start"
        """
        start_time = time.time()
        
        # Step 1: Analyze plan structure
        structure_analysis = self._analyze_plan_structure(plan)
        
        # Step 2: Check action schemas
        schema_violations = self._check_action_schemas(plan)
        
        # Step 3: Check pre/post-conditions
        condition_violations = self._check_pre_post_conditions(plan, initial_state)
        
        # Step 4: Check safety invariants
        invariant_violations = self._check_safety_invariants(plan, initial_state)
        
        # Step 5: Check domain-specific safety rules
        rule_violations = self._check_safety_rules(plan, initial_state)
        
        # Step 6: Combine all violations
        all_violations = (schema_violations + condition_violations + 
                         invariant_violations + rule_violations)
        
        # Step 7: Determine safety level
        safety_level = self._determine_safety_level(all_violations)
        is_safe_to_start = safety_level == SafetyLevel.SAFE_TO_START
        
        # Step 8: Calculate safety score
        safety_score = self._calculate_safety_score(all_violations, len(plan))
        
        # Step 9: Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            plan, all_violations, structure_analysis
        )
        
        # Step 10: Generate recommendations
        recommendations = self._generate_recommendations(all_violations, safety_level)
        
        verification_time = time.time() - start_time
        
        return PlanSafetyReport(
            plan=plan,
            safety_level=safety_level,
            is_safe_to_start=is_safe_to_start,
            violations=all_violations,
            safety_score=safety_score,
            verification_time=verification_time,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations
        )
    
    def _analyze_plan_structure(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the structure of the plan."""
        analysis = {
            'total_actions': len(plan),
            'action_types': {},
            'parameter_usage': {},
            'plan_complexity': 'low',
            'potential_issues': []
        }
        
        # Count action types
        for action in plan:
            action_type = action.get('type', 'unknown')
            analysis['action_types'][action_type] = analysis['action_types'].get(action_type, 0) + 1
        
        # Analyze parameter usage
        for action in plan:
            for param, value in action.items():
                if param != 'type':
                    analysis['parameter_usage'][param] = analysis['parameter_usage'].get(param, 0) + 1
        
        # Determine complexity
        if len(plan) > 10:
            analysis['plan_complexity'] = 'high'
        elif len(plan) > 5:
            analysis['plan_complexity'] = 'medium'
        
        # Identify potential issues
        if 'mix' in analysis['action_types']:
            analysis['potential_issues'].append('Chemical mixing detected - high risk')
        if 'heat' in analysis['action_types']:
            analysis['potential_issues'].append('Heating operations detected - medium risk')
        if analysis['plan_complexity'] == 'high':
            analysis['potential_issues'].append('High complexity plan - increased risk')
        
        return analysis
    
    def _check_action_schemas(self, plan: List[Dict[str, Any]]) -> List[PlanViolation]:
        """Check if actions conform to defined schemas."""
        violations = []
        
        for i, action in enumerate(plan):
            action_type = action.get('type', 'unknown')
            
            if action_type not in self.action_schemas:
                violations.append(PlanViolation(
                    action_index=i,
                    action=action,
                    violation_type='unknown_action',
                    violated_rule='action_schema_unknown',
                    severity='high',
                    description=f'Unknown action type: {action_type}',
                    suggested_fix=f'Define schema for action type: {action_type}'
                ))
                continue
            
            schema = self.action_schemas[action_type]
            
            # Check required parameters
            missing_params = []
            for param in schema.parameters:
                if param not in action:
                    missing_params.append(param)
            
            if missing_params:
                violations.append(PlanViolation(
                    action_index=i,
                    action=action,
                    violation_type='missing_parameters',
                    violated_rule='required_parameters',
                    severity='high',
                    description=f'Missing required parameters: {missing_params}',
                    suggested_fix=f'Add missing parameters: {missing_params}'
                ))
            
            # Check for unknown parameters
            unknown_params = []
            for param in action.keys():
                if param != 'type' and param not in schema.parameters:
                    unknown_params.append(param)
            
            if unknown_params:
                violations.append(PlanViolation(
                    action_index=i,
                    action=action,
                    violation_type='unknown_parameters',
                    violated_rule='parameter_validation',
                    severity='medium',
                    description=f'Unknown parameters: {unknown_params}',
                    suggested_fix=f'Remove or define parameters: {unknown_params}'
                ))
        
        return violations
    
    def _check_pre_post_conditions(self, plan: List[Dict[str, Any]], 
                                 initial_state: State) -> List[PlanViolation]:
        """Check pre/post-conditions for the entire plan."""
        violations = []
        current_state = initial_state
        
        for i, action in enumerate(plan):
            action_type = action.get('type', 'unknown')
            
            if action_type not in self.action_schemas:
                continue
            
            schema = self.action_schemas[action_type]
            
            # Check preconditions
            for precondition in schema.preconditions:
                if not self._evaluate_precondition(precondition, current_state, action):
                    violations.append(PlanViolation(
                        action_index=i,
                        action=action,
                        violation_type='precondition_violation',
                        violated_rule=precondition,
                        severity='high',
                        description=f'Precondition not satisfied: {precondition}',
                        suggested_fix=f'Ensure precondition is met: {precondition}'
                    ))
            
            # Simulate action execution for postcondition checking
            try:
                current_state = self.env.apply_action(current_state, action)
            except ValueError:
                violations.append(PlanViolation(
                    action_index=i,
                    action=action,
                    violation_type='execution_failure',
                    violated_rule='action_executable',
                    severity='critical',
                    description='Action cannot be executed',
                    suggested_fix='Fix action parameters or preconditions'
                ))
                break
        
        return violations
    
    def _check_safety_invariants(self, plan: List[Dict[str, Any]], 
                               initial_state: State) -> List[PlanViolation]:
        """Check safety invariants throughout the plan."""
        violations = []
        current_state = initial_state
        
        for i, action in enumerate(plan):
            action_type = action.get('type', 'unknown')
            
            if action_type not in self.action_schemas:
                continue
            
            schema = self.action_schemas[action_type]
            
            # Check safety invariants
            for invariant in schema.safety_invariants:
                if not self._evaluate_safety_invariant(invariant, current_state, action):
                    violations.append(PlanViolation(
                        action_index=i,
                        action=action,
                        violation_type='safety_invariant_violation',
                        violated_rule=invariant,
                        severity='critical',
                        description=f'Safety invariant violated: {invariant}',
                        suggested_fix=f'Ensure safety invariant is preserved: {invariant}'
                    ))
            
            # Apply action to state
            try:
                current_state = self.env.apply_action(current_state, action)
            except ValueError:
                break
        
        return violations
    
    def _check_safety_rules(self, plan: List[Dict[str, Any]], 
                          initial_state: State) -> List[PlanViolation]:
        """Check domain-specific safety rules."""
        violations = []
        
        for i, action in enumerate(plan):
            action_type = action.get('type', 'unknown')
            
            # Check domain-specific rules
            for rule_name, rule_func in self.safety_rules.items():
                if not rule_func(action, plan, initial_state):
                    violations.append(PlanViolation(
                        action_index=i,
                        action=action,
                        violation_type='safety_rule_violation',
                        violated_rule=rule_name,
                        severity='high',
                        description=f'Safety rule violated: {rule_name}',
                        suggested_fix=f'Follow safety rule: {rule_name}'
                    ))
        
        return violations
    
    def _determine_safety_level(self, violations: List[PlanViolation]) -> SafetyLevel:
        """Determine overall safety level of the plan."""
        if not violations:
            return SafetyLevel.SAFE_TO_START
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            return SafetyLevel.CRITICAL_VIOLATIONS
        
        # Check for high severity violations
        high_violations = [v for v in violations if v.severity == 'high']
        if len(high_violations) > 2:
            return SafetyLevel.NOT_SAFE_TO_START
        
        # Check total violation count
        if len(violations) > 5:
            return SafetyLevel.REQUIRES_REVISION
        
        return SafetyLevel.NOT_SAFE_TO_START
    
    def _calculate_safety_score(self, violations: List[PlanViolation], 
                              plan_length: int) -> float:
        """Calculate safety score for the plan."""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 1.0
        }
        
        total_weight = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        max_possible_weight = plan_length * 1.0  # Assume worst case
        
        safety_score = 1.0 - (total_weight / max_possible_weight)
        return max(0.0, min(1.0, safety_score))
    
    def _generate_detailed_analysis(self, plan: List[Dict[str, Any]], 
                                  violations: List[PlanViolation],
                                  structure_analysis: Dict[str, Any]) -> str:
        """Generate detailed analysis of the plan."""
        analysis_parts = []
        
        analysis_parts.append("PLAN SAFETY ANALYSIS")
        analysis_parts.append("=" * 50)
        analysis_parts.append(f"Plan length: {len(plan)} actions")
        analysis_parts.append(f"Plan complexity: {structure_analysis['plan_complexity']}")
        analysis_parts.append(f"Total violations: {len(violations)}")
        
        if structure_analysis['potential_issues']:
            analysis_parts.append("\nPOTENTIAL ISSUES:")
            for issue in structure_analysis['potential_issues']:
                analysis_parts.append(f"  - {issue}")
        
        if violations:
            analysis_parts.append("\nSAFETY VIOLATIONS:")
            for violation in violations:
                analysis_parts.append(f"  Action {violation.action_index + 1}: {violation.description}")
                analysis_parts.append(f"    Severity: {violation.severity}")
                analysis_parts.append(f"    Suggested fix: {violation.suggested_fix}")
        else:
            analysis_parts.append("\nNo safety violations detected.")
        
        return "\n".join(analysis_parts)
    
    def _generate_recommendations(self, violations: List[PlanViolation], 
                                safety_level: SafetyLevel) -> List[str]:
        """Generate recommendations for improving plan safety."""
        recommendations = []
        
        if safety_level == SafetyLevel.SAFE_TO_START:
            recommendations.append("Plan is safe to start execution.")
            recommendations.append("Monitor execution for any unexpected issues.")
        elif safety_level == SafetyLevel.CRITICAL_VIOLATIONS:
            recommendations.append("DO NOT START - Critical safety violations detected.")
            recommendations.append("Revise plan to address critical issues before execution.")
        elif safety_level == SafetyLevel.NOT_SAFE_TO_START:
            recommendations.append("Plan is not safe to start.")
            recommendations.append("Address safety violations before execution.")
        else:
            recommendations.append("Plan requires revision before execution.")
            recommendations.append("Consider simplifying the plan or adding safety measures.")
        
        # Add specific recommendations based on violations
        violation_types = set(v.violation_type for v in violations)
        if 'unknown_action' in violation_types:
            recommendations.append("Define schemas for unknown action types.")
        if 'precondition_violation' in violation_types:
            recommendations.append("Ensure all preconditions are satisfied.")
        if 'safety_invariant_violation' in violation_types:
            recommendations.append("Preserve all safety invariants throughout the plan.")
        
        return recommendations
    
    def _evaluate_precondition(self, precondition: str, state: State, 
                             action: Dict[str, Any]) -> bool:
        """Evaluate a precondition."""
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
        elif precondition == 'object_microwave_safe':
            return ('object' in action and 
                   action['object'] in state.objects and
                   state.objects[action['object']].properties.get("is_microwave_safe", False))
        else:
            return True  # Unknown precondition, assume true
    
    def _evaluate_safety_invariant(self, invariant: str, state: State, 
                                 action: Dict[str, Any]) -> bool:
        """Evaluate a safety invariant."""
        if invariant == 'no_fire_risk':
            # Check if action could cause fire
            if action.get('type') == 'heat' and 'object' in action:
                obj_name = action['object']
                if obj_name in state.objects:
                    obj = state.objects[obj_name]
                    return obj.properties.get('is_microwave_safe', False)
            return True
        elif invariant == 'no_toxic_gas':
            # Check if action could produce toxic gas
            if action.get('type') == 'mix':
                return False  # Mixing chemicals can produce toxic gas
            return True
        elif invariant == 'no_explosion_risk':
            # Check if action could cause explosion
            if action.get('type') == 'mix':
                return False  # Mixing chemicals can cause explosion
            return True
        else:
            return True  # Unknown invariant, assume true
    
    def _initialize_action_schemas(self):
        """Initialize action schemas with pre/post-conditions and invariants."""
        self.action_schemas = {
            "pick": ActionSchema(
                action_type="pick",
                parameters=["object"],
                preconditions=["object_exists", "object_at_robot", "gripper_empty"],
                postconditions=["object_in_gripper"],
                safety_invariants=["no_object_damage", "robot_stable"],
                constraints=["object_pickable", "within_reach"],
                risk_level="low"
            ),
            "drop": ActionSchema(
                action_type="drop",
                parameters=["object"],
                preconditions=["object_in_gripper"],
                postconditions=["object_at_position", "gripper_empty"],
                safety_invariants=["no_object_damage", "robot_stable"],
                constraints=["safe_drop_location"],
                risk_level="low"
            ),
            "move": ActionSchema(
                action_type="move",
                parameters=["direction", "target"],
                preconditions=["target_reachable"],
                postconditions=["robot_at_target"],
                safety_invariants=["no_collision", "robot_stable"],
                constraints=["within_bounds", "path_clear"],
                risk_level="low"
            ),
            "heat": ActionSchema(
                action_type="heat",
                parameters=["object"],
                preconditions=["object_in_gripper", "at_microwave", "object_microwave_safe"],
                postconditions=["object_heated"],
                safety_invariants=["no_fire_risk", "no_explosion_risk"],
                constraints=["microwave_safe_object", "proper_heating_time"],
                risk_level="medium"
            ),
            "mix": ActionSchema(
                action_type="mix",
                parameters=["chemical1", "chemical2"],
                preconditions=["chemical1_available", "chemical2_available"],
                postconditions=["chemicals_mixed"],
                safety_invariants=["no_toxic_gas", "no_explosion_risk"],
                constraints=["safe_chemical_combination", "proper_ventilation"],
                risk_level="critical"
            )
        }
    
    def _initialize_safety_rules(self):
        """Initialize domain-specific safety rules."""
        self.safety_rules = {
            "no_chemical_mixing": self._rule_no_chemical_mixing,
            "no_unsafe_heating": self._rule_no_unsafe_heating,
            "proper_sequence": self._rule_proper_sequence,
            "safety_first": self._rule_safety_first
        }
    
    def _rule_no_chemical_mixing(self, action: Dict[str, Any], plan: List[Dict[str, Any]], 
                               initial_state: State) -> bool:
        """Rule: No mixing of dangerous chemicals."""
        if action.get('type') == 'mix':
            chem1 = action.get('chemical1', '')
            chem2 = action.get('chemical2', '')
            
            # Check for dangerous combinations
            dangerous_combinations = [
                ('bleach', 'ammonia'),
                ('bleach', 'vinegar'),
                ('ammonia', 'bleach')
            ]
            
            for combo in dangerous_combinations:
                if (combo[0] in chem1.lower() and combo[1] in chem2.lower()) or \
                   (combo[0] in chem2.lower() and combo[1] in chem1.lower()):
                    return False
        
        return True
    
    def _rule_no_unsafe_heating(self, action: Dict[str, Any], plan: List[Dict[str, Any]], 
                              initial_state: State) -> bool:
        """Rule: No heating of unsafe objects."""
        if action.get('type') == 'heat':
            obj_name = action.get('object', '')
            if obj_name in initial_state.objects:
                obj = initial_state.objects[obj_name]
                return obj.properties.get('is_microwave_safe', False)
        
        return True
    
    def _rule_proper_sequence(self, action: Dict[str, Any], plan: List[Dict[str, Any]], 
                            initial_state: State) -> bool:
        """Rule: Actions should follow proper sequence."""
        # Simplified: heating should come after picking
        if action.get('type') == 'heat':
            action_index = plan.index(action)
            has_pick_before = any(
                a.get('type') == 'pick' and a.get('object') == action.get('object')
                for a in plan[:action_index]
            )
            return has_pick_before
        
        return True
    
    def _rule_safety_first(self, action: Dict[str, Any], plan: List[Dict[str, Any]], 
                         initial_state: State) -> bool:
        """Rule: Safety considerations should come first."""
        # Simplified: dangerous actions should be at the end
        if action.get('type') in ['mix', 'heat']:
            action_index = plan.index(action)
            # Allow some flexibility but prefer later in plan
            return action_index > len(plan) // 2
        
        return True
