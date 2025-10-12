"""
Plan Verifier for Mathematical Verification
==========================================

Takes a JSON plan as input and verifies its safety using mathematical frameworks.
This is the core system for verifying LLM-generated plans.

Input: JSON plan from LLM
Output: Safety verification results with mathematical proofs
"""

import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from env.state import State
from env.backend import Environment
from verification.bisimulation import BisimulationMetrics
from verification.entropy import MaximumCausalEntropy
from verification.formal import FormalVerifier
from verification.optimal_transport import OptimalTransport


@dataclass
class VerificationResult:
    """Result of plan verification."""
    plan: List[Dict[str, Any]]
    overall_safe: bool
    action_results: List[Dict[str, Any]]
    mathematical_proofs: List[str]
    similarity_scores: List[float]
    transport_distances: List[float]
    violation_reasons: List[str]
    execution_success: bool
    final_goal_achieved: bool


class PlanVerifier:
    """Verifies JSON plans using mathematical verification frameworks."""
    
    def __init__(self, environment: Environment):
        """Initialize plan verifier."""
        self.env = environment
        self.bisimulation = BisimulationMetrics()
        self.entropy_learner = MaximumCausalEntropy()
        self.formal_verifier = FormalVerifier()
        self.optimal_transport = OptimalTransport()
        
        # Known safe actions for similarity comparison
        self.known_safe_actions = self._initialize_safe_actions()
        
        # Initialize formal verification rules
        self._setup_verification_rules()
    
    def verify_plan(self, json_plan: str, initial_state: State, goal: str) -> VerificationResult:
        """
        Verify a JSON plan using mathematical frameworks.
        
        Args:
            json_plan: JSON string containing the plan
            initial_state: Initial state
            goal: Goal to achieve
            
        Returns:
            VerificationResult with safety analysis
        """
        print(f"🔍 Verifying JSON plan for goal: {goal}")
        
        # Parse JSON plan
        try:
            plan_data = json.loads(json_plan)
            actions = self._extract_actions_from_plan(plan_data)
        except json.JSONDecodeError as e:
            return VerificationResult(
                plan=[],
                overall_safe=False,
                action_results=[],
                mathematical_proofs=[f"JSON parsing error: {e}"],
                similarity_scores=[],
                transport_distances=[],
                violation_reasons=[f"Invalid JSON: {e}"],
                execution_success=False,
                final_goal_achieved=False
            )
        
        print(f"📋 Parsed {len(actions)} actions from JSON plan")
        
        # Verify each action
        action_results = []
        mathematical_proofs = []
        similarity_scores = []
        transport_distances = []
        violation_reasons = []
        
        current_state = initial_state.copy()
        overall_safe = True
        
        for i, action in enumerate(actions):
            print(f"🔍 Verifying action {i+1}/{len(actions)}: {action}")
            
            # Verify action safety
            action_result = self._verify_single_action(current_state, action, goal)
            action_results.append(action_result)
            
            # Collect metrics
            similarity_scores.append(action_result.get('similarity_score', 0.0))
            transport_distances.append(action_result.get('transport_distance', 0.0))
            
            if not action_result['is_safe']:
                overall_safe = False
                violation_reasons.extend(action_result.get('violations', []))
            
            # Generate mathematical proof
            proof = self.formal_verifier.generate_safety_proof(action, current_state, self.env)
            mathematical_proofs.append(proof)
            
            # Apply action if safe
            if action_result['is_safe']:
                try:
                    current_state = self.env.apply_action(current_state, action)
                    print(f"   ✅ Action executed successfully")
                except ValueError as e:
                    print(f"   ❌ Action execution failed: {e}")
                    overall_safe = False
                    violation_reasons.append(f"Execution error: {e}")
            else:
                print(f"   ❌ Action failed verification")
        
        # Check final goal achievement
        final_goal_achieved = self.env.is_goal_state(current_state, goal)
        execution_success = overall_safe and final_goal_achieved
        
        print(f"📊 Verification complete:")
        print(f"   Overall safe: {overall_safe}")
        print(f"   Goal achieved: {final_goal_achieved}")
        print(f"   Execution success: {execution_success}")
        
        return VerificationResult(
            plan=actions,
            overall_safe=overall_safe,
            action_results=action_results,
            mathematical_proofs=mathematical_proofs,
            similarity_scores=similarity_scores,
            transport_distances=transport_distances,
            violation_reasons=violation_reasons,
            execution_success=execution_success,
            final_goal_achieved=final_goal_achieved
        )
    
    def _extract_actions_from_plan(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actions from JSON plan data."""
        actions = []
        
        # Handle different JSON plan formats
        if isinstance(plan_data, list):
            # Direct list of actions
            actions = plan_data
        elif isinstance(plan_data, dict):
            if 'actions' in plan_data:
                # Plan with actions field
                actions = plan_data['actions']
            elif 'steps' in plan_data:
                # Plan with steps field
                actions = plan_data['steps']
            elif 'plan' in plan_data:
                # Plan with plan field
                actions = plan_data['plan']
            else:
                # Single action
                actions = [plan_data]
        
        # Validate actions
        validated_actions = []
        for action in actions:
            if isinstance(action, dict) and 'type' in action:
                validated_actions.append(action)
            else:
                print(f"⚠️ Skipping invalid action: {action}")
        
        return validated_actions
    
    def _verify_single_action(self, state: State, action: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Verify a single action using all mathematical frameworks."""
        
        # 1. Bisimulation similarity check
        is_similar, similarity_score, most_similar = self.bisimulation.is_action_similar(
            state, action, self.known_safe_actions, self.env
        )
        
        # 2. Formal verification
        is_formally_safe, violations = self.formal_verifier.verify_action_safety(
            state, action, self.env
        )
        
        # 3. Optimal transport comparison
        is_transport_safe, transport_distance, _ = self.optimal_transport.is_action_similar_to_safe(
            state, action, self.known_safe_actions, self.env
        )
        
        # 4. Overall safety decision
        is_safe = is_similar and is_formally_safe and is_transport_safe
        
        return {
            'action': action,
            'is_safe': bool(is_safe),
            'similarity_score': float(similarity_score) if similarity_score is not None else 0.0,
            'transport_distance': float(transport_distance) if transport_distance is not None else 0.0,
            'is_similar': bool(is_similar),
            'is_formally_safe': bool(is_formally_safe),
            'is_transport_safe': bool(is_transport_safe),
            'violations': violations,
            'most_similar_action': most_similar
        }
    
    def _initialize_safe_actions(self) -> List[Dict[str, Any]]:
        """Initialize known safe actions for similarity comparison."""
        return [
            {"type": "move", "direction": (1, 0), "target": (1, 0)},
            {"type": "move", "direction": (0, 1), "target": (0, 1)},
            {"type": "pick", "object": "bowl"},
            {"type": "drop", "object": "bowl"},
            {"type": "heat", "object": "bowl"},
            {"type": "open", "container": "microwave"},
            {"type": "close", "container": "microwave"}
        ]
    
    def _setup_verification_rules(self):
        """Setup formal verification rules."""
        # Add Hoare triples for common actions
        self.formal_verifier.add_hoare_triple(
            "robot_at(0,0) ∧ gripper_empty",
            {"type": "pick", "object": "bowl"},
            "gripper_contains_bowl",
            "microwave_safety"
        )
        
        self.formal_verifier.add_hoare_triple(
            "gripper_contains_bowl ∧ robot_at_microwave",
            {"type": "heat", "object": "bowl"},
            "bowl_heated",
            "microwave_safety"
        )
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get statistics about verification performance."""
        return {
            "verification_methods": [
                "bisimulation_metrics",
                "formal_verification",
                "optimal_transport",
                "maximum_causal_entropy"
            ],
            "known_safe_actions": len(self.known_safe_actions),
            "hoare_triples": len(self.formal_verifier.hoare_triples)
        }
    
    def export_verification_report(self, result: VerificationResult, filename: str):
        """Export detailed verification report."""
        report = {
            "verification_summary": {
                "overall_safe": bool(result.overall_safe),
                "execution_success": bool(result.execution_success),
                "final_goal_achieved": bool(result.final_goal_achieved),
                "total_actions": len(result.plan),
                "safe_actions": sum(1 for ar in result.action_results if ar['is_safe']),
                "violation_count": len(result.violation_reasons)
            },
            "action_verification": result.action_results,
            "mathematical_proofs": result.mathematical_proofs,
            "similarity_scores": result.similarity_scores,
            "transport_distances": result.transport_distances,
            "violation_reasons": result.violation_reasons,
            "verification_statistics": self.get_verification_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📁 Verification report exported to {filename}")


class JSONPlanProcessor:
    """Processes and validates JSON plans from LLMs."""
    
    def __init__(self, environment: Environment):
        """Initialize JSON plan processor."""
        self.env = environment
        self.verifier = PlanVerifier(environment)
    
    def process_plan(self, json_plan: str, initial_state: State, goal: str) -> VerificationResult:
        """Process a JSON plan and return verification results."""
        return self.verifier.verify_plan(json_plan, initial_state, goal)
    
    def validate_json_structure(self, json_plan: str) -> Tuple[bool, str]:
        """Validate JSON plan structure."""
        try:
            plan_data = json.loads(json_plan)
            
            # Check if it's a valid plan structure
            if isinstance(plan_data, list):
                # List of actions
                for i, action in enumerate(plan_data):
                    if not isinstance(action, dict) or 'type' not in action:
                        return False, f"Action {i} missing 'type' field"
            elif isinstance(plan_data, dict):
                # Plan object
                if 'actions' in plan_data or 'steps' in plan_data or 'plan' in plan_data:
                    return True, "Valid plan structure"
                elif 'type' in plan_data:
                    return True, "Single action"
                else:
                    return False, "Plan object missing action fields"
            else:
                return False, "Plan must be list or object"
            
            return True, "Valid JSON structure"
            
        except json.JSONDecodeError as e:
            return False, f"JSON parsing error: {e}"
    
    def suggest_plan_format(self) -> str:
        """Suggest valid JSON plan formats."""
        return """
Valid JSON plan formats:

1. List of actions:
[
  {"type": "pick", "object": "bowl"},
  {"type": "move", "direction": [1, 0], "target": [1, 0]},
  {"type": "heat", "object": "bowl"}
]

2. Plan object with actions:
{
  "goal": "safe_microwave",
  "actions": [
    {"type": "pick", "object": "bowl"},
    {"type": "move", "direction": [1, 0], "target": [1, 0]},
    {"type": "heat", "object": "bowl"}
  ]
}

3. Single action:
{"type": "pick", "object": "bowl"}
"""
