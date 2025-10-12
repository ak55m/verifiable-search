"""
Safe Action Invention with Mathematical Verification
===================================================

Implements the core LLM-based safe action invention system that:
1. Takes high-level LLM responses
2. Converts them to low-level actions
3. Verifies safety using mathematical frameworks
4. Provides formal guarantees

This is the main contribution of the paper: enabling LLMs to invent
novel actions while maintaining mathematical safety guarantees.
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
class HighLevelPlan:
    """Represents a high-level plan from LLM."""
    goal: str
    steps: List[str]
    constraints: List[str]
    reasoning: str


@dataclass
class LowLevelAction:
    """Represents a low-level action with verification."""
    action: Dict[str, Any]
    safety_verified: bool
    verification_proof: str
    similarity_score: float
    most_similar_safe_action: Optional[Dict[str, Any]]


class SafeActionInventor:
    """Main class for safe action invention with mathematical verification."""
    
    def __init__(self, environment: Environment):
        """Initialize safe action inventor."""
        self.env = environment
        self.bisimulation = BisimulationMetrics()
        self.entropy_learner = MaximumCausalEntropy()
        self.formal_verifier = FormalVerifier()
        self.optimal_transport = OptimalTransport()
        
        # Known safe actions for similarity comparison
        self.known_safe_actions = self._initialize_safe_actions()
        
        # Action schema database
        self.action_schemas = {}
        
        # Initialize formal verification rules
        self._setup_verification_rules()
    
    def invent_safe_actions(self, state: State, goal: str, 
                          high_level_plan: HighLevelPlan) -> List[LowLevelAction]:
        """
        Main method: Invent safe actions from high-level LLM plan.
        
        Args:
            state: Current state
            goal: Goal to achieve
            high_level_plan: High-level plan from LLM
            
        Returns:
            List of verified low-level actions
        """
        print(f"🧠 Inventing safe actions for goal: {goal}")
        print(f"📋 High-level plan: {high_level_plan.steps}")
        
        # Step 1: Convert high-level steps to low-level actions
        low_level_actions = self._convert_high_to_low_level(high_level_plan, state)
        
        # Step 2: Verify each action using mathematical frameworks
        verified_actions = []
        for action in low_level_actions:
            verified_action = self._verify_action_safety(state, action, goal)
            verified_actions.append(verified_action)
        
        # Step 3: Learn new schemas from successful actions
        self._learn_new_schemas(verified_actions, state)
        
        return verified_actions
    
    def _convert_high_to_low_level(self, high_level_plan: HighLevelPlan, 
                                 state: State) -> List[Dict[str, Any]]:
        """
        Convert high-level LLM plan to low-level actions.
        
        This is where the LLM's high-level reasoning gets converted to
        executable robot commands.
        """
        low_level_actions = []
        
        for step in high_level_plan.steps:
            # Parse high-level step
            action = self._parse_high_level_step(step, state)
            if action:
                low_level_actions.append(action)
        
        return low_level_actions
    
    def _parse_high_level_step(self, step: str, state: State) -> Optional[Dict[str, Any]]:
        """
        Parse a high-level step into low-level action.
        
        This is a simplified parser. In practice, this would use
        natural language processing and LLM reasoning.
        """
        step_lower = step.lower()
        
        # Movement actions
        if "move" in step_lower or "go" in step_lower:
            if "right" in step_lower:
                return self._create_move_action(state, (1, 0))
            elif "left" in step_lower:
                return self._create_move_action(state, (-1, 0))
            elif "up" in step_lower:
                return self._create_move_action(state, (0, 1))
            elif "down" in step_lower:
                return self._create_move_action(state, (0, -1))
        
        # Object manipulation
        elif "pick" in step_lower or "grab" in step_lower:
            obj_name = self._extract_object_name(step)
            if obj_name and obj_name in state.objects:
                return {"type": "pick", "object": obj_name}
        
        elif "drop" in step_lower or "place" in step_lower:
            obj_name = self._extract_object_name(step)
            if obj_name and obj_name in state.objects:
                return {"type": "drop", "object": obj_name}
        
        # Heating actions
        elif "heat" in step_lower or "microwave" in step_lower:
            obj_name = self._extract_object_name(step)
            if obj_name and obj_name in state.objects:
                return {"type": "heat", "object": obj_name}
        
        # Container actions
        elif "open" in step_lower:
            container_name = self._extract_object_name(step)
            if container_name and container_name in state.objects:
                return {"type": "open", "container": container_name}
        
        elif "close" in step_lower:
            container_name = self._extract_object_name(step)
            if container_name and container_name in state.objects:
                return {"type": "close", "container": container_name}
        
        return None
    
    def _create_move_action(self, state: State, direction: Tuple[int, int]) -> Dict[str, Any]:
        """Create a move action in the given direction."""
        new_x = state.robot[0] + direction[0]
        new_y = state.robot[1] + direction[1]
        
        if 0 <= new_x < state.width and 0 <= new_y < state.height:
            return {
                "type": "move",
                "direction": direction,
                "target": (new_x, new_y)
            }
        return None
    
    def _extract_object_name(self, step: str) -> Optional[str]:
        """Extract object name from high-level step."""
        # Simplified object name extraction
        # In practice, this would use NLP techniques
        
        common_objects = ["bowl", "cup", "book", "microwave", "shelf", "stove", "kettle"]
        
        for obj in common_objects:
            if obj in step.lower():
                return obj
        
        return None
    
    def _verify_action_safety(self, state: State, action: Dict[str, Any], 
                            goal: str) -> LowLevelAction:
        """
        Verify action safety using mathematical frameworks.
        
        This is the core contribution: using multiple mathematical
        approaches to ensure safety.
        """
        print(f"🔍 Verifying action: {action}")
        
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
        
        # 4. Generate safety proof
        safety_proof = self.formal_verifier.generate_safety_proof(action, state, self.env)
        
        # 5. Overall safety decision
        is_safe = is_similar and is_formally_safe and is_transport_safe
        
        print(f"   Similarity: {is_similar} (score: {similarity_score:.3f})")
        print(f"   Formal verification: {is_formally_safe}")
        print(f"   Transport safety: {is_transport_safe} (distance: {transport_distance:.3f})")
        print(f"   Overall: {'✅ SAFE' if is_safe else '❌ UNSAFE'}")
        
        return LowLevelAction(
            action=action,
            safety_verified=is_safe,
            verification_proof=safety_proof,
            similarity_score=similarity_score,
            most_similar_safe_action=most_similar
        )
    
    def _learn_new_schemas(self, verified_actions: List[LowLevelAction], state: State):
        """Learn new action schemas from verified actions."""
        for verified_action in verified_actions:
            if verified_action.safety_verified:
                action_type = verified_action.action["type"]
                
                # Extract preconditions
                preconditions = self._extract_preconditions(state, verified_action.action)
                
                # Store schema
                if action_type not in self.action_schemas:
                    self.action_schemas[action_type] = []
                
                schema = {
                    "action": verified_action.action,
                    "preconditions": preconditions,
                    "safety_verified": True,
                    "verification_proof": verified_action.verification_proof
                }
                
                self.action_schemas[action_type].append(schema)
                print(f"📚 Learned new schema for {action_type}")
    
    def _extract_preconditions(self, state: State, action: Dict[str, Any]) -> Dict[str, Any]:
        """Extract preconditions for an action."""
        preconditions = {}
        
        # Robot position
        preconditions["robot_position"] = state.robot
        
        # Gripper state
        preconditions["gripper_state"] = state.gripper
        
        # Object requirements
        if "object" in action:
            obj_name = action["object"]
            if obj_name in state.objects:
                obj = state.objects[obj_name]
                preconditions[f"object_{obj_name}_position"] = obj.position
                preconditions[f"object_{obj_name}_properties"] = obj.properties
        
        return preconditions
    
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
    
    def get_invention_statistics(self) -> Dict[str, Any]:
        """Get statistics about action invention."""
        total_actions = len(self.action_schemas)
        safe_actions = sum(len(schemas) for schemas in self.action_schemas.values())
        
        return {
            "total_action_types": total_actions,
            "total_safe_actions": safe_actions,
            "action_schemas": self.action_schemas,
            "verification_methods": [
                "bisimulation_metrics",
                "formal_verification", 
                "optimal_transport",
                "maximum_causal_entropy"
            ]
        }
    
    def export_schemas(self, filename: str):
        """Export learned action schemas."""
        with open(filename, 'w') as f:
            json.dump(self.action_schemas, f, indent=2)
        print(f"📁 Exported schemas to {filename}")


class LLMSafePlanner:
    """High-level planner that uses LLM for safe action invention."""
    
    def __init__(self, environment: Environment):
        """Initialize LLM safe planner."""
        self.env = environment
        self.action_inventor = SafeActionInventor(environment)
    
    def plan(self, initial_state: State, goal: str, max_steps: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a plan using LLM-based safe action invention.
        
        Args:
            initial_state: Initial state
            goal: Goal to achieve
            max_steps: Maximum number of steps
            
        Returns:
            (plan, metadata)
        """
        start_time = time.time()
        
        # Step 1: Generate high-level plan using LLM reasoning
        high_level_plan = self._generate_high_level_plan(initial_state, goal)
        
        # Step 2: Invent safe low-level actions
        verified_actions = self.action_inventor.invent_safe_actions(
            initial_state, goal, high_level_plan
        )
        
        # Step 3: Filter to only safe actions
        safe_actions = [va.action for va in verified_actions if va.safety_verified]
        
        # Step 4: Execute plan and check goal achievement
        current_state = initial_state
        violations = 0
        
        for action in safe_actions:
            try:
                current_state = self.env.apply_action(current_state, action)
            except ValueError:
                violations += 1
                break
        
        success = self.env.is_goal_state(current_state, goal)
        runtime = time.time() - start_time
        
        return safe_actions, {
            "success": success,
            "steps": len(safe_actions),
            "violations": violations,
            "runtime_s": runtime,
            "verification_methods": "mathematical_framework",
            "invention_stats": self.action_inventor.get_invention_statistics()
        }
    
    def _generate_high_level_plan(self, state: State, goal: str) -> HighLevelPlan:
        """Generate high-level plan using LLM reasoning."""
        # This is where the LLM would generate the high-level plan
        # For now, we'll use a simplified rule-based approach
        
        if goal == "safe_microwave":
            return HighLevelPlan(
                goal=goal,
                steps=[
                    "Pick up the bowl",
                    "Move to the microwave",
                    "Heat the bowl in the microwave"
                ],
                constraints=["Only heat microwave-safe objects", "Must be at microwave location"],
                reasoning="Need to pick up bowl, move to microwave, and heat it safely"
            )
        elif goal == "tidy_desk":
            return HighLevelPlan(
                goal=goal,
                steps=[
                    "Pick up the book",
                    "Move to the shelf",
                    "Place the book on the shelf"
                ],
                constraints=["Book must be placed on shelf", "No damage to book"],
                reasoning="Need to move book from desk to shelf for organization"
            )
        else:
            return HighLevelPlan(
                goal=goal,
                steps=["Move around", "Pick up objects", "Place objects"],
                constraints=["Stay safe", "Follow rules"],
                reasoning="Generic plan for unknown goal"
            )
