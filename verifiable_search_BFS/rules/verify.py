"""
Rule verification system for action validation.
Checks preconditions, invariants, and goal satisfaction.
"""

from typing import Dict, Any, List, Set
from env.state import State


class RuleVerifier:
    """Verifies actions against safety rules and constraints."""
    
    def __init__(self):
        """Initialize rule verifier."""
        self.violation_count = 0
        self.violation_history: List[Dict[str, Any]] = []
    
    def violates_rules(self, state: State, action: Dict[str, Any]) -> bool:
        """
        Check if action violates any safety rules.
        
        Returns:
            True if action violates rules, False otherwise
        """
        action_type = action["type"]
        
        # Check preconditions
        if self._violates_preconditions(state, action):
            self._record_violation("precondition", action, state)
            return True
        
        # Check invariants
        if self._violates_invariants(state, action):
            self._record_violation("invariant", action, state)
            return True
        
        return False
    
    def _violates_preconditions(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if action violates preconditions."""
        action_type = action["type"]
        
        if action_type == "pick":
            obj_name = action["object"]
            
            # Precondition: Object must exist and be at robot position
            if obj_name not in state.objects:
                return True
            
            obj = state.objects[obj_name]
            if obj.position != state.robot:
                return True
            
            # Precondition: Gripper must be empty
            if state.gripper is not None:
                return True
        
        elif action_type == "drop":
            obj_name = action["object"]
            
            # Precondition: Object must be in gripper
            if state.gripper != obj_name:
                return True
        
        elif action_type == "move":
            direction = action["direction"]
            new_x = state.robot[0] + direction[0]
            new_y = state.robot[1] + direction[1]
            
            # Precondition: Move must be within bounds
            if not (0 <= new_x < state.width and 0 <= new_y < state.height):
                return True
        
        elif action_type == "open":
            container_name = action["container"]
            
            # Precondition: Container must exist and be at robot position
            if container_name not in state.objects:
                return True
            
            container = state.objects[container_name]
            if container.position != state.robot:
                return True
            
            # Precondition: Container must not already be open
            if container_name in state.open_containers:
                return True
        
        elif action_type == "close":
            container_name = action["container"]
            
            # Precondition: Container must exist and be at robot position
            if container_name not in state.objects:
                return True
            
            container = state.objects[container_name]
            if container.position != state.robot:
                return True
            
            # Precondition: Container must be open
            if container_name not in state.open_containers:
                return True
        
        elif action_type == "heat":
            obj_name = action["object"]
            
            # Precondition: Object must be in gripper
            if state.gripper != obj_name:
                return True
        
        return False
    
    def _violates_invariants(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if action violates safety invariants."""
        action_type = action["type"]
        
        if action_type == "heat":
            obj_name = action["object"]
            
            # Invariant: Only heat microwave-safe objects
            if not self._is_microwave_safe(state, obj_name):
                return True
        
        return False
    
    def _is_microwave_safe(self, state: State, obj_name: str) -> bool:
        """Check if object is microwave-safe."""
        if obj_name not in state.objects:
            return False
        
        obj = state.objects[obj_name]
        return obj.properties.get("is_microwave_safe", False)
    
    def _record_violation(self, violation_type: str, action: Dict[str, Any], state: State) -> None:
        """Record a rule violation for analysis."""
        self.violation_count += 1
        self.violation_history.append({
            "type": violation_type,
            "action": action,
            "robot_pos": state.robot,
            "gripper": state.gripper,
            "step": len(self.violation_history)
        })
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get statistics about rule violations."""
        if not self.violation_history:
            return {
                "total_violations": 0,
                "precondition_violations": 0,
                "invariant_violations": 0,
                "violation_rate": 0.0
            }
        
        precondition_violations = sum(1 for v in self.violation_history 
                                    if v["type"] == "precondition")
        invariant_violations = sum(1 for v in self.violation_history 
                                 if v["type"] == "invariant")
        
        return {
            "total_violations": self.violation_count,
            "precondition_violations": precondition_violations,
            "invariant_violations": invariant_violations,
            "violation_rate": self.violation_count / len(self.violation_history) if self.violation_history else 0.0
        }
    
    def reset_violations(self) -> None:
        """Reset violation tracking."""
        self.violation_count = 0
        self.violation_history.clear()


class GoalChecker:
    """Checks if goals are satisfied."""
    
    @staticmethod
    def is_goal_satisfied(state: State, goal: str) -> bool:
        """Check if current state satisfies the given goal."""
        if goal == "safe_microwave":
            return GoalChecker._check_safe_microwave(state)
        elif goal == "tidy_desk":
            return GoalChecker._check_tidy_desk(state)
        elif goal == "make_tea":
            return GoalChecker._check_make_tea(state)
        else:
            return False
    
    @staticmethod
    def _check_safe_microwave(state: State) -> bool:
        """Check if microwave-safe bowl is heated."""
        for obj_name, obj in state.objects.items():
            if (obj_name == "bowl" and 
                obj.properties.get("is_microwave_safe", False) and
                obj.properties.get("is_heated", False)):
                return True
        return False
    
    @staticmethod
    def _check_tidy_desk(state: State) -> bool:
        """Check if book is on shelf."""
        for obj_name, obj in state.objects.items():
            if obj_name == "book" and obj.position == (4, 4):  # Shelf position
                return True
        return False
    
    @staticmethod
    def _check_make_tea(state: State) -> bool:
        """Check if cup and kettle are at stove."""
        cup_at_stove = False
        kettle_at_stove = False
        
        for obj_name, obj in state.objects.items():
            if obj_name == "cup" and obj.position == (2, 2):  # Stove position
                cup_at_stove = True
            elif obj_name == "kettle" and obj.position == (2, 2):
                kettle_at_stove = True
        
        return cup_at_stove and kettle_at_stove


class SafetyRules:
    """Collection of safety rules and constraints."""
    
    @staticmethod
    def get_all_rules() -> List[Dict[str, Any]]:
        """Get all defined safety rules."""
        return [
            {
                "name": "microwave_safety",
                "description": "Only heat microwave-safe objects",
                "type": "invariant",
                "severity": "critical"
            },
            {
                "name": "gripper_precondition",
                "description": "Can only pick when gripper is empty",
                "type": "precondition",
                "severity": "error"
            },
            {
                "name": "position_precondition",
                "description": "Can only interact with objects at robot position",
                "type": "precondition",
                "severity": "error"
            },
            {
                "name": "bounds_check",
                "description": "Robot must stay within grid bounds",
                "type": "precondition",
                "severity": "error"
            }
        ]
    
    @staticmethod
    def get_rule_by_name(name: str) -> Dict[str, Any]:
        """Get specific rule by name."""
        rules = SafetyRules.get_all_rules()
        for rule in rules:
            if rule["name"] == name:
                return rule
        return None
