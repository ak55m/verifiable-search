"""
Enhanced simulator for mathematical verification framework.
Includes support for verification logging and mathematical analysis.
"""

from typing import Dict, Any, Tuple, List
import time
from .state import State, Object


class Simulator:
    """Enhanced simulator for action execution with verification support."""
    
    def __init__(self):
        """Initialize simulator."""
        self.execution_log = []  # Log of all action executions
        self.verification_metrics = {}  # Metrics for verification
    
    def apply_action(self, state: State, action: Dict[str, Any]) -> State:
        """Apply action to state and return new state."""
        start_time = time.time()
        
        new_state = state.copy()
        
        action_type = action["type"]
        
        if action_type == "move":
            self._apply_move(new_state, action)
        elif action_type == "pick":
            self._apply_pick(new_state, action)
        elif action_type == "drop":
            self._apply_drop(new_state, action)
        elif action_type == "open":
            self._apply_open(new_state, action)
        elif action_type == "close":
            self._apply_close(new_state, action)
        elif action_type == "heat":
            self._apply_heat(new_state, action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        # Log execution for verification
        execution_time = time.time() - start_time
        self._log_action_execution(state, action, new_state, execution_time)
        
        return new_state
    
    def _apply_move(self, state: State, action: Dict[str, Any]) -> None:
        """Apply move action with verification."""
        direction = action["direction"]
        new_x = state.robot[0] + direction[0]
        new_y = state.robot[1] + direction[1]
        
        # Validate move is within bounds
        if not (0 <= new_x < state.width and 0 <= new_y < state.height):
            raise ValueError(f"Move to ({new_x}, {new_y}) is out of bounds")
        
        state.robot = (new_x, new_y)
    
    def _apply_pick(self, state: State, action: Dict[str, Any]) -> None:
        """Apply pick action with verification."""
        obj_name = action["object"]
        
        # Validate object exists and is at robot position
        if obj_name not in state.objects:
            raise ValueError(f"Object {obj_name} does not exist")
        
        obj = state.objects[obj_name]
        if obj.position != state.robot:
            raise ValueError(f"Object {obj_name} is not at robot position")
        
        # Validate gripper is empty
        if state.gripper is not None:
            raise ValueError("Gripper is not empty")
        
        state.gripper = obj_name
    
    def _apply_drop(self, state: State, action: Dict[str, Any]) -> None:
        """Apply drop action with verification."""
        obj_name = action["object"]
        
        # Validate object is in gripper
        if state.gripper != obj_name:
            raise ValueError(f"Object {obj_name} is not in gripper")
        
        # Update object position to robot position
        state.objects[obj_name].position = state.robot
        state.gripper = None
    
    def _apply_open(self, state: State, action: Dict[str, Any]) -> None:
        """Apply open container action with verification."""
        container_name = action["container"]
        
        # Validate container exists and is at robot position
        if container_name not in state.objects:
            raise ValueError(f"Container {container_name} does not exist")
        
        container = state.objects[container_name]
        if container.position != state.robot:
            raise ValueError(f"Container {container_name} is not at robot position")
        
        # Validate container is not already open
        if container_name in state.open_containers:
            raise ValueError(f"Container {container_name} is already open")
        
        state.open_containers.add(container_name)
    
    def _apply_close(self, state: State, action: Dict[str, Any]) -> None:
        """Apply close container action with verification."""
        container_name = action["container"]
        
        # Validate container exists and is at robot position
        if container_name not in state.objects:
            raise ValueError(f"Container {container_name} does not exist")
        
        container = state.objects[container_name]
        if container.position != state.robot:
            raise ValueError(f"Container {container_name} is not at robot position")
        
        # Validate container is open
        if container_name not in state.open_containers:
            raise ValueError(f"Container {container_name} is not open")
        
        state.open_containers.remove(container_name)
    
    def _apply_heat(self, state: State, action: Dict[str, Any]) -> None:
        """Apply heat action with verification."""
        obj_name = action["object"]
        
        # Validate object exists and is in gripper
        if obj_name not in state.objects:
            raise ValueError(f"Object {obj_name} does not exist")
        
        if state.gripper != obj_name:
            raise ValueError(f"Object {obj_name} is not in gripper")
        
        # Mark object as heated
        state.objects[obj_name].properties["is_heated"] = True
    
    def _log_action_execution(self, state: State, action: Dict[str, Any], 
                            new_state: State, execution_time: float) -> None:
        """Log action execution for verification and analysis."""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "pre_state": state.to_dict(),
            "post_state": new_state.to_dict(),
            "execution_time": execution_time,
            "safety_score_pre": state.get_safety_score(),
            "safety_score_post": new_state.get_safety_score()
        }
        
        self.execution_log.append(log_entry)
        
        # Update verification metrics
        self._update_verification_metrics(log_entry)
    
    def _update_verification_metrics(self, log_entry: Dict[str, Any]) -> None:
        """Update verification metrics based on action execution."""
        action_type = log_entry["action"]["type"]
        
        if action_type not in self.verification_metrics:
            self.verification_metrics[action_type] = {
                "count": 0,
                "total_time": 0.0,
                "safety_improvements": 0,
                "safety_degradations": 0
            }
        
        metrics = self.verification_metrics[action_type]
        metrics["count"] += 1
        metrics["total_time"] += log_entry["execution_time"]
        
        safety_pre = log_entry["safety_score_pre"]
        safety_post = log_entry["safety_score_post"]
        
        if safety_post > safety_pre:
            metrics["safety_improvements"] += 1
        elif safety_post < safety_pre:
            metrics["safety_degradations"] += 1
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about action executions."""
        total_actions = len(self.execution_log)
        total_time = sum(entry["execution_time"] for entry in self.execution_log)
        
        return {
            "total_actions": total_actions,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total_actions if total_actions > 0 else 0,
            "action_metrics": self.verification_metrics,
            "safety_trend": self._compute_safety_trend()
        }
    
    def _compute_safety_trend(self) -> List[float]:
        """Compute safety trend over time."""
        if not self.execution_log:
            return []
        
        return [entry["safety_score_post"] for entry in self.execution_log]
    
    def get_action_success_rate(self, action_type: str) -> float:
        """Get success rate for a specific action type."""
        if action_type not in self.verification_metrics:
            return 0.0
        
        metrics = self.verification_metrics[action_type]
        total_attempts = metrics["count"]
        successful_attempts = total_attempts - metrics["safety_degradations"]
        
        return successful_attempts / total_attempts if total_attempts > 0 else 0.0
    
    def clear_log(self) -> None:
        """Clear execution log and metrics."""
        self.execution_log.clear()
        self.verification_metrics.clear()
    
    def export_execution_log(self, filename: str) -> None:
        """Export execution log to file."""
        import json
        
        with open(filename, 'w') as f:
            json.dump({
                "execution_log": self.execution_log,
                "verification_metrics": self.verification_metrics,
                "statistics": self.get_execution_statistics()
            }, f, indent=2)