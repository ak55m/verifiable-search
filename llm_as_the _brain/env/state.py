"""
Enhanced state representation for mathematical verification framework.
Includes features needed for bisimulation, optimal transport, and formal verification.
"""

from typing import Dict, Set, Optional, Tuple, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class Object:
    """Enhanced object representation with mathematical properties."""
    position: Tuple[int, int]
    properties: Dict[str, Any]
    
    def __post_init__(self):
        """Validate object properties."""
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError("Position must be a tuple of (x, y)")
        if not all(isinstance(coord, int) for coord in self.position):
            raise ValueError("Position coordinates must be integers")
    
    def get_feature_vector(self) -> np.ndarray:
        """Get mathematical feature vector for this object."""
        features = []
        features.extend(self.position)  # Position
        features.append(1.0 if self.properties.get("is_microwave_safe", False) else 0.0)
        features.append(1.0 if self.properties.get("is_heated", False) else 0.0)
        features.append(1.0 if self.properties.get("is_container", False) else 0.0)
        features.append(1.0 if self.properties.get("is_microwave", False) else 0.0)
        features.append(1.0 if self.properties.get("is_breakable", False) else 0.0)
        features.append(1.0 if self.properties.get("is_heavy", False) else 0.0)
        return np.array(features)


@dataclass
class State:
    """Enhanced state representation for mathematical verification."""
    width: int
    height: int
    robot: Tuple[int, int]
    gripper: Optional[str]
    objects: Dict[str, Object]
    open_containers: Set[str]
    
    def __post_init__(self):
        """Validate state properties."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if not (0 <= self.robot[0] < self.width and 0 <= self.robot[1] < self.height):
            raise ValueError("Robot position must be within grid bounds")
        if self.gripper is not None and self.gripper not in self.objects:
            raise ValueError("Gripper object must exist in objects dict")
    
    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        return State(
            width=self.width,
            height=self.height,
            robot=self.robot,
            gripper=self.gripper,
            objects={name: Object(obj.position, obj.properties.copy()) 
                    for name, obj in self.objects.items()},
            open_containers=self.open_containers.copy()
        )
    
    def get_object_at_position(self, pos: Tuple[int, int]) -> Optional[str]:
        """Get the name of object at given position, or None if empty."""
        for name, obj in self.objects.items():
            if obj.position == pos:
                return name
        return None
    
    def is_position_occupied(self, pos: Tuple[int, int]) -> bool:
        """Check if position is occupied by any object."""
        return self.get_object_at_position(pos) is not None
    
    def get_robot_objects(self) -> List[str]:
        """Get names of all objects at robot's current position."""
        robot_pos = self.robot
        return [name for name, obj in self.objects.items() 
                if obj.position == robot_pos]
    
    def get_mathematical_features(self) -> np.ndarray:
        """Get mathematical feature vector for this state."""
        features = []
        
        # Robot position (normalized)
        features.extend([self.robot[0] / self.width, self.robot[1] / self.height])
        
        # Gripper state
        features.append(1.0 if self.gripper is not None else 0.0)
        
        # Object features (sorted for consistency)
        for obj_name in sorted(self.objects.keys()):
            obj = self.objects[obj_name]
            features.extend(obj.get_feature_vector())
        
        # Open containers
        features.append(len(self.open_containers))
        
        return np.array(features)
    
    def get_transition_probability(self, action: Dict[str, Any]) -> float:
        """Get probability of successful transition for action."""
        # Simplified probability model
        # In practice, this would use learned transition models
        
        if action["type"] == "move":
            return 1.0  # Movement always succeeds if valid
        elif action["type"] in ["pick", "drop"]:
            return 0.9  # 90% success rate for manipulation
        elif action["type"] == "heat":
            return 0.8  # 80% success rate for heating
        else:
            return 0.7  # Default success rate
    
    def get_safety_score(self) -> float:
        """Get safety score for this state (0-1, higher is safer)."""
        score = 1.0
        
        # Check robot position
        if not (0 <= self.robot[0] < self.width and 0 <= self.robot[1] < self.height):
            score -= 0.5
        
        # Check object positions
        for obj in self.objects.values():
            if not (0 <= obj.position[0] < self.width and 0 <= obj.position[1] < self.height):
                score -= 0.1
        
        # Check gripper consistency
        if self.gripper is not None and self.gripper not in self.objects:
            score -= 0.3
        
        return max(0.0, score)
    
    def __hash__(self) -> int:
        """Enhanced hash function for state comparison."""
        obj_positions = tuple(sorted((name, obj.position) for name, obj in self.objects.items()))
        obj_properties = tuple(sorted((name, tuple(sorted(obj.properties.items()))) 
                                    for name, obj in self.objects.items()))
        return hash((
            self.width, self.height, self.robot, self.gripper,
            obj_positions, obj_properties, tuple(sorted(self.open_containers))
        ))
    
    def __eq__(self, other) -> bool:
        """Enhanced equality comparison for states."""
        if not isinstance(other, State):
            return False
        return (self.width == other.width and
                self.height == other.height and
                self.robot == other.robot and
                self.gripper == other.gripper and
                self.objects == other.objects and
                self.open_containers == other.open_containers)
    
    def __str__(self) -> str:
        """Enhanced string representation of state."""
        lines = []
        lines.append(f"Grid: {self.width}x{self.height}")
        lines.append(f"Robot: {self.robot}, Gripper: {self.gripper}")
        lines.append(f"Safety Score: {self.get_safety_score():.2f}")
        lines.append("Objects:")
        for name, obj in self.objects.items():
            lines.append(f"  {name}: {obj.position} {obj.properties}")
        lines.append(f"Open containers: {self.open_containers}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "robot": self.robot,
            "gripper": self.gripper,
            "objects": {name: {
                "position": obj.position,
                "properties": obj.properties
            } for name, obj in self.objects.items()},
            "open_containers": list(self.open_containers),
            "safety_score": self.get_safety_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create state from dictionary."""
        objects = {}
        for name, obj_data in data["objects"].items():
            objects[name] = Object(obj_data["position"], obj_data["properties"])
        
        return cls(
            width=data["width"],
            height=data["height"],
            robot=tuple(data["robot"]),
            gripper=data["gripper"],
            objects=objects,
            open_containers=set(data["open_containers"])
        )