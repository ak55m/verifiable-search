"""
State representation for gridworld environment.
Defines the state structure with grid, objects, and robot.
"""

from typing import Dict, Set, Optional, Tuple, Any, List
from dataclasses import dataclass


@dataclass
class Object:
    """Represents an object in the environment."""
    position: Tuple[int, int]
    properties: Dict[str, Any]
    
    def __post_init__(self):
        """Validate object properties."""
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError("Position must be a tuple of (x, y)")
        if not all(isinstance(coord, int) for coord in self.position):
            raise ValueError("Position coordinates must be integers")


@dataclass
class State:
    """Complete state representation of the environment."""
    width: int
    height: int
    robot: Tuple[int, int]
    gripper: Optional[str]  # Name of object in gripper, None if empty
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
    
    def __hash__(self) -> int:
        """Hash function for state comparison."""
        # Create hashable representation
        obj_positions = tuple(sorted((name, obj.position) for name, obj in self.objects.items()))
        return hash((
            self.width, self.height, self.robot, self.gripper,
            obj_positions, tuple(sorted(self.open_containers))
        ))
    
    def __eq__(self, other) -> bool:
        """Equality comparison for states."""
        if not isinstance(other, State):
            return False
        return (self.width == other.width and
                self.height == other.height and
                self.robot == other.robot and
                self.gripper == other.gripper and
                self.objects == other.objects and
                self.open_containers == other.open_containers)
    
    def __str__(self) -> str:
        """String representation of state."""
        lines = []
        lines.append(f"Grid: {self.width}x{self.height}")
        lines.append(f"Robot: {self.robot}, Gripper: {self.gripper}")
        lines.append("Objects:")
        for name, obj in self.objects.items():
            lines.append(f"  {name}: {obj.position} {obj.properties}")
        lines.append(f"Open containers: {self.open_containers}")
        return "\n".join(lines)
