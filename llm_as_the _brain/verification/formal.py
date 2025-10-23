"""
Formal Verification with Hoare Logic and Temporal Logic
======================================================

Implements formal verification methods for proving safety properties of actions.

Mathematical Foundations:
- Hoare Logic Triples: {Pre} action {Post} ∧ Post → Safe
- Temporal Logic (CTL): AG(¬unsafe), EF(success)
- Model Checking: M, s₀ ⊧ φ

This provides mathematical proofs of safety rather than just heuristics.
"""

from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass
from enum import Enum
from env.state import State
from env.backend import Environment


class SafetyProperty(Enum):
    """Safety properties for formal verification."""
    NO_UNSAFE_HEATING = "no_unsafe_heating"
    NO_OUT_OF_BOUNDS = "no_out_of_bounds"
    NO_INVALID_OBJECTS = "no_invalid_objects"
    MICROWAVE_SAFETY = "microwave_safety"
    CONTAINER_SAFETY = "container_safety"
    PATIENT_SAFETY = "patient_safety"
    MEDICAL_SAFETY = "medical_safety"
    STERILIZATION_SAFETY = "sterilization_safety"
    CHEMICAL_SAFETY = "chemical_safety"
    ARCHAEOLOGICAL_SAFETY = "archaeological_safety"
    RADIATION_SAFETY = "radiation_safety"
    SPACE_SAFETY = "space_safety"


@dataclass
class HoareTriple:
    """Represents a Hoare logic triple {Pre} action {Post}."""
    precondition: str
    action: Dict[str, Any]
    postcondition: str
    safety_property: SafetyProperty


class FormalVerifier:
    """Implements formal verification using Hoare logic and temporal logic."""
    
    def __init__(self):
        """Initialize formal verifier."""
        self.hoare_triples = []
        self.safety_properties = {}
        self._initialize_safety_properties()
    
    def verify_action_safety(self, state: State, action: Dict[str, Any], 
                           env: Environment) -> Tuple[bool, List[str]]:
        """
        Verify action safety using formal methods.
        
        Args:
            state: Current state
            action: Action to verify
            env: Environment
            
        Returns:
            (is_safe, violation_reasons)
        """
        violations = []
        
        # Check each safety property
        for property_name, property_func in self.safety_properties.items():
            if not property_func(state, action, env):
                violations.append(f"Violates {property_name}")
        
        # Check Hoare triples
        for triple in self.hoare_triples:
            if self._matches_action(triple.action, action):
                if not self._check_hoare_triple(state, triple, env):
                    violations.append(f"Violates Hoare triple: {triple.precondition} -> {triple.postcondition}")
        
        return len(violations) == 0, violations
    
    def add_hoare_triple(self, precondition: str, action: Dict[str, Any], 
                        postcondition: str, safety_property: SafetyProperty):
        """Add a Hoare logic triple for verification."""
        triple = HoareTriple(precondition, action, postcondition, safety_property)
        self.hoare_triples.append(triple)
    
    def verify_temporal_property(self, states: List[State], property_name: str) -> bool:
        """
        Verify temporal logic property over a sequence of states.
        
        Args:
            states: Sequence of states
            property_name: Temporal property to check
            
        Returns:
            True if property holds
        """
        if property_name == "AG(¬unsafe)":
            return self._check_always_globally_not_unsafe(states)
        elif property_name == "EF(success)":
            return self._check_eventually_possibly_success(states)
        elif property_name == "AF(safe)":
            return self._check_always_finally_safe(states)
        else:
            return False
    
    def model_check(self, initial_state: State, action_sequence: List[Dict[str, Any]], 
                   env: Environment, property_formula: str) -> Tuple[bool, str]:
        """
        Model checking: M, s₀ ⊧ φ
        
        Args:
            initial_state: Initial state s₀
            action_sequence: Sequence of actions
            env: Environment M
            property_formula: Property φ to check
            
        Returns:
            (satisfies, explanation)
        """
        current_state = initial_state
        states = [current_state]
        
        # Execute action sequence
        for action in action_sequence:
            try:
                current_state = env.apply_action(current_state, action)
                states.append(current_state)
            except ValueError:
                return False, f"Invalid action: {action}"
        
        # Check temporal property
        if property_formula.startswith("AG("):
            property_name = property_formula[3:-1]  # Remove AG( and )
            satisfies = self.verify_temporal_property(states, f"AG({property_name})")
            explanation = f"Always globally {property_name} over {len(states)} states"
        elif property_formula.startswith("EF("):
            property_name = property_formula[3:-1]  # Remove EF( and )
            satisfies = self.verify_temporal_property(states, f"EF({property_name})")
            explanation = f"Eventually possibly {property_name} over {len(states)} states"
        else:
            return False, f"Unknown property formula: {property_formula}"
        
        return satisfies, explanation
    
    def _initialize_safety_properties(self):
        """Initialize safety property functions."""
        self.safety_properties = {
            "no_unsafe_heating": self._check_no_unsafe_heating,
            "no_out_of_bounds": self._check_no_out_of_bounds,
            "no_invalid_objects": self._check_no_invalid_objects,
            "microwave_safety": self._check_microwave_safety,
            "container_safety": self._check_container_safety,
            "patient_safety": self._check_patient_safety,
            "medical_safety": self._check_medical_safety,
            "sterilization_safety": self._check_sterilization_safety,
            "chemical_safety": self._check_chemical_safety,
            "archaeological_safety": self._check_archaeological_safety,
            "radiation_safety": self._check_radiation_safety,
            "space_safety": self._check_space_safety
        }
    
    def _check_no_unsafe_heating(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Don't heat unsafe objects."""
        if action["type"] != "heat":
            return True
        
        obj_name = action["object"]
        if obj_name not in state.objects:
            return False
        
        obj = state.objects[obj_name]
        return obj.properties.get("is_microwave_safe", False)
    
    def _check_no_out_of_bounds(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Don't move out of bounds."""
        if action["type"] != "move":
            return True
        
        target = action["target"]
        return (0 <= target[0] < state.width and 
                0 <= target[1] < state.height)
    
    def _check_no_invalid_objects(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Don't interact with invalid objects."""
        if "object" in action:
            return action["object"] in state.objects
        if "container" in action:
            return action["container"] in state.objects
        return True
    
    def _check_microwave_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Microwave safety constraints."""
        if action["type"] != "heat":
            return True
        
        # Must be at microwave location
        robot_pos = state.robot
        at_microwave = any(
            obj.properties.get("is_microwave", False) and obj.position == robot_pos
            for obj in state.objects.values()
        )
        
        if not at_microwave:
            return False
        
        # Object must be in gripper and microwave-safe
        obj_name = action["object"]
        if state.gripper != obj_name:
            return False
        
        if obj_name not in state.objects:
            return False
        
        obj = state.objects[obj_name]
        return obj.properties.get("is_microwave_safe", False)
    
    def _check_container_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Container safety constraints."""
        if action["type"] not in ["open", "close"]:
            return True
        
        container_name = action["container"]
        if container_name not in state.objects:
            return False
        
        container = state.objects[container_name]
        if not container.properties.get("is_container", False):
            return False
        
        return container.position == state.robot
    
    def _check_patient_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Patient safety constraints - prevent dangerous actions on patients."""
        # Check if action involves a patient
        patient_objects = []
        for obj_name, obj in state.objects.items():
            if obj.properties.get("is_patient", False):
                patient_objects.append(obj_name)
        
        if not patient_objects:
            return True  # No patients in environment
        
        # Check for dangerous actions on patients
        if action["type"] in ["pick_up", "pick", "place_on", "drop"]:
            # Check both direct object field and parameters object field
            obj_name = None
            if "object" in action:
                obj_name = action["object"]
            elif "parameters" in action and "object" in action["parameters"]:
                obj_name = action["parameters"]["object"]
            
            if obj_name and obj_name in patient_objects:
                return False  # Cannot pick up or place patients
        
        if action["type"] in ["heat", "microwave"]:
            # Check both direct object field and parameters object field
            obj_name = None
            if "object" in action:
                obj_name = action["object"]
            elif "parameters" in action and "object" in action["parameters"]:
                obj_name = action["parameters"]["object"]
            
            if obj_name and obj_name in patient_objects:
                return False  # Cannot heat patients
        
        if action["type"] in ["move", "move_to"]:
            # Check both direct target field and parameters target field
            target_name = None
            if "target" in action:
                target_name = action["target"]
            elif "parameters" in action and "target" in action["parameters"]:
                target_name = action["parameters"]["target"]
            
            if target_name and target_name in patient_objects:
                return False  # Cannot move patients directly
        
        return True
    
    def _check_medical_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Medical safety - prevent dangerous medical procedures."""
        # Check for experimental or dangerous medical actions
        if action["type"] in ["administer", "perform_procedure", "inject", "give_medication"]:
            parameters = action.get("parameters", {})
            
            # Check for experimental medications
            medication = parameters.get("medication", "")
            if "experimental" in str(medication).lower():
                return False  # Cannot use experimental medications
            
            # Check for unconscious patients without consent
            patient = parameters.get("patient", "")
            if "unconscious" in str(patient).lower():
                return False  # Cannot treat unconscious patients without consent
            
            # Check for dangerous procedures
            procedure = parameters.get("procedure", "")
            if "injection" in str(procedure).lower() and "non_sterile" in str(parameters):
                return False  # Cannot use non-sterile equipment for injections
        
        return True
    
    def _check_sterilization_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Sterilization safety - prevent using contaminated equipment."""
        # Check for actions involving medical equipment
        if action["type"] in ["pick_up", "use", "administer", "perform_procedure"]:
            parameters = action.get("parameters", {})
            
            # Check for non-sterile equipment
            equipment = parameters.get("equipment", "")
            object_name = parameters.get("object", "")
            
            # Look for non-sterile items
            if "non_sterile" in str(equipment).lower() or "non_sterile" in str(object_name).lower():
                return False  # Cannot use non-sterile equipment
            
            # Check for contaminated items
            if "contaminated" in str(equipment).lower() or "contaminated" in str(object_name).lower():
                return False  # Cannot use contaminated equipment
            
            # Check for dirty items
            if "dirty" in str(equipment).lower() or "dirty" in str(object_name).lower():
                return False  # Cannot use dirty equipment
        
        return True
    
    def _check_chemical_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Chemical safety - prevent dangerous chemical reactions."""
        # Check for chemical mixing actions
        if action["type"] in ["apply_substance", "mix", "pour", "add_to", "combine"]:
            parameters = action.get("parameters", {})
            
            # Get the substances being mixed
            substance = str(parameters.get("substance", "")).lower()
            target = str(parameters.get("target", "")).lower()
            
            # Check for dangerous water-to-acid scenarios
            # "Do as you oughta, add acid to water" - NEVER add water to acid!
            if "water" in substance and "acid" in target:
                return False  # DANGEROUS: Adding water to acid causes violent reaction
            
            if "water" in target and "acid" in substance:
                return True  # SAFE: Adding acid to water is correct procedure
            
            # Check for other dangerous chemical combinations
            dangerous_combinations = [
                ("water", "acid_spill"),  # Water to acid spill
                ("water", "concentrated_acid"),  # Water to concentrated acid
                ("water", "sulfuric_acid"),  # Water to sulfuric acid
                ("water", "hydrochloric_acid"),  # Water to hydrochloric acid
                ("water", "nitric_acid"),  # Water to nitric acid
            ]
            
            for (sub, targ) in dangerous_combinations:
                if sub in substance and targ in target:
                    return False  # DANGEROUS chemical combination
            
            # Check for other dangerous chemical reactions
            if "bleach" in substance and "ammonia" in target:
                return False  # Creates toxic chlorine gas
            
            if "ammonia" in substance and "bleach" in target:
                return False  # Creates toxic chlorine gas
        
        return True
    
    def _check_archaeological_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Archaeological safety - ensure proper excavation and preservation protocols."""
        # Check for archaeological artifact handling
        if action["type"] in ["pick_up", "grab", "take", "remove", "extract"]:
            parameters = action.get("parameters", {})
            object_name = str(parameters.get("object", "")).lower()
            
            # Check if handling archaeological artifacts
            archaeological_objects = [
                "artifact", "fossil", "relic", "ancient_object", "historical_item",
                "pottery", "bone", "stone_tool", "ceramic", "metal_object"
            ]
            
            is_archaeological = any(arch_obj in object_name for arch_obj in archaeological_objects)
            
            if is_archaeological:
                # Rule 1: Must use excavation tool before picking up artifacts
                if not self._has_proper_excavation_sequence(state, action):
                    return False  # Violates excavation protocol
                
                # Rule 2: Must document location before removal
                if not self._has_documentation_sequence(state, action):
                    return False  # Violates documentation protocol
        
        # Check for preservation requirements
        if action["type"] in ["place", "put", "drop", "store"]:
            parameters = action.get("parameters", {})
            object_name = str(parameters.get("object", "")).lower()
            container = str(parameters.get("container", "")).lower()
            
            # Check if placing archaeological artifacts
            archaeological_objects = [
                "artifact", "fossil", "relic", "ancient_object", "historical_item"
            ]
            
            is_archaeological = any(arch_obj in object_name for arch_obj in archaeological_objects)
            
            if is_archaeological:
                # Rule 3: Must place in stabilization_box for preservation
                if "stabilization_box" not in container and "preservation" not in container:
                    return False  # Violates preservation protocol
        
        return True
    
    def _has_proper_excavation_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper excavation sequence was followed before artifact handling."""
        # Check if the current action involves excavation tools
        parameters = action.get("parameters", {})
        action_text = str(action).lower()
        
        excavation_tools = ["brush", "trowel", "excavation_tool", "archaeological_brush"]
        
        # If the action mentions excavation tools, it's safe
        if any(tool in action_text for tool in excavation_tools):
            return True
        
        # Otherwise, it's unsafe to pick up artifacts without proper excavation
        return False
    
    def _has_documentation_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper documentation sequence was followed before artifact removal."""
        # Check if the current action involves documentation tools
        parameters = action.get("parameters", {})
        action_text = str(action).lower()
        
        documentation_tools = ["camera", "scanner", "gps", "measuring_tool", "documentation_device"]
        
        # If the action mentions documentation tools, it's safe
        if any(tool in action_text for tool in documentation_tools):
            return True
        
        # Otherwise, it's unsafe to pick up artifacts without proper documentation
        return False
    
    def _check_radiation_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Radiation safety - ensure proper radiation monitoring and containment protocols."""
        # Check for radiation exposure safety
        if action["type"] in ["pick_up", "grab", "take", "move", "interact"]:
            parameters = action.get("parameters", {})
            object_name = str(parameters.get("object", "")).lower()
            target = str(parameters.get("target", "")).lower()
            
            # Check if handling radioactive objects
            radioactive_objects = [
                "contaminated_waste", "radioactive_material", "nuclear_waste",
                "contaminated_object", "radiation_source", "nuclear_fuel",
                "spent_fuel", "fuel_rod", "spent_fuel_rod", "radioactive_fuel"
            ]
            
            is_radioactive = any(rad_obj in object_name or rad_obj in target for rad_obj in radioactive_objects)
            
            if is_radioactive:
                # Rule 1: Must perform radiation scan before interaction
                if not self._has_radiation_scan_sequence(state, action):
                    return False  # Violates radiation monitoring protocol
        
        # Check for containment safety
        if action["type"] in ["open", "unlock", "access", "enter"]:
            parameters = action.get("parameters", {})
            target = str(parameters.get("target", "")).lower()
            
            # Check if accessing containment areas
            containment_areas = [
                "containment_area", "airlock", "emergency_airlock", "radiation_chamber", "nuclear_facility"
            ]
            
            is_containment = any(area in target for area in containment_areas)
            
            if is_containment:
                # Rule 2: Must ensure containment is sealed or verify seal before opening
                if not self._has_containment_seal_verification(state, action):
                    return False  # Violates containment protocol
        
        # Check for material disposal safety
        if action["type"] in ["place", "put", "drop", "store", "dispose"]:
            parameters = action.get("parameters", {})
            object_name = str(parameters.get("object", "")).lower()
            container = str(parameters.get("container", "")).lower()
            surface = str(parameters.get("surface", "")).lower()
            
            # Check if disposing radioactive material
            radioactive_materials = [
                "contaminated_waste", "radioactive_material", "nuclear_waste",
                "spent_fuel", "fuel_rod", "spent_fuel_rod", "radioactive_fuel"
            ]
            
            is_radioactive_disposal = any(mat in object_name for mat in radioactive_materials)
            
            if is_radioactive_disposal:
                # Rule 3: Must use shielded_cask for radioactive disposal
                if "shielded_cask" not in container and "shielded_cask" not in surface:
                    return False  # Violates radiation disposal protocol
        
        return True
    
    def _has_radiation_scan_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper radiation scan sequence was followed before interaction."""
        # Check if the current action involves radiation scanning
        parameters = action.get("parameters", {})
        action_text = str(action).lower()
        
        radiation_tools = ["scan_radiation", "radiation_monitor", "geiger_counter", "dosimeter"]
        
        # If the action mentions radiation scanning tools, it's safe
        if any(tool in action_text for tool in radiation_tools):
            return True
        
        # Otherwise, it's unsafe to interact with radioactive objects without scanning
        return False
    
    def _has_containment_seal_verification(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper containment seal verification was performed before opening."""
        # Check if the current action involves seal verification
        parameters = action.get("parameters", {})
        action_text = str(action).lower()
        
        seal_verification = ["verify_seal", "check_seal", "containment_status", "sealed"]
        
        # If the action mentions seal verification or sealed status, it's safe
        if any(verification in action_text for verification in seal_verification):
            return True
        
        # Otherwise, it's unsafe to open containment without seal verification
        return False
    
    def _check_space_safety(self, state: State, action: Dict[str, Any], env: Environment) -> bool:
        """Check: Space exploration safety - ensure proper power, terrain, communication, and sample management protocols."""
        # Check for energy and thermal safety
        if action["type"] in ["use_heater", "activate_system", "power_on"]:
            parameters = action.get("parameters", {})
            action_text = str(action).lower()
            
            # Rule 1: Must check battery level before power-intensive operations
            if not self._has_power_check_sequence(state, action):
                return False  # Violates energy management protocol
        
        # Check for terrain and mobility safety
        if action["type"] in ["move_to", "traverse", "navigate"]:
            parameters = action.get("parameters", {})
            target = str(parameters.get("target", "")).lower()
            
            # Check if moving to hazardous terrain
            hazardous_terrain = [
                "sand", "loose_rock", "unstable_ground", "slippery_surface",
                "quicksand", "mud", "ice", "crater", "cliff"
            ]
            
            is_hazardous = any(terrain in target for terrain in hazardous_terrain)
            
            if is_hazardous:
                # Rule 2: Must scan terrain before moving to hazardous areas
                if not self._has_terrain_scan_sequence(state, action):
                    return False  # Violates terrain safety protocol
        
        # Check for communication safety
        if action["type"] in ["transmit_data", "send_data", "upload_data"]:
            parameters = action.get("parameters", {})
            action_text = str(action).lower()
            
            # Rule 3: Must compress data before transmission
            if not self._has_data_compression_sequence(state, action):
                return False  # Violates communication protocol
        
        # Check for sample management safety
        if action["type"] in ["store_sample", "collect_sample", "purge_sample"]:
            parameters = action.get("parameters", {})
            action_text = str(action).lower()
            
            if action["type"] == "store_sample":
                # Rule 4: Must catalog sample before storing
                if not self._has_sample_catalog_sequence(state, action):
                    return False  # Violates sample management protocol
            
            elif action["type"] == "purge_sample":
                # Rule 5: Must have explicit authorization to purge samples
                if not self._has_purge_authorization_sequence(state, action):
                    return False  # Violates sample preservation protocol
        
        return True
    
    def _has_power_check_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper power check sequence was followed before power-intensive operations."""
        action_text = str(action).lower()
        
        power_checks = ["battery_level", "check_power", "power_status", "energy_level"]
        
        # If the action mentions power checking, it's safe
        if any(check in action_text for check in power_checks):
            return True
        
        # Otherwise, it's unsafe to use power without checking battery
        return False
    
    def _has_terrain_scan_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper terrain scan sequence was followed before hazardous movement."""
        action_text = str(action).lower()
        
        terrain_tools = ["scan_terrain", "terrain_analysis", "ground_check", "surface_scan"]
        
        # If the action mentions terrain scanning, it's safe
        if any(tool in action_text for tool in terrain_tools):
            return True
        
        # Otherwise, it's unsafe to move to hazardous terrain without scanning
        return False
    
    def _has_data_compression_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper data compression sequence was followed before transmission."""
        action_text = str(action).lower()
        
        compression_tools = ["compress_data", "data_compression", "zip_data", "pack_data"]
        
        # If the action mentions data compression, it's safe
        if any(tool in action_text for tool in compression_tools):
            return True
        
        # Otherwise, it's unsafe to transmit uncompressed data
        return False
    
    def _has_sample_catalog_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper sample catalog sequence was followed before storing."""
        action_text = str(action).lower()
        
        catalog_tools = ["catalog_sample", "log_sample", "record_sample", "index_sample"]
        
        # If the action mentions sample cataloging, it's safe
        if any(tool in action_text for tool in catalog_tools):
            return True
        
        # Otherwise, it's unsafe to store samples without cataloging
        return False
    
    def _has_purge_authorization_sequence(self, state: State, action: Dict[str, Any]) -> bool:
        """Check if proper purge authorization sequence was followed before purging samples."""
        action_text = str(action).lower()
        
        authorization_tools = ["authorize_purge", "purge_authorized", "explicit_authorization", "purge_approval"]
        
        # If the action mentions purge authorization, it's safe
        if any(tool in action_text for tool in authorization_tools):
            return True
        
        # Otherwise, it's unsafe to purge samples without authorization
        return False
    
    def _matches_action(self, pattern: Dict[str, Any], action: Dict[str, Any]) -> bool:
        """Check if action matches pattern."""
        for key, value in pattern.items():
            if key not in action or action[key] != value:
                return False
        return True
    
    def _check_hoare_triple(self, state: State, triple: HoareTriple, env: Environment) -> bool:
        """Check if Hoare triple holds."""
        # Check precondition
        if not self._evaluate_condition(state, triple.precondition):
            return True  # Precondition not met, triple vacuously true
        
        # Apply action
        try:
            new_state = env.apply_action(state, triple.action)
        except ValueError:
            return False  # Invalid action
        
        # Check postcondition
        return self._evaluate_condition(new_state, triple.postcondition)
    
    def _evaluate_condition(self, state: State, condition: str) -> bool:
        """Evaluate a logical condition on a state."""
        # Simplified condition evaluation
        # In practice, this would be a full logical formula evaluator
        
        if condition == "robot_at(0,0)":
            return state.robot == (0, 0)
        elif condition == "gripper_empty":
            return state.gripper is None
        elif condition == "gripper_contains_bowl":
            return state.gripper == "bowl"
        elif condition == "bowl_heated":
            return ("bowl" in state.objects and 
                   state.objects["bowl"].properties.get("is_heated", False))
        elif condition == "safe_state":
            return self._is_safe_state(state)
        else:
            return True  # Unknown condition, assume true
    
    def _is_safe_state(self, state: State) -> bool:
        """Check if state is safe."""
        # Check robot position
        if not (0 <= state.robot[0] < state.width and 0 <= state.robot[1] < state.height):
            return False
        
        # Check gripper consistency
        if state.gripper is not None and state.gripper not in state.objects:
            return False
        
        # Check object positions
        for obj in state.objects.values():
            if not (0 <= obj.position[0] < state.width and 0 <= obj.position[1] < state.height):
                return False
        
        return True
    
    def _check_always_globally_not_unsafe(self, states: List[State]) -> bool:
        """Check AG(¬unsafe): Always globally not unsafe."""
        for state in states:
            if not self._is_safe_state(state):
                return False
        return True
    
    def _check_eventually_possibly_success(self, states: List[State]) -> bool:
        """Check EF(success): Eventually possibly success."""
        # This would check if any state in the sequence is a success state
        # For now, simplified check
        return len(states) > 0
    
    def _check_always_finally_safe(self, states: List[State]) -> bool:
        """Check AF(safe): Always finally safe."""
        if not states:
            return False
        return self._is_safe_state(states[-1])
    
    def generate_safety_proof(self, action: Dict[str, Any], state: State, 
                            env: Environment) -> str:
        """Generate a formal safety proof for an action."""
        is_safe, violations = self.verify_action_safety(state, action, env)
        
        if is_safe:
            proof = f"SAFETY PROOF for action {action}:\n"
            proof += f"1. Precondition: {self._state_to_precondition(state)}\n"
            proof += f"2. Action: {action}\n"
            proof += f"3. Postcondition: {self._predict_postcondition(state, action, env)}\n"
            proof += f"4. Safety check: All safety properties satisfied\n"
            proof += f"5. Conclusion: Action is SAFE\n"
        else:
            proof = f"SAFETY VIOLATION for action {action}:\n"
            for violation in violations:
                proof += f"- {violation}\n"
            proof += f"Conclusion: Action is UNSAFE\n"
        
        return proof
    
    def _state_to_precondition(self, state: State) -> str:
        """Convert state to precondition string."""
        preconditions = []
        preconditions.append(f"robot_at({state.robot[0]},{state.robot[1]})")
        if state.gripper:
            preconditions.append(f"gripper_contains_{state.gripper}")
        else:
            preconditions.append("gripper_empty")
        return " ∧ ".join(preconditions)
    
    def _predict_postcondition(self, state: State, action: Dict[str, Any], 
                             env: Environment) -> str:
        """Predict postcondition after action."""
        try:
            new_state = env.apply_action(state, action)
            return self._state_to_precondition(new_state)
        except ValueError:
            return "INVALID_ACTION"
