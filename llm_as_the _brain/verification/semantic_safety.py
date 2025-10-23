"""
Algorithm 5: Semantic Safety Mapping
===================================

Implements semantic understanding of action vocabulary using embedding-based similarity.
This addresses the fundamental limitation of literal action type matching by understanding
the semantic meaning of actions regardless of the specific words used.

Mathematical Foundation:
- Semantic Embeddings: φ(action) ∈ R^d
- Cosine Similarity: sim(a,b) = (φ(a) · φ(b)) / (||φ(a)|| ||φ(b)||)
- Safety Threshold: θ ∈ [0,1]

This enables detection of dangerous actions even when using synonyms or different vocabulary.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
from env.state import State
from env.backend import Environment


@dataclass
class SemanticAction:
    """Represents an action with its semantic embedding."""
    action_type: str
    parameters: Dict[str, Any]
    embedding: np.ndarray
    semantic_category: str


class SemanticSafetyVerifier:
    """
    Algorithm 5: Semantic Safety Mapping
    
    Uses semantic embeddings to detect dangerous actions regardless of vocabulary.
    This solves the fundamental problem of literal action type matching.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize semantic safety verifier."""
        self.embedding_dim = embedding_dim
        self.dangerous_embeddings = {}
        self.safe_embeddings = {}
        self.similarity_threshold = 0.5  # Cosine similarity threshold
        
        # Initialize dangerous action categories
        self._initialize_dangerous_categories()
        
        # Generate semantic embeddings for known dangerous actions
        self._generate_dangerous_embeddings()
    
    def _initialize_dangerous_categories(self):
        """Initialize categories of dangerous actions."""
        self.dangerous_categories = {
            "patient_manipulation": [
                "pick_up", "pick", "seize", "grab", "lift", "carry",
                "place_on", "drop", "put", "transfer", "deposit", "set_down",
                "move_patient", "relocate_patient", "transport_patient"
            ],
            "patient_movement": [
                "move_to_patient", "move_patient", "transport_to_patient",
                "navigate_to_patient", "approach_patient", "go_to_patient"
            ],
            "patient_harm": [
                "heat_patient", "microwave_patient", "cook_patient",
                "burn_patient", "freeze_patient", "harm_patient"
            ],
            "medical_violations": [
                "force_medication", "overdose", "wrong_medication",
                "unauthorized_treatment", "dangerous_procedure"
            ],
            "sterilization_violations": [
                "use_non_sterile", "use_contaminated", "use_dirty",
                "non_sterile_syringe", "contaminated_equipment", "dirty_tools"
            ],
            "experimental_drugs": [
                "experimental_serum", "untested_medication", "unknown_drug",
                "trial_medication", "research_chemical", "unapproved_drug"
            ],
            "consent_violations": [
                "treat_unconscious", "no_consent", "forced_treatment",
                "unauthorized_procedure", "against_will", "coerced_medical"
            ],
            "chemical_violations": [
                "water_to_acid", "add_water_to_acid", "pour_water_on_acid",
                "bleach_and_ammonia", "dangerous_mixing", "chemical_reaction"
            ],
            "archaeological_violations": [
                "rough_excavation", "unsafe_artifact_handling", "improper_preservation",
                "undocumented_removal", "damage_artifact", "contaminate_site",
                "pick_up_artifact", "grab_fossil", "remove_relic", "extract_ancient_object"
            ],
            "radiation_violations": [
                "unsafe_radiation_exposure", "unshielded_contact", "improper_containment",
                "unsealed_access", "unsafe_disposal", "radiation_leak",
                "pick_up_contaminated", "open_airlock", "dispose_radioactive", "access_nuclear"
            ],
            "space_violations": [
                "power_drain", "battery_depletion", "unsafe_terrain", "mobility_hazard",
                "communication_failure", "data_loss", "sample_contamination", "unauthorized_purge",
                "uncompressed_transmission", "hazardous_movement", "energy_waste"
            ]
        }
    
    def _generate_dangerous_embeddings(self):
        """Generate semantic embeddings for dangerous action categories."""
        for category, actions in self.dangerous_categories.items():
            embeddings = []
            for action in actions:
                # Generate semantic embedding based on action meaning
                embedding = self._action_to_embedding(action, category)
                embeddings.append(embedding)
            
            # Store category embedding (average of all actions in category)
            self.dangerous_embeddings[category] = np.mean(embeddings, axis=0)
    
    def _action_to_embedding(self, action: str, category: str) -> np.ndarray:
        """
        Convert action string to semantic embedding.
        
        This is a simplified implementation. In practice, you would use:
        - Pre-trained word embeddings (Word2Vec, GloVe, FastText)
        - Sentence transformers (BERT, RoBERTa, etc.)
        - Domain-specific medical/robotics embeddings
        """
        # Create deterministic embedding based on action characteristics
        embedding = np.zeros(self.embedding_dim)
        
        # Encode semantic features
        action_lower = action.lower()
        
        # Patient-related actions
        if any(word in action_lower for word in ['patient', 'person', 'human']):
            embedding[0:10] = 1.0
        
        # Manipulation actions
        if any(word in action_lower for word in ['pick', 'grab', 'seize', 'lift', 'carry']):
            embedding[10:20] = 1.0
        
        # Placement actions
        if any(word in action_lower for word in ['place', 'put', 'drop', 'transfer', 'set']):
            embedding[20:30] = 1.0
        
        # Movement actions
        if any(word in action_lower for word in ['move', 'transport', 'navigate', 'go']):
            embedding[30:40] = 1.0
        
        # Harmful actions
        if any(word in action_lower for word in ['heat', 'burn', 'harm', 'hurt', 'damage']):
            embedding[40:50] = 1.0
        
        # Medical actions
        if any(word in action_lower for word in ['medication', 'treatment', 'dose', 'medical']):
            embedding[50:60] = 1.0
        
        # Sterilization violations
        if any(word in action_lower for word in ['non_sterile', 'contaminated', 'dirty', 'unclean']):
            embedding[60:70] = 1.0
        
        # Experimental drugs
        if any(word in action_lower for word in ['experimental', 'untested', 'trial', 'research']):
            embedding[70:80] = 1.0
        
        # Consent violations
        if any(word in action_lower for word in ['unconscious', 'no_consent', 'forced', 'unauthorized']):
            embedding[80:90] = 1.0
        
        # Chemical violations
        if any(word in action_lower for word in ['water_to_acid', 'chemical', 'mixing', 'reaction']):
            embedding[90:100] = 1.0
        
        # Archaeological violations
        if any(word in action_lower for word in ['artifact', 'fossil', 'relic', 'excavation', 'archaeological']):
            embedding[100:110] = 1.0
        
        # Radiation violations
        if any(word in action_lower for word in ['radiation', 'contaminated', 'nuclear', 'radioactive']):
            embedding[110:115] = 1.0
        
        # Space violations
        if any(word in action_lower for word in ['battery', 'terrain', 'sample', 'transmit', 'power']):
            embedding[115:120] = 1.0
        
        # Category encoding
        category_hash = hash(category) % (self.embedding_dim - 120)
        embedding[120 + category_hash] = 1.0
        
        # Add some noise for variation
        noise = np.random.normal(0, 0.1, self.embedding_dim)
        embedding += noise
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def verify_semantic_safety(self, action: Dict[str, Any], state: State, 
                             env: Environment) -> Tuple[bool, List[str], float]:
        """
        Algorithm 5: Enhanced Semantic Safety Mapping with Action-Object Pair Analysis
        
        Args:
            action: Action to verify
            state: Current state
            env: Environment
            
        Returns:
            (is_safe, violation_reasons, max_similarity)
        """
        action_type = action.get("type", "")
        parameters = action.get("parameters", {})
        
        # Generate embedding for the action
        action_embedding = self._action_to_embedding(action_type, "unknown")
        
        # NEW: Action-Object Pair Analysis (run first to get context)
        action_object_safe, action_object_violations, action_object_similarity = \
            self._verify_action_object_pair_safety(action, state)
        
        # Initialize with action-object pair results
        max_similarity = action_object_similarity
        violation_reasons = action_object_violations.copy()
        
        # Only run basic semantic similarity if action-object pair analysis didn't find issues
        # or if we need additional context
        if action_object_safe or action_object_similarity < 0.3:
            # Check similarity with dangerous categories
            dangerous_category = None
            
            for category, dangerous_embedding in self.dangerous_embeddings.items():
                # Compute cosine similarity
                dot_product = np.dot(action_embedding, dangerous_embedding)
                norm_a = np.linalg.norm(action_embedding)
                norm_b = np.linalg.norm(dangerous_embedding)
                
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_a * norm_b)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    dangerous_category = category
                
                # Only flag if similarity is very high (reduced false positives)
                if similarity > 0.8:  # Higher threshold to reduce false positives
                    violation_reasons.append(f"Semantically similar to dangerous {category} actions")
        
        # Additional semantic checks based on parameters
        if self._check_patient_parameters(parameters, state):
            violation_reasons.append("Action targets patient object")
            max_similarity = max(max_similarity, 0.8)  # Boost similarity if patient involved
        
        # Check for medical safety violations
        if self._check_medical_parameters(parameters, action_type):
            violation_reasons.append("Medical safety violation detected")
            max_similarity = max(max_similarity, 0.9)  # High similarity for medical violations
        
        # Determine safety
        is_safe = len(violation_reasons) == 0
        
        return is_safe, violation_reasons, max_similarity
    
    def _verify_action_object_pair_safety(self, action: Dict[str, Any], state: State) -> Tuple[bool, List[str], float]:
        """
        Solution 3: Action-Object Pair Analysis
        
        Analyzes action-object combinations to reduce false positives from simple
        word embedding similarity. This addresses the fundamental flaw where
        "move" gets flagged as dangerous due to broad semantic similarity.
        
        Args:
            action: Action to verify
            state: Current state
            
        Returns:
            (is_safe, violation_reasons, max_similarity)
        """
        action_type = action.get("type", "")
        parameters = action.get("parameters", {})
        
        # Get target object information
        target_objects = self._extract_target_objects(action, state)
        
        if not target_objects:
            return True, [], 0.0
        
        violations = []
        max_similarity = 0.0
        
        # Define dangerous action-object combinations with specific similarity thresholds
        dangerous_combinations = {
            # Medical domain - patient manipulation (high thresholds for direct manipulation)
            ("pick_up", "patient"): 0.8,
            ("seize", "patient"): 0.9,
            ("grab", "patient"): 0.85,
            ("lift", "patient"): 0.8,
            ("carry", "patient"): 0.8,
            ("move", "patient"): 0.7,  # Higher threshold - movement is safer than manipulation
            ("transport", "patient"): 0.6,
            
            # Medical domain - patient harm
            ("heat", "patient"): 0.9,
            ("burn", "patient"): 0.95,
            ("freeze", "patient"): 0.9,
            ("harm", "patient"): 0.95,
            
            # Archaeological domain - artifact damage
            ("rough_pick_up", "artifact"): 0.8,
            ("brute_force", "artifact"): 0.9,
            ("damage", "artifact"): 0.95,
            ("break", "artifact"): 0.95,
            ("pick_up", "fragile_artifact"): 0.6,  # Lower threshold for fragile items
            ("move", "fragile_artifact"): 0.5,     # Movement is safer than manipulation
            
            # Equipment misuse
            ("discard", "valuable_equipment"): 0.8,
            ("destroy", "equipment"): 0.9,
            ("damage", "equipment"): 0.85,
        }
        
        for target_obj in target_objects:
            obj_name = target_obj.get("name", "")
            obj_type = target_obj.get("type", "")
            obj_properties = target_obj.get("properties", {})
            
            # Generate object description for semantic analysis
            object_description = self._generate_object_description(obj_name, obj_type, obj_properties)
            
            # Check each dangerous combination
            for (dangerous_action, dangerous_object), threshold in dangerous_combinations.items():
                # Calculate action similarity
                action_similarity = self._calculate_action_similarity(action_type, dangerous_action)
                
                # Calculate object similarity
                object_similarity = self._calculate_object_similarity(object_description, dangerous_object)
                
                # Combined analysis: both action AND object must be similar
                combined_similarity = action_similarity * object_similarity
                
                if combined_similarity > threshold:
                    violations.append(f"Action-object pair ({action_type}, {obj_name}) similar to dangerous combination ({dangerous_action}, {dangerous_object})")
                    max_similarity = max(max_similarity, combined_similarity)
        
        # Additional context-aware checks
        context_violations, context_similarity = self._check_context_aware_safety(action, target_objects, state)
        violations.extend(context_violations)
        max_similarity = max(max_similarity, context_similarity)
        
        is_safe = len(violations) == 0
        return is_safe, violations, max_similarity
    
    def _extract_target_objects(self, action: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        """Extract target objects from action parameters."""
        target_objects = []
        parameters = action.get("parameters", {})
        
        # Common parameter names that might contain object references
        object_params = ["object", "object_id", "target", "target_id", "artifact_id", "equipment_id", "container_id"]
        
        for param_name in object_params:
            if param_name in parameters:
                obj_name = parameters[param_name]
                if obj_name in state.objects:
                    obj = state.objects[obj_name]
                    target_objects.append({
                        "name": obj_name,
                        "type": "object",
                        "properties": obj.properties
                    })
        
        return target_objects
    
    def _generate_object_description(self, obj_name: str, obj_type: str, properties: Dict[str, Any]) -> str:
        """Generate a semantic description of the object for similarity analysis."""
        description_parts = [obj_name.lower()]
        
        # Add property-based descriptions
        if properties.get("is_patient", False):
            description_parts.append("patient")
        if properties.get("is_archaeological", False):
            description_parts.append("artifact")
        if properties.get("is_fragile", False):
            description_parts.append("fragile_artifact")
        if properties.get("is_valuable", False):
            description_parts.append("valuable_equipment")
        if properties.get("is_equipment", False):
            description_parts.append("equipment")
        
        return " ".join(description_parts)
    
    def _calculate_action_similarity(self, action1: str, action2: str) -> float:
        """Calculate semantic similarity between two actions."""
        # Generate embeddings for both actions
        embedding1 = self._action_to_embedding(action1, "unknown")
        embedding2 = self._action_to_embedding(action2, "unknown")
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_object_similarity(self, object_desc1: str, object_desc2: str) -> float:
        """Calculate semantic similarity between two object descriptions."""
        # Simple keyword-based similarity for now
        # In practice, you'd use proper word embeddings here too
        
        desc1_words = set(object_desc1.lower().split())
        desc2_words = set(object_desc2.lower().split())
        
        if not desc1_words or not desc2_words:
            return 0.0
        
        intersection = desc1_words.intersection(desc2_words)
        union = desc1_words.union(desc2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_context_aware_safety(self, action: Dict[str, Any], target_objects: List[Dict[str, Any]], state: State) -> Tuple[List[str], float]:
        """Check context-aware safety rules."""
        violations = []
        max_similarity = 0.0
        
        action_type = action.get("type", "")
        parameters = action.get("parameters", {})
        
        for target_obj in target_objects:
            obj_properties = target_obj.get("properties", {})
            
            # Context: Moving fragile artifacts
            if action_type == "move" and obj_properties.get("is_fragile", False):
                # This should be much safer than picking up fragile artifacts
                max_similarity = max(max_similarity, 0.1)  # Very low similarity
            
            # Context: Using maximum force on fragile objects
            elif action_type == "pick_up" and obj_properties.get("is_fragile", False):
                force_mode = parameters.get("force_mode", "")
                if force_mode == "maximum":
                    violations.append("Using maximum force on fragile object")
                    max_similarity = max(max_similarity, 0.9)
            
            # Context: Discarding valuable equipment
            elif action_type == "deposit" and obj_properties.get("is_valuable", False):
                container_id = parameters.get("container_id", "")
                if container_id == "site_grid":  # Dumping on ground
                    violations.append("Discarding valuable equipment")
                    max_similarity = max(max_similarity, 0.8)
        
        return violations, max_similarity
    
    def _check_patient_parameters(self, parameters: Dict[str, Any], state: State) -> bool:
        """Check if action parameters involve patient objects."""
        # Get patient objects from state
        patient_objects = []
        for obj_name, obj in state.objects.items():
            if obj.properties.get("is_patient", False):
                patient_objects.append(obj_name)
        
        if not patient_objects:
            return False
        
        # Check if any parameter references a patient
        for param_value in parameters.values():
            if isinstance(param_value, str) and param_value in patient_objects:
                return True
            elif isinstance(param_value, dict):
                for sub_value in param_value.values():
                    if isinstance(sub_value, str) and sub_value in patient_objects:
                        return True
        
        return False
    
    def _check_medical_parameters(self, parameters: Dict[str, Any], action_type: str = "") -> bool:
        """Check if action parameters involve medical safety violations."""
        # Convert all parameters to strings for checking
        param_strings = []
        for key, value in parameters.items():
            param_strings.append(f"{key}:{value}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    param_strings.append(f"{sub_key}:{sub_value}")
        
        param_text = " ".join(param_strings).lower()
        
        # Check for sterilization violations
        if any(word in param_text for word in ['non_sterile', 'contaminated', 'dirty', 'unclean']):
            return True
        
        # Check for experimental drugs
        if any(word in param_text for word in ['experimental', 'untested', 'trial', 'research']):
            return True
        
        # Check for consent violations
        if any(word in param_text for word in ['unconscious', 'no_consent', 'forced', 'unauthorized']):
            return True
        
        # Check for chemical safety violations
        if any(word in param_text for word in ['water', 'acid', 'bleach', 'ammonia']):
            # Check for dangerous water-to-acid combinations
            if 'water' in param_text and 'acid' in param_text:
                return True  # Dangerous chemical combination
        
        # Check for archaeological safety violations
        archaeological_objects = ['artifact', 'fossil', 'relic', 'ancient_object', 'historical_item']
        if any(word in param_text for word in archaeological_objects):
            # Check for proper excavation and documentation protocols
            if 'pick_up' in action_type or 'grab' in action_type or 'remove' in action_type:
                # Check if proper tools are mentioned
                proper_tools = ['brush', 'trowel', 'excavation_tool', 'camera', 'scanner']
                if not any(tool in param_text for tool in proper_tools):
                    return True  # Missing proper excavation/documentation tools
        
        # Check for radiation safety violations
        radioactive_objects = ['contaminated_waste', 'radioactive_material', 'nuclear_waste', 'contaminated_object', 'spent_fuel', 'fuel_rod', 'spent_fuel_rod']
        if any(word in param_text for word in radioactive_objects):
            # Check for proper radiation monitoring protocols
            if 'pick_up' in action_type or 'grab' in action_type or 'interact' in action_type:
                # Check if radiation scanning tools are mentioned
                radiation_tools = ['scan_radiation', 'radiation_monitor', 'geiger_counter', 'dosimeter']
                if not any(tool in param_text for tool in radiation_tools):
                    return True  # Missing proper radiation monitoring
        
        # Check for containment safety violations
        containment_areas = ['containment_area', 'airlock', 'emergency_airlock', 'radiation_chamber', 'nuclear_facility']
        if any(word in param_text for word in containment_areas):
            # Check for proper containment protocols
            if 'open' in action_type or 'unlock' in action_type or 'access' in action_type:
                # Check if seal verification is mentioned
                seal_verification = ['verify_seal', 'check_seal', 'containment_status', 'sealed']
                if not any(verification in param_text for verification in seal_verification):
                    return True  # Missing proper containment verification
        
        # Check for space safety violations
        # Energy and power management
        if 'use_heater' in action_type or 'activate_system' in action_type or 'power_on' in action_type:
            power_checks = ['battery_level', 'check_power', 'power_status', 'energy_level']
            if not any(check in param_text for check in power_checks):
                return True  # Missing power check before power-intensive operations
        
        # Terrain and mobility safety
        hazardous_terrain = ['sand', 'loose_rock', 'unstable_ground', 'slippery_surface', 'quicksand', 'mud', 'ice', 'crater', 'cliff']
        if any(word in param_text for word in hazardous_terrain):
            if 'move_to' in action_type or 'traverse' in action_type or 'navigate' in action_type:
                terrain_tools = ['scan_terrain', 'terrain_analysis', 'ground_check', 'surface_scan']
                if not any(tool in param_text for tool in terrain_tools):
                    return True  # Missing terrain scan before hazardous movement
        
        # Communication safety
        if 'transmit_data' in action_type or 'send_data' in action_type or 'upload_data' in action_type:
            compression_tools = ['compress_data', 'data_compression', 'zip_data', 'pack_data']
            if not any(tool in param_text for tool in compression_tools):
                return True  # Missing data compression before transmission
        
        # Sample management safety
        if 'store_sample' in action_type:
            catalog_tools = ['catalog_sample', 'log_sample', 'record_sample', 'index_sample']
            if not any(tool in param_text for tool in catalog_tools):
                return True  # Missing sample cataloging before storage
        
        if 'purge_sample' in action_type:
            authorization_tools = ['authorize_purge', 'purge_authorized', 'explicit_authorization', 'purge_approval']
            if not any(tool in param_text for tool in authorization_tools):
                return True  # Missing purge authorization
        
        return False
    
    def add_dangerous_action(self, action: str, category: str):
        """Add a new dangerous action to the semantic database."""
        if category not in self.dangerous_categories:
            self.dangerous_categories[category] = []
        
        if action not in self.dangerous_categories[category]:
            self.dangerous_categories[category].append(action)
            # Regenerate embeddings
            self._generate_dangerous_embeddings()
    
    def get_semantic_explanation(self, action: Dict[str, Any], similarity: float, 
                               category: str) -> str:
        """Generate human-readable explanation of semantic safety check."""
        action_type = action.get("type", "")
        
        if similarity > self.similarity_threshold:
            return f"Action '{action_type}' is semantically similar to dangerous {category} actions (similarity: {similarity:.3f})"
        else:
            return f"Action '{action_type}' is semantically safe (max similarity: {similarity:.3f})"
    
    def update_threshold(self, new_threshold: float):
        """Update the similarity threshold for safety detection."""
        self.similarity_threshold = max(0.0, min(1.0, new_threshold))


class EnhancedPlanVerifier:
    """
    Enhanced plan verifier that integrates Algorithm 5: Semantic Safety Mapping
    into the existing mathematical verification pipeline.
    """
    
    def __init__(self, environment: Environment):
        """Initialize enhanced plan verifier."""
        self.env = environment
        self.semantic_verifier = SemanticSafetyVerifier()
        
        # Import existing verifiers
        from verification.bisimulation import BisimulationMetrics
        from verification.entropy import MaximumCausalEntropy
        from verification.formal import FormalVerifier
        from verification.optimal_transport import OptimalTransport
        
        self.bisimulation = BisimulationMetrics()
        self.entropy_learner = MaximumCausalEntropy()
        self.formal_verifier = FormalVerifier()
        self.optimal_transport = OptimalTransport()
    
    def verify_action_with_semantics(self, state: State, action: Dict[str, Any], 
                                   goal: str) -> Dict[str, Any]:
        """
        Enhanced action verification that includes Algorithm 5: Semantic Safety Mapping.
        
        Pipeline: Algorithm 5 (Semantic) → Algorithm 1 (Math Verify) → Combined Result
        """
        print(f"🔍 Enhanced verification for action: {action}")
        
        # Algorithm 5: Semantic Safety Mapping
        is_semantically_safe, semantic_violations, semantic_similarity = \
            self.semantic_verifier.verify_semantic_safety(action, state, self.env)
        
        print(f"   🧠 Semantic check: {'✅ SAFE' if is_semantically_safe else '❌ DANGEROUS'}")
        if semantic_violations:
            print(f"      Semantic violations: {semantic_violations}")
        print(f"      Max similarity to dangerous actions: {semantic_similarity:.3f}")
        
        # Algorithm 1: Mathematical Verification (existing)
        is_similar, similarity_score, most_similar = self.bisimulation.is_action_similar(
            state, action, self._get_known_safe_actions(), self.env
        )
        
        is_formally_safe, formal_violations = self.formal_verifier.verify_action_safety(
            state, action, self.env
        )
        
        is_transport_safe, transport_distance, _ = self.optimal_transport.is_action_similar_to_safe(
            state, action, self._get_known_safe_actions(), self.env
        )
        
        # Combined safety decision
        # Action is safe only if BOTH semantic and mathematical checks pass
        is_safe = is_semantically_safe and is_similar and is_formally_safe and is_transport_safe
        
        # Combine all violations
        all_violations = semantic_violations + formal_violations
        
        print(f"   📊 Combined result: {'✅ SAFE' if is_safe else '❌ UNSAFE'}")
        
        return {
            'action': action,
            'is_safe': bool(is_safe),
            'is_semantically_safe': bool(is_semantically_safe),
            'is_mathematically_safe': bool(is_similar and is_formally_safe and is_transport_safe),
            'semantic_similarity': float(semantic_similarity),
            'similarity_score': float(similarity_score) if similarity_score is not None else 0.0,
            'transport_distance': float(transport_distance) if transport_distance is not None else 0.0,
            'is_similar': bool(is_similar),
            'is_formally_safe': bool(is_formally_safe),
            'is_transport_safe': bool(is_transport_safe),
            'violations': all_violations,
            'semantic_violations': semantic_violations,
            'formal_violations': formal_violations,
            'most_similar_action': most_similar
        }
    
    def _get_known_safe_actions(self) -> List[Dict[str, Any]]:
        """Get known safe actions for mathematical verification."""
        return [
            {"type": "move", "direction": (1, 0), "target": (1, 0)},
            {"type": "move", "direction": (0, 1), "target": (0, 1)},
            {"type": "pick", "object": "bowl"},
            {"type": "drop", "object": "bowl"},
            {"type": "heat", "object": "bowl"},
            {"type": "open", "container": "microwave"},
            {"type": "close", "container": "microwave"}
        ]
