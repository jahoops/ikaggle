from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import uuid
from deap import base, creator

# Create the fitness class if it doesn't exist
if 'FitnessMin' not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

def genome_to_int(genome: List[int]) -> int:
    """Convert binary genome to integer for storage"""
    if not genome:
        return 0
    return int(''.join(map(str, map(int, genome))), 2)

def int_to_genome(num: int, length: int) -> List[int]:
    """Convert integer back to binary genome"""
    binary = bin(num)[2:]  # Remove '0b' prefix
    # Pad with zeros to match original length
    binary = binary.zfill(length)
    return [int(b) for b in binary]

@dataclass
class Individual:
    """Represents a single individual in the genetic algorithm with tracking"""
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    birth_gen: int = 0
    birth_time: datetime = field(default_factory=datetime.now)
    parent_ids: List[str] = field(default_factory=list)
    
    # Genetic information - store as integer
    genome: List[int] = field(default_factory=list)
    genome_length: int = field(init=False)
    active_features: List[str] = field(default_factory=list)
    
    # Performance metrics
    fitness: creator.FitnessMin = field(default_factory=lambda: creator.FitnessMin())
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    
    # Additional metrics
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_stats: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    
    # History
    mutation_count: int = 0
    crossover_count: int = 0
    improvement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Add removal tracking
    removal_gen: Optional[int] = None
    removal_reason: Optional[str] = None
    lifetime_generations: int = field(init=False)
    
    def __post_init__(self):
        self.genome_length = len(self.genome)
        self.lifetime_generations = 0
    
    def update_fitness(self, new_fitness: float, gen: int):
        """Update fitness and track improvement history"""
        if not isinstance(self.fitness, creator.FitnessMin):
            self.fitness = creator.FitnessMin()
            
        old_fitness = float('inf') if not self.fitness.valid else self.fitness.values[0]
        self.fitness.values = (new_fitness,)
        
        if new_fitness < old_fitness:
            self.improvement_history.append({
                'generation': gen,
                'old_fitness': old_fitness,
                'new_fitness': new_fitness,
                'time': datetime.now()
            })
    
    def record_mutation(self):
        """Track mutation events"""
        self.mutation_count += 1
    
    def record_crossover(self, parent2_id: str):
        """Track crossover events"""
        self.crossover_count += 1
        if parent2_id not in self.parent_ids:
            self.parent_ids.append(parent2_id)
    
    def add_prediction_stats(self, predictions: np.ndarray):
        """Record statistics about model predictions"""
        self.prediction_stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions))
        }
    
    def mark_removed(self, generation: int, reason: str = "replaced"):
        """Track when and why an individual was removed from population"""
        self.removal_gen = generation
        self.removal_reason = reason
        self.lifetime_generations = generation - self.birth_gen
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary for storage"""
        return {
            'id': self.id,
            'birth_gen': self.birth_gen,
            'birth_time': self.birth_time.isoformat(),
            'parent_ids': self.parent_ids,
            'genome_int': genome_to_int(self.genome),  # Store as integer
            'genome_length': self.genome_length,
            'active_features': self.active_features,
            'fitness': float(self.fitness.values[0]) if self.fitness.valid else float('inf'),
            'validation_score': self.validation_score,
            'test_score': self.test_score,
            'feature_importance': self.feature_importance,
            'prediction_stats': self.prediction_stats,
            'training_time': self.training_time,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'improvement_history': self.improvement_history,
            'removal_gen': self.removal_gen,
            'removal_reason': self.removal_reason,
            'lifetime_generations': self.lifetime_generations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create individual from dictionary"""
        genome = int_to_genome(data['genome_int'], data['genome_length'])
        data['genome'] = genome
        data['birth_time'] = datetime.fromisoformat(data['birth_time'])
        return cls(**data)
