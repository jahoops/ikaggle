import os
import json
import pickle
from typing import List, Dict, Any
from datetime import datetime
from ikaggle.AAA.models.individual import Individual, genome_to_int, int_to_genome

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles special objects"""
    def default(self, obj):
        if isinstance(obj, Individual):
            # Convert Individual to dict with integer genome
            return {
                'id': obj.id,
                'birth_gen': obj.birth_gen,
                'birth_time': obj.birth_time.isoformat(),
                'parent_ids': obj.parent_ids,
                'genome_int': genome_to_int(obj.genome),
                'genome_length': obj.genome_length,
                'active_features': obj.active_features,
                'fitness': float(obj.fitness.values[0]) if obj.fitness.valid else float('inf'),
                # ...rest of fields...
            }
        elif hasattr(obj, 'values') and hasattr(obj, 'valid'):
            return float(obj.values[0]) if obj.valid else float('inf')
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_generation(population: List, gen: int, save_dir: str) -> str:
    """Save generation with JSON serializable data"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create summary of generation
    summary = {
        'generation': gen,
        'timestamp': datetime.now(),
        'population_size': len(population),
        'individuals': [ind.to_dict() for ind in population]
    }
    
    # Save JSON summary
    json_path = os.path.join(save_dir, f'gen_{gen:04d}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=CustomJSONEncoder)
    
    # Save full pickle backup
    pickle_path = os.path.join(save_dir, f'gen_{gen:04d}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(population, f)
    
    return save_dir

def load_generation(gen: int, save_dir: str) -> List[Individual]:
    """Load generation from JSON file"""
    json_path = os.path.join(save_dir, f'gen_{gen:04d}.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Generation {gen} not found in {save_dir}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        individuals = []
        for ind_data in data['individuals']:
            # Convert integer back to genome
            genome = int_to_genome(ind_data['genome_int'], ind_data['genome_length'])
            ind_data['genome'] = genome
            individuals.append(Individual.from_dict(ind_data))
    return individuals
