import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "experiment_state.json"
        self.state = self.load_state()

    def load_state(self):
        """Load the experiment state from a file."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as file:
                return json.load(file)
        return {"success": [], "failed": [], "completed": []}

    def save_state(self):
        """Save the experiment state to a file."""
        with open(self.state_path, 'w') as file:
            json.dump(self.state, file)

    def update_state(self, experiment_name, score=None, error=None, status="success"):
        """Update the state with the result of an experiment."""
        if status not in self.state:
            self.state[status] = []
        self.state[status].append({
            "experiment": experiment_name,
            "score": score,
            "error": error,
            "status": status
        })
        self.save_state()
        logger.info(f"State saved to {self.state_path}")