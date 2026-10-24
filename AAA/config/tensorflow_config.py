import os
import platform
import subprocess
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def get_cpu_features() -> Dict[str, bool