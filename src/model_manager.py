import signal
import subprocess
import time
from typing import Optional
import requests
from openai import OpenAI
from src.config import *

class ModelManager:
    """Manages lifecycle of LM Studio model server instances"""

    def __init__(self):
        """Initialize model manager with empty state"""
        self.current_model: Optional[str] = None  # Display name from MODEL_MAPPING
        self.process: Optional[subprocess.Popen] = None  # Server process handle
        self.client: Optional[OpenAI] = None  # API client instance
        self.current_model_internal_name: Optional[str] = None  # Model ID for API

    def start_server(self, model_name: str) -> None:
        """Launch LM Studio server with specified model

        Args:
            model_name: Key from MODEL_MAPPING configuration

        Raises:
            ValueError: For unregistered models
            FileNotFoundError: Missing model/LM Studio executable
            TimeoutError: Server startup failure
        """
        # Validate model configuration
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Model {model_name} not in config")

        # Cleanup previous instance
        self.stop_server()

        model_config = MODEL_MAPPING[model_name]
        model_rel_path = os.path.join(MODEL_DIR, *model_config["path"])

        # Verify model file existence
        if not os.path.exists(model_rel_path):
            raise FileNotFoundError(f"Model file missing: {model_rel_path}")
        if not os.path.exists(LM_STUDIO_PATH):
            raise FileNotFoundError(f"LM Studio not found: {LM_STUDIO_PATH}")

        # Windows-specific process flags
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0

        # Launch LM Studio with API mode
        self.process = subprocess.Popen(
            [LM_STUDIO_PATH, "--model", model_rel_path, "--api-port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags  # Process group handling
        )

        # Server health check with timeout
        timeout = 40
        start_time = time.time()
        while True:
            try:
                response = requests.get(f"http://localhost:{PORT}/v1/models", timeout=2)
                if response.status_code == 200:
                    models = response.json()["data"]
                    # Verify model loaded
                    if any(m['id'] == model_config["internal_name"] for m in models):
                        break
            except Exception as e:
                if time.time() - start_time > timeout:
                    self.stop_server()
                    raise TimeoutError("Server failed to start") from e
                time.sleep(2)  # Retry interval

        # Initialize API client
        self.current_model = model_name
        self.current_model_internal_name = model_config["internal_name"]
        self.client = OpenAI(
            base_url=f"http://localhost:{PORT}/v1",
            api_key="lm-studio"  # LM Studio's fixed API key
        )

    def stop_server(self) -> None:
        """Terminate server process and clean up resources"""
        if self.process:
            try:
                if os.name == 'nt':
                    # Windows process termination
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    # Unix process group termination
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process already exited

            # Reset state
            self.process = None
            self.client = None
            self.current_model = None
            self.current_model_internal_name = None