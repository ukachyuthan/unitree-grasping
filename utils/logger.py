import logging
import os
from datetime import datetime


class Logger:
    """Centralized logging utility"""
    
    def __init__(self, log_dir: str = "data/logs", name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
