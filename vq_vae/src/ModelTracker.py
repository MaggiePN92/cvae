from dataclasses import dataclass, field
import time
from typing import List


@dataclass
class ModelTracker:
    start_time_ : float = time.perf_counter()
    durations: List[float] = field(default_factory=list)   
    losses: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def start(self):
        self.start_time_ = time.perf_counter()

    def reset(self):
        self.start_time_ = time.perf_counter()
        self.durations = []
        self.losses = []
        self.epochs = []
    
    def update(self, loss, epoch):
        elapsed_time = time.perf_counter() - self.start_time_

        self.durations.append(elapsed_time)
        self.epochs.append(epoch)
        self.losses.append(loss)

        self.start_time_ = time.perf_counter() # Reset timer

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "durations": self.durations,
            "losses": self.losses,
        }
    