"""The traffic pattern for the simulation.
"""

import numpy as np
from collections import defaultdict


class Traffic:
    """The traffic pattern for the simulation.
    """
    def __init__(self, config_file: str, seed: int = 0):
        self.config_file = config_file
        self.seed = seed
        self.distributions = defaultdict(dict)  # src -> dict of (dst, distribution)
        self.set_distributions()

    def set_distributions(self):
        """Set the traffic distributions based on the configuration file.
           Assume Poisson process for request arrivals, which is commonly used in network traffic modeling.
           Counting view: Number of arrivals in time interval T -- follows a Poisson distribution
           Timing view: Time between arrivals -- follows an Exponential distribution
        """
        np.random.seed(self.seed)
        match self.config_file:
            case 'bottleneck_2':
                # self.distributions['router_0'] = {'router_3': lambda: np.random.exponential(scale=1/100)}  # 100 requests per second
                self.distributions['router_1'] = {'router_4': lambda: np.random.exponential(scale=1/50)}   # 50 requests per second
                # self.distributions['router_2'] = {'router_5': lambda: np.random.exponential(scale=1/20)}   # 20 requests per second
            case topology if topology.startswith('linear_'):
                nodes = int(topology.removeprefix('linear_'))
                self.distributions['router_0'] = {f'router_{nodes - 1}': lambda: 1}  # 1 requests per second
                # self.distributions[f'router_{nodes - 1}'] = {'router_0': lambda: 1}  # 1 requests per second
            case _:
                raise ValueError(f"Unsupported config file: {self.config_file}")
    
