"""A surrogate for a entanglement generation in a repeater chain 
"""

import sequence as sq
from torch import nn

REPEATER_CHAIN_SURROGATE = "repeater_chain_surrogate"


@sq.EntanglementGenerationA.register(REPEATER_CHAIN_SURROGATE)
class RepeaterChainASurrogate(sq.EntanglementGenerationA):
    """A surrogate for entanglement generation in a repeater chain.

    Class Attributes:
        link_distance: dict[str, float] = None                # link name to its distance
        attenuation: float = None                             # quantum channel attenuation
        memory_efficiency: float = None                       # memory efficiency
        detector_efficiency: float = None                     # detector efficiency
        swapping_probability: float = None                    # swapping operation success probability
        classical_delay_node: float = None                    # classical delay
        classical_delay_hop: float = None                     # hop delay
        initial_fidelity: float = None                        # initial fidelity of the entangled state
        coherence_time: float = None                          # coherence time of the quantum memory
        surrogates: dict[tuple[str, ...], nn.Module]] = {}    # (node, ...) to surrogate model
    """

    link_distance: dict[str, float] = {}
    attenuation: float = None
    memory_efficiency: float = None
    detector_efficiency: float = None
    swapping_probability: float = None
    classical_delay_node: float = None
    classical_delay_hop: float = None
    initial_fidelity: float = None
    coherence_time: float = None
    surrogates: dict[tuple[str, ...], nn.Module] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    @classmethod
    def set_surrogate_parameter(cls, timeline: sq.Timeline, classical_delay_node: float, classical_delay_hop: float) -> None:
        """Set the surrogate parameters from the timeline.

        Args:
            timeline (sq.Timeline): The simulation timeline containing the entities.
            classical_delay_node (float): The classical delay for the node.
            classical_delay_hop (float): The classical delay for each hop.
        """
        cls.classical_delay_node = classical_delay_node
        cls.classical_delay_hop = classical_delay_hop
        link_distance = {}

        for name, entity in timeline.entities.items():
            if isinstance(entity, sq.ClassicalChannel) and 'BSM' not in entity.name:
                link_distance[name] = entity.distance
            if isinstance(entity, sq.QuantumChannel):
                if cls.attenuation is None:
                    cls.attenuation = entity.attenuation
            elif isinstance(entity, sq.Memory):
                if cls.memory_efficiency is None:
                    cls.memory_efficiency = entity.efficiency
                if cls.initial_fidelity is None:
                    cls.initial_fidelity = entity.raw_fidelity
                if cls.coherence_time is None:
                    cls.coherence_time = entity.coherence_time
            elif isinstance(entity, sq.QuantumRouter):
                if cls.swapping_probability is None:
                    cls.swapping_probability = entity.swapping_success_prob
            elif isinstance(entity, sq.Detector):
                if cls.detector_efficiency is None:
                    cls.detector_efficiency = entity.efficiency

        cls.link_distance = link_distance


@sq.EntanglementGenerationB.register(REPEATER_CHAIN_SURROGATE)
class RepeaterChainBSurrogate(sq.EntanglementGenerationB):
    pass

