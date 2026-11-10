"""A surrogate for single heralded entanglement generation.
"""

import sequence as sq
SINGLE_HERALDED_SURROGATE = "single_heralded_surrogate"


@sq.EntanglementGenerationA.register(SINGLE_HERALDED_SURROGATE)
class SingleHeraldedASurrogate(sq.SingleHeraldedA):
    """
    A surrogate for SingleHeraldedA. It should behave very similarly to the original 
    SingleHeraldedA protocol performance wise, but will skip a lot of the actual physical operations.

    Class Attributes:
        memory_efficiency (dict[str, float]): Memory name to its efficiency.
        qchannel_transmission_probability (dict[str, float]): Quantum channel name to its transmission efficiency.
        detector_efficiency (dict[str, float]): Detector name to efficiency.
        bsm_success_rate (dict[str, float]): BSM name to success rate.
    
    Note: The class attributes should be set at the beginning of the simulation. 
          If the class attributes are not all set, the surrogate will reduce to the original SingleHeraldedA.
    """
    memory_efficiency: dict[str, float] = None                 # memory name to its efficiency
    qchannel_transmission_probability: dict[str, float] = None # quantum channel name to its transmission efficiency
    detector_efficiency: dict[str, float] = None               # detector name to efficiency
    bsm_success_rate: dict[str, float] = None                  # BSM name to success rate

    def __init__(self, owner: sq.Node, name: str, middle: str, other: str, memory: sq.Memory, 
                 raw_fidelity: float = None, raw_epr_errors: list[float] = None):
        """Initialize the SingleHeraldedASurrogate protocol.
        
        Args:
            owner (sq.Node): The node that owns this protocol instance.
            name (str): The name of the protocol instance.
            middle (str): The middle node in the entanglement generation.
            other (str): The other node in the entanglement generation.
            memory (sq.Memory): The memory used for entanglement generation.
            raw_fidelity (float, optional): The raw fidelity of the entangled state.
            raw_epr_errors (list[float], optional): The raw EPR errors of the entangled state.
        """
        super().__init__(owner, name, middle, other, memory, raw_fidelity, raw_epr_errors)
        self.protocol_type = SINGLE_HERALDED_SURROGATE

    @classmethod
    def set_surrogate_from_timeline(cls, timeline: sq.Timeline) -> None:
        """Set the surrogate parameters from the timeline.

        Args:
            timeline (sq.Timeline): The simulation timeline containing the entities.
        """
        memory_efficiency = {}
        qchannel_transmission_probability = {}
        detector_efficiency = {}
        bsm_success_rate = {}

        for name, entity in timeline.entities.items():
            if isinstance(entity, sq.Memory):
                memory_efficiency[name] = entity.efficiency
            elif isinstance(entity, sq.QuantumChannel):
                qchannel_transmission_probability[name] = 1 - entity.loss
            elif isinstance(entity, sq.Detector):
                detector_efficiency[name] = entity.efficiency
            elif isinstance(entity, sq.SingleHeraldedBSM):
                bsm_success_rate[name] = entity.success_rate

        cls.memory_efficiency = memory_efficiency
        cls.qchannel_transmission_probability = qchannel_transmission_probability
        cls.detector_efficiency = detector_efficiency
        cls.bsm_success_rate = bsm_success_rate

    def surrogate_is_ready(self) -> bool:
        """Check if the surrogate parameters are all set.
        """
        if (self.memory_efficiency and self.qchannel_transmission_probability and self.detector_efficiency 
                and self.bsm_success_rate):
            return True
        return False


@sq.EntanglementGenerationB.register(SINGLE_HERALDED_SURROGATE)
class SingleHeraldedBSurrogate(sq.SingleHeraldedB):
    """
    A surrogate for SingleHeraldedB
    """
    def __init__(self, owner: sq.BSMNode, name: str, others: list[str]):
        super().__init__(owner, name, others)
        self.protocol_type = SINGLE_HERALDED_SURROGATE
