"""A surrogate for single heralded entanglement generation.
"""

import sequence as sq
SINGLE_HERALDED_SURROGATE = "single_heralded_surrogate"


@sq.EntanglementGenerationA.register(SINGLE_HERALDED_SURROGATE)
class SingleHeraldedSurrogateA(sq.SingleHeraldedA):
    """
    A surrogate for SingleHeraldedA. It should behave very similarly to the original 
    SingleHeraldedA protocol performance wise, but will skip a lot of the actual physical operations.
    """
    def __init__(self, owner: sq.Node, name: str, middle: str, other: str, memory: sq.Memory, 
                 raw_fidelity: float, raw_epr_errors: list[float] = None):
        super().__init__(owner, name, middle, other, memory, raw_fidelity, raw_epr_errors)
        self.memory_efficiency: float = 1
        self.memory_efficiency_other: float = 1
        self.transmission_prob: float = 1
        self.optical_bsm_success_prob: float = 1
        self.detector_efficiency: float = 1


@sq.EntanglementGenerationB.register(SINGLE_HERALDED_SURROGATE)
class SingleHeraldedSurrogateB(sq.SingleHeraldedB):
    """
    A surrogate for SingleHeraldedB
    """
    def __init__(self, owner: sq.BSMNode, name: str, others: list[str]):
        super().__init__(owner, name, others)
        