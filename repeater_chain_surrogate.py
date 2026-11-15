"""A surrogate for a entanglement generation in a repeater chain 
"""

import sequence as sq
from torch import nn

REPEATER_CHAIN_SURROGATE = "repeater_chain_surrogate"
from sequence.constants import MILLISECOND


class RepeaterChainSurrogateMessage(sq.Message):
    """Message used by the repeater chain surrogate protocol.

    Attributes:
        msg_type (sq.GenerationMsgType): defines the message type.
        receiver (str): name of destination protocol instance.
        protocol_type (str): identifies the generation protocol.
        time (int): surrogate time (if `msg_type == NEGOTIATE`).
    """

    __slots__ = ["time", "fidelity", "reservation"]

    def __init__(self, msg_type: sq.GenerationMsgType, protocol_type: str, reservation: sq.Reservation, **kwargs):
        super().__init__(msg_type, receiver=REPEATER_CHAIN_SURROGATE)
        self.protocol_type = protocol_type
        self.reservation = reservation
        self.time: int | None = None        # the entanglement succeed time computed from the surrogate
        self.fidelity: float | None = None  # the fidelity of the generated entangled pair

        fields = {sq.GenerationMsgType.NEGOTIATE: ["time", "fidelity"],
                  sq.GenerationMsgType.NEGOTIATE_ACK: []}

        if msg_type in fields:
            for field in fields[msg_type]:
                setattr(self, field, kwargs.get(field))
        else:
            raise ValueError(f"RepeaterChainSurrogate generated invalid message type {msg_type}")

    def __repr__(self):
        match self.msg_type:
            case sq.GenerationMsgType.NEGOTIATE:
                return f"type: {self.msg_type}, reservation: {self.reservation}, time: {self.time}, fidelity: {self.fidelity}"
            case sq.GenerationMsgType.NEGOTIATE_ACK:
                return f"type: {self.msg_type}, reservation: {self.reservation}"
            case _:
                raise ValueError(f"RepeaterChainSurrogate generated invalid message type {self.msg_type}")


@sq.EntanglementGenerationA.register(REPEATER_CHAIN_SURROGATE)
class RepeaterChainASurrogate(sq.EntanglementGenerationA):
    """A surrogate for entanglement generation in a repeater chain.

    Class Attributes:
        link_distance: dict[str, float] = None             # link name to its distance
        attenuation: float = None                          # quantum channel attenuation
        memory_efficiency: float = None                    # memory efficiency
        detector_efficiency: float = None                  # detector efficiency
        swapping_probability: float = None                 # swapping operation success probability
        classical_delay_node: float = None                 # classical delay
        classical_delay_hop: float = None                  # hop delay
        initial_fidelity: float = None                     # initial fidelity of the entangled state
        coherence_time: float = None                       # coherence time of the quantum memory
        surrogates: dict[tuple[str, ...], nn.Module]] = {} # (node, ...) to surrogate model
    
    Instance Attributes:
        primary: bool = False  # the primary node in the chain is the initiator
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

    def __init__(self, owner: sq.Node, name: str, middle: str, other: str, memory: sq.Memory):
        super().__init__(owner, name, middle, other, memory)
        self.protocol_type = REPEATER_CHAIN_SURROGATE
        self.name = REPEATER_CHAIN_SURROGATE # every protocol instance has the same name
        self.others: list[str] = [] # if the node is the primary, this stores the other nodes in the chain

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

    def surrogate_parameater_is_set(self) -> bool:
        """Check if all surrogate parameters are set.

        Returns:
            bool: True if all surrogate parameters are set, False otherwise.
        """
        return all([self.classical_delay_node is not None,
                    self.classical_delay_hop is not None,
                    self.attenuation is not None,
                    self.memory_efficiency is not None,
                    self.initial_fidelity is not None,
                    self.coherence_time is not None,
                    self.swapping_probability is not None,
                    self.detector_efficiency is not None,
                    self.link_distance is not None])

    def _entanglement_succeed(self):
        """"""
        pass

    def emit_event(self):
        """"""
        pass

    def set_others(self, protocol: str, node: str, memories: list[str]):
        """Set other protocol-specific parameters.

        Args:
            protocol (str): The remote protocol name. (not used)
            node (str): The remote node name. (not used)
            memories (list[str]): The list of memory IDs. (not used)
        """
        if self.surrogate_parameater_is_set() is False:
            raise ValueError("Surrogate parameters are not set.")
        self.remote_protocol_name = None
        self.remote_memo_id = None
        self.primary = True if self.owner.name == self.rule.reservation.path[0] else False

    def start(self):
        """"""
        if self.surrogate_parameater_is_set() is False:
            raise ValueError("Surrogate parameters are not set.")
        if self.primary: # only the protocol at primary node starts, protocol at all other nodes waits
            # calculate the time and fidelity from the surrogate
            time = 100 * MILLISECOND
            fidelity = 0.8
            path = self.rule.reservation.path
            n = len(path)
            for i in range(1, n):
                message = RepeaterChainSurrogateMessage(sq.GenerationMsgType.NEGOTIATE, REPEATER_CHAIN_SURROGATE, 
                                                        self.rule.reservation, time=time, fidelity=fidelity)
                self.owner.send_message(path[i], message)

    def received_message(self, src: str, msg: RepeaterChainSurrogateMessage):
        """Handle received messages.

        Args:
            src (str): The source of the message.
            msg (RepeaterChainSurrogateMessage): The received message.
        """
        if self.surrogate_parameater_is_set() is False:
            raise ValueError("Surrogate parameters are not set.")
        if msg.reservation == self.rule.reservation: # make sure talking to the correct protocol
            if msg.msg_type == sq.GenerationMsgType.NEGOTIATE:
                # handle the negotiation message
                pass

@sq.EntanglementGenerationB.register(REPEATER_CHAIN_SURROGATE)
class RepeaterChainBSurrogate(sq.EntanglementGenerationB):
    pass

