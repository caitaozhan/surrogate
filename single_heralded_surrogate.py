"""A surrogate for single heralded entanglement generation.
"""

from scipy.stats import geom
from scipy.stats._distn_infrastructure import rv_discrete_frozen

import sequence as sq
from sequence.message import Message

SINGLE_HERALDED_SURROGATE = "single_heralded_surrogate"


class SurrogateGenerationMessage(Message):
    """Message used by surrogate entanglement generation protocols.

    Attributes:
        msg_type (sq.GenerationMsgType): defines the message type.
        receiver (str): name of destination protocol instance.
        protocol_type (str): identifies the generation protocol.
        time (int): surrogate time (if `msg_type == NEGOTIATE`).
        fidelity (float): surrogate fidelity (if `msg_type == NEGOTIATE`).
    """

    __slots__ = ["time", "fidelity"]

    def __init__(self, msg_type: sq.GenerationMsgType, receiver: str | None, protocol_type: str, **kwargs):
        super().__init__(msg_type, receiver)

        self.protocol_type = protocol_type

        self.time: int | None = None
        self.fidelity: float | None = None

        fields = {sq.GenerationMsgType.NEGOTIATE: ["time", "fidelity"],
                  sq.GenerationMsgType.NEGOTIATE_ACK: []}

        if msg_type in fields:
            for field in fields[msg_type]:
                setattr(self, field, kwargs.get(field))
        else:
            raise ValueError(f"SurrogateGeneration generated invalid message type {msg_type}")

    def __repr__(self):
        match self.msg_type:
            case sq.GenerationMsgType.NEGOTIATE:
                return f"type: {self.msg_type}, time: {self.time}, fidelity: {self.fidelity}"
            case sq.GenerationMsgType.NEGOTIATE_ACK:
                return f"type: {self.msg_type}"
            case _:
                raise ValueError(f"SurrogateGeneration generated invalid message type {self.msg_type}")


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
    cchannel_delay: dict[str, float] = None                    # classical channel name to its delay
    surrogates: dict[tuple[str, str], rv_discrete_frozen] = {}  # (primary node, non-primary node) to their surrogate

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
        self.surrogate = None

    def get_memory_name(self, protocol_name: str) -> str:
        """Get the memory ID associated with the given protocol instance. 
        Eg. EGA.router_1.MemoryArray[0] -> router_1.MemoryArray[0]
        
        Args:
            protocol_name (str): The name of the protocol instance.

        Returns:
            str: The memory ID associated with the given protocol instance.
        """
        return protocol_name.split('.', 1)[1]
    
    def get_qchannel_name(self, node_name: str, bsm_node_name: str) -> str:
        """Get the quantum channel name between the given node and BSM node.

        Args:
            node_name (str): The name of the node.
            bsm_node_name (str): The name of the BSM node.

        Returns:
            str: The name of the quantum channel.
        """
        return f"qc-{node_name}-{bsm_node_name}"

    def get_detector_name(self, bsm_node_name: str, index: int) -> str:
        """Get the detector name for the given BSM node and index.

        Args:
            bsm_node_name (str): The name of the BSM node.
            index (int): The index of the detector.

        Returns:
            str: The name of the detector.
        """
        return f'{bsm_node_name}.BSM_{index}'

    def get_bsm_name(self, bsm_node_name: str) -> str:
        """Get the BSM name for the given BSM node.

        Args:
            bsm_node_name (str): The name of the BSM node.

        Returns:
            str: The name of the BSM.
        """
        return f'{bsm_node_name}.BSM'

    def get_cchannel_name(self, node_name: str, other_node_name: str) -> str:
        """Get the classical channel name between the given node and other node.

        Args:
            node_name (str): The name of the node.
            other_node_name (str): The name of the other node.

        Returns:
            str: The name of the classical channel.
        """
        return f"cc-{node_name}-{other_node_name}"

    def get_rtt_count(self) -> float:
        """If this node appears first in the reservation path, the RTT is 2.5
           If the remote node appears first in the reservation path, the RTT is 2

        Returns:
            float: The RTT for a single entanglement generation attempt.
        """
        path = self.rule.reservation.path
        for node in path:
            if node == self.owner.name:
                return 2.5
            elif node == self.remote_node_name:
                return 2
        else:
            raise ValueError("Node not found in the path.")

    def set_others(self, protocol: str, node: str, memories: list[str]) -> None:
        """Pair this protocol with the remote protocol instance. Sets the primary flag based on node names.
           Also set up the surrogate model.

        Args:
            protocol (str): The name of the remote protocol instance.
            node (str): The name of the remote node.
            memories (list[str]): The list of memory IDs at the remote node.
        """
        super().set_others(protocol, node, memories)
        if self.surrogate_parameter_is_set():
            if self.primary:
                key = (self.owner.name, node)
                if key in self.surrogates:
                    self.surrogate = self.surrogates[key]
                else:
                    # get memory efficiency
                    memory = self.get_memory_name(self.name)
                    memory_other = self.get_memory_name(self.remote_protocol_name)
                    memory_efficiency = self.memory_efficiency[memory]
                    memory_efficiency_other = self.memory_efficiency[memory_other]
                    # get quantum channel transmission probability
                    qchannel = self.get_qchannel_name(self.owner.name, self.middle)
                    qchannel_other = self.get_qchannel_name(self.remote_node_name, self.middle)
                    qchannel_transmission_probability = self.qchannel_transmission_probability[qchannel]
                    qchannel_transmission_probability_other = self.qchannel_transmission_probability[qchannel_other]
                    # get detector efficiency
                    detector1 = self.get_detector_name(self.middle, 0)
                    detector2 = self.get_detector_name(self.middle, 1)
                    detector_efficiency1 = self.detector_efficiency[detector1]
                    detector_efficiency2 = self.detector_efficiency[detector2]
                    # get bsm success rate
                    bsm = self.get_bsm_name(self.middle)
                    bsm_success_rate = self.bsm_success_rate[bsm]
                    # overall success probability per attempt
                    overall_probability = (memory_efficiency * memory_efficiency_other 
                                           * qchannel_transmission_probability * qchannel_transmission_probability_other
                                           * detector_efficiency1 * detector_efficiency2 * bsm_success_rate)
                    # get classical channel delay
                    cchannel = self.get_cchannel_name(self.owner.name, self.remote_node_name)
                    cchannel_delay = self.cchannel_delay[cchannel]
                    # get time per entanglement generation attempt
                    rtt_count = self.get_rtt_count()
                    time_per_attempt = rtt_count * (2 * cchannel_delay)
                    # finally get the expected time for successful entanglement generation
                    expected_time = time_per_attempt / overall_probability
                    # store the expected time for this surrogate
                    self.surrogate[key] = (geom(overall_probability), time_per_attempt)
                    sq.log.logger.info(f"Surrogate key: {key}, Overall probability: {overall_probability}, "
                                       f"Time per attempt: {time_per_attempt}, Expected time: {expected_time}")
        
    @classmethod
    def set_surrogate_parameter_from_timeline(cls, timeline: sq.Timeline) -> None:
        """Set the surrogate parameters from the timeline.

        Args:
            timeline (sq.Timeline): The simulation timeline containing the entities.
        """
        cchannel_delay = {}
        memory_efficiency = {}
        qchannel_transmission_probability = {}
        detector_efficiency = {}
        bsm_success_rate = {}

        for name, entity in timeline.entities.items():
            if isinstance(entity, sq.ClassicalChannel):
                cchannel_delay[name] = entity.delay
            elif isinstance(entity, sq.Memory):
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
        cls.cchannel_delay = cchannel_delay

    def surrogate_parameter_is_set(self) -> bool:
        """Check if the surrogate parameters are all set.

        Return:
            True if all surrogate parameters are set, False otherwise.
        """
        if (self.cchannel_delay and self.memory_efficiency and self.qchannel_transmission_probability 
                and self.detector_efficiency and self.bsm_success_rate):
            return True
        return False

    def start(self) -> None:
        """Method to start "one round" in the entanglement generation protocol (there are two rounds in Barrett-Kok).

        Will start negotiations with other protocol (if primary).

        Side Effects:
            Will send message through attached node.
        """
        if self.surrogate_parameter_is_set():
            sq.log.logger.info(f"{self.owner.name} {self.name} protocol start with partner {self.remote_protocol_name}")
        else:
            super().start()

            # if self not in self.owner.protocols:
            #     return

            # if self.update_memory() and self.primary:
            #     self.qc_delay = self.owner.qchannels[self.middle].delay
            #     message = sq.EntanglementGenerationMessage(sq.GenerationMsgType.NEGOTIATE,
            #                                                 self.remote_protocol_name,
            #                                                 protocol_type=self.protocol_type,
            #                                                 qc_delay=self.qc_delay)
            #     self.owner.send_message(self.remote_node_name, message)


@sq.EntanglementGenerationB.register(SINGLE_HERALDED_SURROGATE)
class SingleHeraldedBSurrogate(sq.SingleHeraldedB):
    """
    A surrogate for SingleHeraldedB
    """
    def __init__(self, owner: sq.BSMNode, name: str, others: list[str]):
        super().__init__(owner, name, others)
        self.protocol_type = SINGLE_HERALDED_SURROGATE
