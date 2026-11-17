"""A surrogate for a entanglement generation in a repeater chain 
"""

import sequence as sq
import torch
from pathlib import Path
from typing import Callable
import numpy as np
from train_surrogate_chain import GLOBAL_FEATURE_NAMES, build_condition_sampler, load_model

REPEATER_CHAIN_SURROGATE = "repeater_chain_surrogate"
from sequence.constants import MILLISECOND


class RepeaterChainSurrogateMessage(sq.Message):
    """Message used by the repeater chain surrogate protocol.

    Attributes:
        msg_type (sq.GenerationMsgType): defines the message type.
        receiver (str): name of destination protocol instance.
        protocol_type (str): identifies the generation protocol.
        reservation (sq.Reservation): the reservation associated with the entanglement generation.
        memo_id (str): the identifier of the memory involved in the entanglement generation.
        time (int): surrogate time (if `msg_type == NEGOTIATE`).
        fidelity (float): the fidelity of the generated entangled pair. (if `msg_type == NEGOTIATE`)
    """

    def __init__(self, msg_type: sq.GenerationMsgType, protocol_type: str, reservation: sq.Reservation, memo_id: str, **kwargs):
        super().__init__(msg_type, receiver=REPEATER_CHAIN_SURROGATE)
        self.protocol_type = protocol_type
        self.reservation = reservation
        self.memo_id: str = memo_id
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
        link_distance: dict[str, float] = None     # link name to its distance
        attenuation: float = None                  # quantum channel attenuation
        memory_efficiency: float = None            # memory efficiency
        detector_efficiency: float = None          # detector efficiency
        swapping_probability: float = None         # swapping operation success probability
        classical_delay_node: float = None         # classical delay
        classical_delay_hop: float = None          # hop delay
        initial_fidelity: float = None             # initial fidelity of the entangled state
        coherence_time: float = None               # coherence time of the quantum memory
        surrogate: Callable[..., np.ndarray] = None  # shared conditional sampler
        model_path: str                            # the path for the surrogate model
        
    Instance Attributes:
        primary: bool = False  # the primary node in the chain is the initiator
    """

    link_distance: dict[str, float] = None
    cchannel_delay: dict[str, int] = None
    attenuation: float = None
    memory_efficiency: float = None
    detector_efficiency: float = None
    swapping_probability: float = None
    classical_delay_node: float = None
    classical_delay_hop: float = None
    initial_fidelity: float = None
    coherence_time: float = None
    model_path: str = "models/combo_big_lr_validation_nll=-0.8771663870.pt"
    surrogate: Callable[..., np.ndarray] | None = None

    def __init__(self, owner: sq.Node, name: str, middle: str, other: str, memory: sq.Memory):
        super().__init__(owner, name, middle, other, memory)
        self.protocol_type = REPEATER_CHAIN_SURROGATE
        self.name = REPEATER_CHAIN_SURROGATE # every protocol instance has the same name
        self.fidelity: float = 0

    @classmethod
    def set_surrogate_parameter(cls, timeline: sq.Timeline, classical_delay_node: float, classical_delay_hop: float) -> None:
        """Set the surrogate parameters from the timeline.

        Args:
            timeline (sq.Timeline): The simulation timeline containing the entities.
            classical_delay_node (float): The classical delay for the node.
            classical_delay_hop (float): The classical delay for each hop.
        """
        if cls.surrogate is None:
            artifact = load_model(Path(cls.model_path), torch.device("cpu"))
            cls.surrogate = staticmethod(build_condition_sampler(artifact, torch.device("cpu")))

        cls.classical_delay_node = classical_delay_node
        cls.classical_delay_hop = classical_delay_hop
        link_distance = {}
        cchannel_delay = {}

        for name, entity in timeline.entities.items():
            if isinstance(entity, sq.ClassicalChannel) and 'BSM' not in entity.name:
                link_distance[name] = entity.distance
                cchannel_delay[name] = entity.delay
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
        cls.cchannel_delay = cchannel_delay

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
                    self.link_distance is not None,
                    self.cchannel_delay is not None])

    def _entanglement_succeed(self):
        """"""
        sq.log.logger.info(f'{self.owner.name} successful entanglement of memory {self.memory}')
        self.memory.entangled_memory['node_id'] = self.remote_node_name
        self.memory.entangled_memory['memo_id'] = self.remote_memo_id
        self.memory.fidelity = self.fidelity
        self.update_resource_manager(self.memory, sq.MemoryInfo.ENTANGLED)

    def emit_event(self):
        """Not needed for the surrogate protocol
        """
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

    def get_cchannel_name(self, node_name: str, other_node_name: str) -> str:
        """Get the classical channel name between the given node and other node.

        Args:
            node_name (str): The name of the node.
            other_node_name (str): The name of the other node.

        Returns:
            str: The name of the classical channel.
        """
        return f"cc-{node_name}-{other_node_name}"

    def sample_from_surrogate(self, path: list[str]) -> tuple[int, float]:
        """Sample the time and fidelity from the surrogate model.

        Args:
            path (list[str]): The path of the entanglement.

        Returns:
            tuple[int, float]: Sampled integer time in simulation units (ps) and fidelity.
        """
        if self.surrogate is None:
            raise ValueError("Surrogate sampler is not set.")
        if len(path) < 2:
            raise ValueError("The path must contain at least two nodes.")
        config = {name: getattr(self, name) for name in GLOBAL_FEATURE_NAMES}
        config["link_distances"] = [self.link_distance[self.get_cchannel_name(node, other)] for node, other in zip(path, path[1:])]
        seed = int(self.owner.get_generator().integers(0, 2**31))
        time, fidelity = self.surrogate(config, 1, seed)[0]
        return int(time * MILLISECOND), float(fidelity)

    def start(self):
        """"""
        if self.surrogate_parameater_is_set() is False:
            raise ValueError("Surrogate parameters are not set.")
        if self.primary: # only the protocol at primary node starts, protocol at all other nodes waits
            # calculate the time and fidelity from the surrogate
            sq.log.logger.info(f'{self.memory.name}: primary node infers the time and fidelity')
            path = self.rule.reservation.path
            time, self.fidelity = self.sample_from_surrogate(path)
            assert len(path) >= 2 and self.owner.name == path[0], f"Invalid path and primary node: {path}, {self.owner.name}"
            cchannel_name = self.get_cchannel_name(path[0], path[1])
            cchannel_delay = self.cchannel_delay.get(cchannel_name)
            entanglement_succeed_time = self.owner.timeline.now() - 2*cchannel_delay + time
            entanglement_succeed_time = self.validate_entanglement_succeed_time(entanglement_succeed_time, path)
            process = sq.Process(self, "entanglement_succeed", [self.fidelity])
            event = sq.Event(entanglement_succeed_time, process)
            self.owner.timeline.schedule(event)
            self.scheduled_events.append(event)
            # send the (time, fidelity) message to the responder
            self.remote_node_name = path[-1]
            message = RepeaterChainSurrogateMessage(sq.GenerationMsgType.NEGOTIATE, REPEATER_CHAIN_SURROGATE, 
                                                    self.rule.reservation, self.memory.name, 
                                                    time=entanglement_succeed_time, fidelity=self.fidelity)
            self.owner.send_message(self.remote_node_name, message)
        else:
            sq.log.logger.info(f'{self.memory.name}: non-primary node do nothing at start()')

    def validate_entanglement_succeed_time(self, entanglement_succeed_time: int, path: list[str]) -> int:
        """Return the entanglement succeed time adjusted for classical communication delay.

        Args:
            entanglement_succeed_time (int): The entanglement succeed time in simulation units (ps).
            path (list[str]): The path of the entanglement.

        Returns:
            int: The original or adjusted entanglement succeed time in ps.
        """
        now = self.owner.timeline.now()
        cc_channel_name = self.get_cchannel_name(path[0], path[-1])
        cc_channel_delay = self.cchannel_delay.get(cc_channel_name)
        if entanglement_succeed_time < now + 2 * cc_channel_delay:
            entanglement_succeed_time = now + 2 * cc_channel_delay + 1
        return entanglement_succeed_time

    def received_message(self, src: str, msg: RepeaterChainSurrogateMessage):
        """Handle received messages.

        Args:
            src (str): The source of the message.
            msg (RepeaterChainSurrogateMessage): The received message.
        """
        if msg.reservation == self.rule.reservation: # make sure talking to the correct protocol
            if self.surrogate_parameater_is_set() is False:
                raise ValueError("Surrogate parameters are not set.")
            if msg.msg_type == sq.GenerationMsgType.NEGOTIATE: # the responder receives the negotiation message
                sq.log.logger.info(f"{self.memory.name}: Handling NEGOTIATE message from {src}")
                entanglement_succeed_time = msg.time
                self.fidelity = msg.fidelity
                self.remote_node_name = src
                self.remote_memo_id = msg.memo_id
                process = sq.Process(self, "entanglement_succeed", [self.fidelity])
                event = sq.Event(entanglement_succeed_time, process)
                self.owner.timeline.schedule(event)
                self.scheduled_events.append(event)
                message = RepeaterChainSurrogateMessage(sq.GenerationMsgType.NEGOTIATE_ACK, REPEATER_CHAIN_SURROGATE, 
                                                        self.rule.reservation, self.memory.name)
                self.owner.send_message(src, message)

            elif msg.msg_type == sq.GenerationMsgType.NEGOTIATE_ACK:
                sq.log.logger.info(f"{self.memory.name}: Received NEGOTIATE_ACK from {src}")
                self.remote_memo_id = msg.memo_id
            
    def entanglement_succeed(self, fidelity: float):
        """Handle the entanglement succeed event.

        Args:
            fidelity (float): The fidelity of the entanglement.
        """
        sq.log.logger.info(f"{self.memory.name}: Entanglement succeed with fidelity {fidelity}")
        if not (self.primary or self.owner.name == self.rule.reservation.path[-1]): # first node or last node of the chain
            raise Exception("Entanglement succeed called on intermediate nodes.")
        quantum_manager = self.owner.timeline.quantum_manager
        self_key = self._qstate_key
        remote_memory: sq.Memory = self.owner.timeline.get_entity_by_name(self.remote_memo_id)
        remote_key = remote_memory.qstate_key
        keys = [self_key, remote_key]
        if self_key not in quantum_manager.states:
            in_fidelity = 1 - fidelity
            raw_epr_errors = [1/3, 1/3, 1/3]
            x_elem, y_elem, z_elem = (error * in_fidelity for error in raw_epr_errors)
            state = [fidelity, z_elem, x_elem, y_elem]
            quantum_manager.set(keys, state)
            # decoherence not done yet
        self._entanglement_succeed()


@sq.EntanglementGenerationB.register(REPEATER_CHAIN_SURROGATE)
class RepeaterChainBSurrogate(sq.EntanglementGenerationB):
    pass
