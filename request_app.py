
from __future__ import annotations
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter


from sequence.app.request_app import RequestApp
from sequence.network_management.reservation import Reservation
import sequence.utils.log as log
from sequence.constants import SECOND, MILLISECOND
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.components.memory import Memory


class RequestAppTimeToServe(RequestApp):
    '''for the time-to-serve (latency) metric

    The RequestApp can only handle one request (for a node) at a time, it cannot handle multiple requests at the same time.
    It can handle multiple requests one by one (no timing overlap between consequtive requests)

    Args:
        node (QuantumRouter) the node this app is attached to
        entanglement_timestamps (defaultdict(list)): reservation -> list of timestamps when entanglement is generated for the reservation
        time_to_serve (defaultdict(float)): reservation -> time to serve for the reservation
        entanglement_fidelities (defaultdict(list)): reservation -> list of fidelities of the entangled pairs generated for the reservation
        distributions (dict): dst -> distribution function, a function that returns the inter-arrival time (in seconds) for the next request to the dst
        request_duration (int): the duration of each request in milliseconds
        memo_size (int): the memory size of each request
        fidelity (float): the fidelity requirement of each request
        entanglement_number (int): the number of entanglement pairs requested for each request
    '''

    def __init__(self, node: QuantumRouter):
        super().__init__(node)
        self.entanglement_timestamps = defaultdict(list)  # reservation: list[float]
        self.time_to_serve = defaultdict(float)           # reservation: float
        self.entanglement_fidelities = defaultdict(list)  # reservation: list[float]
        self.traffic_distributions: dict = {}
        self.outages: list = []
        self.request_duration = 500 * MILLISECOND
        self.memo_size = 1
        self.fidelity = 0.1
        self.entanglement_number = 1
        self.request_success_counter = 0
        self.request_failure_counter = 0
        self.latest_request_event = None
        self.reservation_to_scheduled_event = {}  # reservation -> event for removing memo reservation map at the end time of the reservation
        self.pathfinding_reservation_duration = 10 * MILLISECOND

    def set_outages(self, outages: list):
        '''set the outage distribution for this app

        Args:
            outages: list of (time, status), where time is the time of the outage event and status is either 'up' or 'down'
        '''
        self.outages = outages

    def set_router_down(self):
        '''set the router down
           when the router is down, the app will not send any request and will not schedule the next request
           It can not send and receive any message.
        '''
        log.logger.info(f"{self.node.name} goes DOWN")
        self.node.set_down(True)
        if self.latest_request_event is not None:
            self.node.timeline.remove_event(self.latest_request_event)
            self.latest_request_event = None

    def set_router_up(self):
        '''set the router up
           when the router is up, the app will send request and schedule the next request based on the distribution
           It is capable of sending and receiving messages.
        '''
        log.logger.info(f"{self.node.name} goes UP")
        self.node.set_down(False)
        self.kickoff_the_first_request()

    def schedule_outages(self):
        '''schedule the outages based on the outage distribution
        '''
        for time, status in self.outages:
            time = time * SECOND
            match status:
                case 'down':
                    event = Event(time, Process(self, "set_router_down", []), priority=0)
                case 'up':
                    event = Event(time, Process(self, "set_router_up", []), priority=0)
                case _:
                    raise ValueError(f"Unsupported status: {status}")
            self.node.timeline.schedule(event)

    def get_reservation_result(self, reservation: Reservation, result: bool) -> None:
        """Method to receive reservation result from network manager. 

        The initiator will call this method once received a response from the responder.

        Args:
            reservation (Reservation): reservation that has been completed.
            result (bool): result of the request (approved/rejected).

        Side Effects:
            May schedule a start/retry event based on reservation result.
        """
        super().get_reservation_result(reservation, result)
        if result:
            self.request_success_counter += 1
        else:
            self.request_failure_counter += 1

    def set_distributions(self, distributions: dict):
        '''set the traffic distribution for this app

        Args:
            distributions: dict of dst -> distribution function, where the distribution function is a function that returns the inter-arrival time (delta_t) for the next request to the dst
        '''
        self.traffic_distributions = distributions

    def kickoff_the_first_request(self):
        '''kick off the first request to each destination based on the distribution
        '''
        routing_duration = 2 * SECOND
        for dst, _ in self.traffic_distributions.items():
            event_t = self.node.timeline.now() + routing_duration
            start_t = event_t + self.pathfinding_reservation_duration # for path finding and reservation setup
            end_t = start_t + self.request_duration
            process = Process(self, "send_request_schedule_next", [dst, start_t, end_t, self.memo_size, self.fidelity, self.entanglement_number, 0])
            event = Event(event_t, process)
            self.node.timeline.schedule(event)
            self.latest_request_event = event

    def send_request_schedule_next(self, responder: str, start_t: int, end_t: int, memo_size: int, fidelity: float, entanglement_number: int = 1, id: int = 0):
        '''send the request and schedule the next request based on the distribution

        Args:
            responder: the destination of the request
            start_t: the start time of the request
            end_t: the end time of the request
            memo_size: the memory size of the request
            fidelity: the fidelity requirement of the request
            entanglement_number: the number of entanglement pairs requested
            id: the id of the request
        '''
        # send the request
        log.logger.debug(f'{self.node.name} sends request to {responder} with start_t={start_t:,}, end_t={end_t:,}, memo_size={memo_size}, fidelity={fidelity}, entanglement_number={entanglement_number}, id={id}')
        self.start(responder, start_t, end_t, memo_size, fidelity, entanglement_number, id)
        # schedule the next request
        distribution_func = self.traffic_distributions[responder]
        delta_t = distribution_func()
        event_t = self.node.timeline.now() + int(delta_t * SECOND)
        start_t = event_t + self.pathfinding_reservation_duration # for path finding and reservation setup
        end_t = start_t + self.request_duration
        process = Process(self, "send_request_schedule_next", [responder, start_t, end_t, memo_size, fidelity, entanglement_number, id+1])
        event = Event(event_t, process)
        self.node.timeline.schedule(event)

    def start(self, responder: str, start_t: int, end_t: int, memo_size: int, fidelity: float, entanglement_number: int = 1, id: int = 0):
        """Method to start the application.

            This method will use arguments to create a request and send to the network.

        Side Effects:
            Will create request for network manager on node.
        """
        assert 0 < fidelity <= 1
        assert 0 <= start_t <= end_t
        assert 0 < memo_size
        self.responder = responder
        self.start_t = start_t
        self.end_t = end_t
        self.memo_size = memo_size
        self.fidelity = fidelity
        self.entanglement_number = entanglement_number
        self.id = id

        self.node.reserve_net_resource(responder, start_t, end_t, memo_size, fidelity, entanglement_number, id)

    def get_memory(self, info: "MemoryInfo") -> None:
        """Method to receive entangled memories.

        Will check if the received memory is qualified.
        If it's a qualified memory, the application sets memory to RAW state
        and release back to resource manager.
        The counter of entanglement memories, 'memory_counter', is added.
        Otherwise, the application does not modify the state of memory and
        release back to the resource manager.

        Args:
            info (MemoryInfo): info on the qualified entangled memory.
        """

        if info.state != "ENTANGLED":
            return

        if info.index in self.memo_to_reservation:
            reservation = self.memo_to_reservation[info.index]
            if info.remote_node == reservation.initiator:  # the responder
                if info.fidelity >= reservation.fidelity:
                    self.entanglement_timestamps[reservation].append(self.node.timeline.now())
                    self.entanglement_fidelities[reservation].append(round(info.fidelity, 6))
                    self.clear_remote_memory_in_quantum_manager(info.memory)
                    self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
                    entanglement_number = len(self.entanglement_timestamps[reservation])
                    if entanglement_number == reservation.entanglement_number:
                        self.time_to_serve[reservation] = self.node.timeline.now() - reservation.start_time
                        self.node.resource_manager.expire_rules_by_reservation(reservation)
                        self.node.network_manager.remove_reservation_from_timecards(reservation)
                        self.remove_memo_reservation_map(info.index)
                        self.reservation_to_scheduled_event[reservation].set_invalid()
                else:
                    log.logger.info(f'Memory={info}, does not meet the fidelity threshold, {reservation}')

            elif info.remote_node == reservation.responder:  # the initiator
                if info.fidelity >= reservation.fidelity:
                    self.entanglement_timestamps[reservation].append(self.node.timeline.now())
                    self.entanglement_fidelities[reservation].append(round(info.fidelity, 6))
                    entanglement_number = len(self.entanglement_timestamps[reservation])
                    self.clear_remote_memory_in_quantum_manager(info.memory)
                    self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)

                    if entanglement_number == reservation.entanglement_number:
                        self.time_to_serve[reservation] = self.node.timeline.now() - reservation.start_time
                        log.logger.info(f"{self.node.name}, request served! reservation identity={reservation.identity}, "
                                        f"time to serve: {self.time_to_serve[reservation]/MILLISECOND:.3f} ms, "
                                        f"fidelities:{self.entanglement_fidelities[reservation]}")
                        self.node.resource_manager.expire_rules_by_reservation(reservation)
                        self.node.network_manager.remove_reservation_from_timecards(reservation)
                        self.remove_memo_reservation_map(info.index)
                        self.reservation_to_scheduled_event[reservation].set_invalid()
                        self.send_expire_rules_message(reservation)
                else:
                    log.logger.info(f'{self.node.name}, Memory={info} has not meet the threshold, reservation identity={reservation.identity}')

    def clear_remote_memory_in_quantum_manager(self, memory: Memory) -> None:
        '''clear the remote memory in quantum manager after the memory is used for a request.
           
        NOTE: only needed for the BDS formalism. When doing memory.decohere(), 
              because it will 'revive' the remote memory qstate_key in quantum manager, which should be removed to avoid confusion.

        Args:
            remote_memory_name: the name of the remote memory to be cleared
        '''
        remote_memory_name = memory.entangled_memory['memo_id']
        remote_memory = self.node.timeline.get_entity_by_name(remote_memory_name)
        if remote_memory.entangled_memory['memo_id'] is None:
            if remote_memory.qstate_key in self.node.timeline.quantum_manager.states: # the remote memory's qstate_key will appear in quantum manager states during memory.bds_decohere() if the swapping node is not exactly in the middle
                self.node.timeline.quantum_manager.states.pop(remote_memory.qstate_key)
                log.logger.debug(f'{self.node.name} cleared remote memory {remote_memory_name} in quantum manager')

    def get_time_stamps(self) -> list:
        '''get the entangled time stamps (for the "first" reservations)
        '''
        for reservation, entangled_timestamps in self.entanglement_timestamps.items():
            return entangled_timestamps   

    def get_time_to_service(self) -> list:
        '''compute the time to service (for the "first" reservations)
        '''
        time_to_sevice = []
        for reservation, entangled_timestamps in self.entanglement_timestamps.items():
            assert len(entangled_timestamps) > 0
            time_to_sevice.append(entangled_timestamps[0] - reservation.start_time)
            for i in range(1, len(entangled_timestamps)):
                time_to_sevice.append(entangled_timestamps[i] - entangled_timestamps[i-1])
            break
        return time_to_sevice

    def send_expire_rules_message(self, reservation: Reservation):
        '''send the expire rule message to nodes other than the initiator and responder

        Args:
            reservation: the rules to expires is generated for this reservation
        '''
        path = reservation.path
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                node = path[i]
                self.node.resource_manager.expire_remote_rules(node, reservation)

    def get_request_to_throughput(self) -> dict:
        '''each request maps to a reservation
        Return:
            dict[Reservation, float]
        '''
        request_to_throughput = {}
        for reservation, entanglement_timestamps in self.entanglement_timestamps.items():
            time_elapse = (reservation.end_time - reservation.start_time) / SECOND
            request_to_throughput[reservation] = len(entanglement_timestamps) / time_elapse
        return request_to_throughput

    def schedule_reservation(self, reservation: Reservation) -> None:
        """Calling the `add_memo_reservation_map` and `remove_memo_reservation_map` methods at the 
           reservation's start_time and end_time for all timecards (memory) involved in the reservation. 
           
           Called by the initiator and the responder when reservation is approved.

        Args:
            reservation (Reservation): reservation to schedule
        """
        if reservation.initiator == self.node.name:
            self.path = reservation.path

        for card in self.node.network_manager.get_timecards():
            if reservation in card.reservations:
                process = Process(self, "add_memo_reservation_map", [card.memory_index, reservation])
                event = Event(reservation.start_time, process)
                self.node.timeline.schedule(event)
                process = Process(self, "remove_memo_reservation_map", [card.memory_index])
                event = Event(reservation.end_time, process)
                self.node.timeline.schedule(event)
                self.reservation_to_scheduled_event[reservation] = event

