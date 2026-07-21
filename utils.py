"""Some utility functions for the simulation.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from networkx import Graph, single_source_dijkstra
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import SPEED_OF_LIGHT, MILLISECOND


class HDF5:
    """Save simulation output in the qnet HDF5 format."""

    def __init__(self, filename: Path):
        self.filename = filename

    def save_link_data(self, parameters: dict[str, float], time_to_serve: list[float]):
        """Save link parameters and time-to-serve samples."""
        reference_state_dtype = h5py.enum_dtype({"bell_pair": 0}, basetype="i1")
        log_format_dtype = h5py.enum_dtype({"fidelity": 0, "state_vector": 1, "werner_vector": 2, "density_matrix": 3,
                                             "pauli_observables": 4}, basetype="i1")
        simulation_mode_dtype = h5py.enum_dtype({"repeated_single_shot": 0, "stateful": 1}, basetype="i1")
        time = np.asarray(time_to_serve, dtype=np.float32)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.filename, "w") as h5_file:
            qnet = h5_file.create_group("qnet")
            qnet.attrs["format_version"] = np.uint64(1)
            qnet.attrs["format_version_minor"] = np.uint64(1)
            qnet.attrs.create("log_format", np.array(0, dtype=log_format_dtype), dtype=log_format_dtype)
            qnet.attrs.create("reference_state", np.array(0, dtype=reference_state_dtype), dtype=reference_state_dtype)
            qnet.attrs.create("simulation_mode", np.array(1, dtype=simulation_mode_dtype), dtype=simulation_mode_dtype)

            metadata = qnet.create_group("metadata")
            sequence_metadata = metadata.create_group("SeQUeNCe")
            sequence_metadata.attrs["SeQUeNCe_format_version"] = np.uint64(1)
            user_optional = metadata.create_group("user_optional")
            for name, value in parameters.items():
                user_optional.attrs[name] = np.float32(value)

            simulation_log = qnet.create_group("simulation_log")
            simulation_log.create_dataset("time", data=time, compression="gzip", shuffle=True)


def classical_delay(distance: float, hop_count: int, classical_delay_node: float, classical_delay_hop: float) -> int:
    """Model the classical delay as a function of distance and hop count

    Args:
        distance (float): the distance between source and destination in km
        hop_count (int): the number of hops/nodes between source and destination
        classical_delay_node (float): delay at the destination node in milliseconds
        classical_delay_hop (float): delay for each intermediate hop in milliseconds
    
    Returns:
        int: the delay in picoseconds
    """
    return int(distance / SPEED_OF_LIGHT + (hop_count * classical_delay_hop + classical_delay_node) * MILLISECOND)


def update_cchannel_delay(topo: RouterNetTopo, config_file_name: str, classical_delay_node: float, classical_delay_hop: float):
    """Update the delay of classical channels based on the network topology and classical delay model.

    Args:
        topo (RouterNetTopo): the topology of the network
        config_file_name (str): the name of configuration file
        classical_delay_node (float): delay at the destination node in milliseconds
        classical_delay_hop (float): delay for each intermediate hop in milliseconds
    """
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    
    graph = Graph()
    for node in config[topo.ALL_NODE]:
        if node[topo.TYPE] == topo.QUANTUM_ROUTER:
            graph.add_node(node[topo.NAME])
    
    all_paths = {}  # (src, dst) -> (length: float, hop: int, path: tuple)
    costs = {}

    for qc in topo.qchannels:
        router = qc.sender.name
        bsm = qc.receiver

        # update all_paths
        router, bsm = qc.sender.name, qc.receiver
        all_paths[(router, bsm)] = (qc.distance, 0, (router, bsm))
        all_paths[(bsm, router)] = (qc.distance, 0, (bsm, router))

        if bsm not in costs:
            costs[bsm] = [router, qc.distance]
        else:
            costs[bsm] = [router] + costs[bsm]
            costs[bsm][-1] += qc.distance
        
    graph.add_weighted_edges_from(costs.values())
    for src in topo.nodes[topo.QUANTUM_ROUTER]:
        for dst_name in graph.nodes:
            if src.name == dst_name:
                continue

            if dst_name > src.name:
                length, path = single_source_dijkstra(graph, src.name, dst_name)
            else:
                length, path = single_source_dijkstra(graph, dst_name, src.name)
                path = path[::-1]
            # update all_paths
            hop_count = len(path) - 2
            all_paths[(src.name, dst_name)] = (length, hop_count, tuple(path))

    for cc in topo.cchannels:
        src = cc.sender.name
        dst = cc.receiver
        length, hop_count, path = all_paths[(src, dst)]
        cc.delay = classical_delay(length, hop_count, classical_delay_node, classical_delay_hop)
        cc.distance = length   # not important
        # print(f'{path}: {cc.delay/1e6}us')
