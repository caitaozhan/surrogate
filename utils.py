"""Some utility functions for the simulation.
"""

from pathlib import Path

import h5py
import numpy as np
import sequence as sq
from networkx import Graph, path_graph, set_edge_attributes


def set_type_and_formalism(generation_type: str, formalism: str) -> None:
    """Configure the global generation type and quantum state formalism.

    Args:
        generation_type (str): Entanglement generation type.
        formalism (str): Quantum state formalism for purification, swapping, and the quantum manager.
    """
    for protocol in (sq.EntanglementGenerationA, sq.EntanglementGenerationB):
        protocol.set_global_type(generation_type)

    for protocol in (sq.PurificationProtocol, sq.EntanglementSwappingA, sq.EntanglementSwappingB):
        protocol.set_formalism(formalism)

    sq.QuantumManager.set_global_manager_formalism(formalism)


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

    def save_chain_data(self, parameters: dict[str, float | int | list[float]], time_to_serve: list[float], fidelity: list[float]):
        """Save repeater-chain parameters, time-to-serve samples, and fidelity samples.

        Args:
            parameters (dict[str, float | int | list[float]]): Repeater-chain configuration values.
            time_to_serve (list[float]): Time-to-serve samples in milliseconds.
            fidelity (list[float]): Fidelity paired with each time-to-serve sample.
        """
        reference_state_dtype = h5py.enum_dtype({"bell_pair": 0}, basetype="i1")
        log_format_dtype = h5py.enum_dtype({"fidelity": 0, "state_vector": 1, "werner_vector": 2, "density_matrix": 3,
                                             "pauli_observables": 4}, basetype="i1")
        simulation_mode_dtype = h5py.enum_dtype({"repeated_single_shot": 0, "stateful": 1}, basetype="i1")
        time = np.asarray(time_to_serve, dtype=np.float32)
        state = np.asarray(fidelity, dtype=np.float32)
        if time.shape != state.shape:
            raise ValueError("time-to-serve and fidelity must contain the same number of samples")
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
                if isinstance(value, list):
                    user_optional.attrs[name] = np.asarray(value, dtype=np.float32)
                else:
                    user_optional.attrs[name] = np.int32(value) if isinstance(value, int) else np.float32(value)

            simulation_log = qnet.create_group("simulation_log")
            simulation_log.create_dataset("time", data=time, compression="gzip", shuffle=True)
            simulation_log.create_dataset("state", data=state, compression="gzip", shuffle=True)


def build_linear_variable_length(lengths: list[float], attenuation: float = 0.0002) -> Graph:
    """Create a linear graph with a different length for each edge.

    Args:
        lengths (list[float]): Edge lengths in kilometers.
        attenuation (float): Attenuation of the edges in dB/m.

    Returns:
        Graph: Linear graph containing one more node than edge lengths.
    """
    if not lengths:
        raise ValueError("lengths must contain at least one edge length")
    if any(length <= 0 for length in lengths):
        raise ValueError("all edge lengths must be positive")

    graph = path_graph(len(lengths) + 1)
    for node in graph.nodes:
        graph.nodes[node]["node_type"] = "processing"
    for index, length in enumerate(lengths):
        graph.edges[index, index + 1]["length"] = length
    set_edge_attributes(graph, attenuation, name="attenuation")
    return graph
