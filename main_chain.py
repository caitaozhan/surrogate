"""Generate simulation data for a linear quantum repeater chain."""

import argparse
import json
from pathlib import Path

from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.purification import BBPSSWProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.quantum_manager import QuantumManager
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import SECOND, MILLISECOND, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
from request_app import RequestAppTimeToServe
from traffic import Traffic
from utils import HDF5, update_cchannel_delay


EntanglementGenerationA.set_global_type(SINGLE_HERALDED)
EntanglementGenerationB.set_global_type(SINGLE_HERALDED)
BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)


def build_linear_config(base_config: dict, number_of_nodes: int, link_distance: float, attenuation: float) -> dict:
    """Build a linear topology with one midpoint BSM node per adjacent router pair.

    Args:
        base_config (dict): Configuration providing component templates and global settings.
        number_of_nodes (int): Number of quantum routers in the chain.
        link_distance (float): Distance between adjacent quantum routers in meters.
        attenuation (float): Quantum-channel attenuation.

    Returns:
        dict: Complete RouterNetTopo configuration.
    """
    config = {"templates": base_config["templates"], "nodes": [], "qchannels": [], "cchannels": [],
              "stop_time": base_config["stop_time"], "formalism": base_config["formalism"]}

    for index in range(number_of_nodes):
        config["nodes"].append({"name": f"router_{index}", "type": "QuantumRouter", "seed": index, "memo_size": 5,
                                "template": "router_template", "gate_fidelity": 0.99, "measurement_fidelity": 0.99})

    for index in range(number_of_nodes - 1):
        left_router = f"router_{index}"
        right_router = f"router_{index + 1}"
        bsm_node = f"BSM_{left_router}_{right_router}"
        config["nodes"].append({"name": bsm_node, "type": "BSMNode", "seed": index, "template": "bsm_template"})
        for router in (left_router, right_router):
            config["qchannels"].append({"source": router, "destination": bsm_node, "distance": link_distance / 2,
                                        "attenuation": attenuation})
            config["cchannels"].append({"source": router, "destination": bsm_node, "distance": link_distance / 2, "delay": 0})
            config["cchannels"].append({"source": bsm_node, "destination": router, "distance": link_distance / 2, "delay": 0})

    for source_index in range(number_of_nodes):
        for destination_index in range(number_of_nodes):
            if source_index == destination_index:
                continue
            config["cchannels"].append({"source": f"router_{source_index}", "destination": f"router_{destination_index}", "delay": 0})
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_of_nodes", type=int, default=3)
    parser.add_argument("-sp", "--swapping_probability", type=float, default=0.9)
    parser.add_argument("-ld", "--link_distance", type=float, default=5000.0)
    parser.add_argument("-me", "--memory_efficiency", type=float, default=0.7)
    parser.add_argument("-de", "--detector_efficiency", type=float, default=0.96)
    parser.add_argument("-a", "--attenuation", type=float, default=0.0002)
    parser.add_argument("-cdn", "--classical_delay_node", type=float, default=0.1, help="classical delay in ms for each node")
    parser.add_argument("-cdh", "--classical_delay_hop", type=float, default=0.02, help="classical delay in ms for each intermediate hop")
    parser.add_argument("-d", "--dir", type=Path, default=Path("chain"))
    parser.add_argument("-st", "--stop_time", type=float, default=5002 * SECOND, help="stop time in seconds")
    args = parser.parse_args()

    if args.number_of_nodes < 2:
        raise ValueError("number_of_nodes must be at least 2")
    if not 0 <= args.swapping_probability <= 1:
        raise ValueError("swapping_probability must be between 0 and 1")

    config_file = f"linear_{args.number_of_nodes}"
    with open("config/linear_2.json", "r") as config_file_object:
        base_config = json.load(config_file_object)
    base_config["stop_time"] = args.stop_time
    router_template = base_config["templates"]["router_template"]
    router_template["MemoryArray"]["efficiency"] = args.memory_efficiency
    router_template["EntanglementSwapping"]["swapping_success_prob"] = args.swapping_probability

    detectors = base_config["templates"]["bsm_template"]["SingleHeraldedBSM"]["detectors"]
    for detector in detectors:
        detector["efficiency"] = args.detector_efficiency

    config = build_linear_config(base_config, args.number_of_nodes, args.link_distance, args.attenuation)
    topology = RouterNetTopo(config)
    update_cchannel_delay(topology, config, args.classical_delay_node, args.classical_delay_hop)
    timeline = topology.tl

    file_stem = (f"{config_file}_sp={args.swapping_probability:g}_ld={args.link_distance:g}_me={args.memory_efficiency:g}_"
                 f"de={args.detector_efficiency:g}_a={args.attenuation:g}_cdn={args.classical_delay_node:g}_cdh={args.classical_delay_hop:g}")
    log_filename = args.dir / "log" / f"{file_stem}.log"
    h5_filename = args.dir / "h5" / f"{file_stem}.h5"
    log_filename.parent.mkdir(parents=True, exist_ok=True)

    traffic = Traffic(config_file, seed=0)
    entanglement_number = 1
    name_to_app = {}
    node_seed = 0
    for router in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        router.set_seed(node_seed)
        node_seed += 1
        app = RequestAppTimeToServe(router)
        app.entanglement_number = entanglement_number
        app.set_distributions(traffic.distributions[router.name])
        app.kickoff_the_first_request()
        name_to_app[router.name] = app

    for bsm_node in topology.get_nodes_by_type(RouterNetTopo.BSM_NODE):
        bsm_node.set_seed(node_seed)
        node_seed += 1

    timeline.init()
    timeline.run()

    samples = {}
    for app in name_to_app.values():
        for reservation, time_to_serve in app.time_to_serve.items():
            fidelities = app.entanglement_fidelities[reservation]
            if fidelities:
                samples[reservation] = (time_to_serve / MILLISECOND, fidelities[-1])
    if not samples:
        raise RuntimeError("simulation completed without producing any entanglement samples")

    reservation_list = []
    time_to_serve_list = []
    fidelity_list = []
    for reservation, (time_to_serve, fidelity) in sorted(samples.items()):
        reservation_list.append(reservation)
        time_to_serve_list.append(time_to_serve)
        fidelity_list.append(fidelity)

    parameters = {"number_of_nodes": args.number_of_nodes, "swapping_probability": args.swapping_probability,
                  "link_distance": args.link_distance, "memory_efficiency": args.memory_efficiency,
                  "detector_efficiency": args.detector_efficiency, "attenuation": args.attenuation,
                  "classical_delay_node": args.classical_delay_node, "classical_delay_hop": args.classical_delay_hop}
    HDF5(h5_filename).save_chain_data(parameters, time_to_serve_list, fidelity_list)

    log_samples = list(zip(reservation_list, time_to_serve_list, fidelity_list))
    if len(log_samples) > 20:
        log_samples = log_samples[:10] + log_samples[-10:]

    chain_log = open(log_filename, "w")
    for reservation, time_to_serve, fidelity in log_samples:
        chain_log.write(f"{reservation}, {time_to_serve}, {fidelity}\n")
    chain_log.write(f"Average time to serve: {sum(time_to_serve_list) / len(time_to_serve_list):.3f} ms\n")
    chain_log.write(f"Average fidelity: {sum(fidelity_list) / len(fidelity_list):.6f}\n")
    chain_log.close()
