"""Generate simulation data for a linear quantum repeater chain."""

import argparse
from pathlib import Path

from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.purification import BBPSSWProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.quantum_manager import QuantumManager
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.utils.config_generator_cli import generate_config
from sequence.utils.graphs import build_linear
from sequence.constants import MILLISECOND, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
from request_app import RequestAppTimeToServe
from traffic import Traffic
from utils import HDF5, update_cchannel_delay


EntanglementGenerationA.set_global_type(SINGLE_HERALDED)
EntanglementGenerationB.set_global_type(SINGLE_HERALDED)
BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)


def generate_linear_config(args) -> tuple[dict, dict]:
    """
    Generate a configuration template for a linear quantum repeater chain.
    """
    
    # contains # 8, 2, 9, 6, 3
    template = {
        "router_template": {
            "routing": "routing_distributed",
            "MemoryArray": {
                "fidelity": args.initial_fidelity,
                "efficiency": args.memory_efficiency,
                "coherence_time": args.coherence_time,
                "decoherence_errors": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
            },
            "EntanglementSwapping": {
                "swapping_success_prob": args.swapping_probability,
            }
        },
        "bsm_template": {
            "encoding_type": "single_heralded",
            "SingleHeraldedBSM": {
                "detectors" :[
                    {"efficiency": args.detector_efficiency}, 
                    {"efficiency": args.detector_efficiency}
                ]
            }
        }
    }

    # contains # 6, 1, 4
    nodes = args.number_of_nodes
    length = args.link_distance / 1000
    attenuation = args.attenuation
    cc_delay = 0
    memory_size = 5
    output = f'linear_{args.number_of_nodes}.json'
    directory = 'config'
    stop_time = args.stop_time
    formalism = BELL_DIAGONAL_STATE_FORMALISM
    measurement_fidelity = 0.99
    gate_fidelity = 0.99
    g = build_linear(nodes, length, attenuation)
    config, _ = generate_config(g, cc_delay, memory_size, output, directory, stop_time, formalism, template, measurement_fidelity, gate_fidelity)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # same arguments as the link
    parser.add_argument("-ld", "--link_distance", type=float, default=5000.0, help="1: distance in meters between two adjacent nodes")
    parser.add_argument("-me", "--memory_efficiency", type=float, default=0.7, help="2: efficiency of the memory qubits")
    parser.add_argument("-de", "--detector_efficiency", type=float, default=0.95, help="3: efficiency of the detectors")
    parser.add_argument("-a", "--attenuation", type=float, default=0.0002, help="4: attenuation of the channel")
    parser.add_argument("-cdn", "--classical_delay_node", type=float, default=0.1, help="5: classical delay in ms for each node")
    # Additional arguments for the chain
    parser.add_argument("-n", "--number_of_nodes", type=int, default=3, help="6: number of nodes in the chain")
    parser.add_argument("-sp", "--swapping_probability", type=float, default=0.9, help="7: probability of successful entanglement swapping")
    parser.add_argument("-if", "--initial_fidelity", type=float, default=0.95, help="8: initial fidelity of link entangled pairs")
    parser.add_argument("-ct", "--coherence_time", type=float, default=2.0, help="9: coherence time in seconds for each memory qubit")
    parser.add_argument("-cdh", "--classical_delay_hop", type=float, default=0.02, help="10: classical delay in ms for each intermediate hop")
    parser.add_argument("-d", "--dir", type=Path, default=Path("chain"))
    parser.add_argument("-st", "--stop_time", type=float, default=2002, help="stop time in seconds")
    args = parser.parse_args()

    config = generate_linear_config(args)

    topology = RouterNetTopo(config)
    # contains # 5, 10
    update_cchannel_delay(topology, config, args.classical_delay_node, args.classical_delay_hop)
    tl = topology.tl

    file_name = (f"n_{args.number_of_nodes}_sp={args.swapping_probability:g}_if={args.initial_fidelity:g}_ct={args.coherence_time:g}_cdh={args.classical_delay_hop:g}_"
                 f"ld={args.link_distance:g}_me={args.memory_efficiency:g}_de={args.detector_efficiency:g}_a={args.attenuation:g}_cdn={args.classical_delay_node:g}")
    log_filename = args.dir / "log" / f"{file_name}.log"
    h5_filename = args.dir / "h5" / f"{file_name}.h5"
    log_filename.parent.mkdir(parents=True, exist_ok=True)

    traffic = Traffic(f'linear_{args.number_of_nodes}', seed=0)
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

    tl.init()
    tl.run()

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
                  "classical_delay_node": args.classical_delay_node, "classical_delay_hop": args.classical_delay_hop,
                  "initial_fidelity": args.initial_fidelity, "coherence_time": args.coherence_time}
    HDF5(h5_filename).save_chain_data(parameters, time_to_serve_list, fidelity_list)

    # log_samples = list(zip(reservation_list, time_to_serve_list, fidelity_list))
    # if len(log_samples) > 20:
    #     log_samples = log_samples[:10] + log_samples[-10:]

    chain_log = open(log_filename, "w")
    # for reservation, time_to_serve, fidelity in log_samples:
    #     chain_log.write(f"{reservation}, {time_to_serve}, {fidelity}\n")
    chain_log.write(f"Average time to serve: {sum(time_to_serve_list) / len(time_to_serve_list):.3f} ms\n")
    chain_log.write(f"Average fidelity: {sum(fidelity_list) / len(fidelity_list):.6f}\n")
    chain_log.close()
