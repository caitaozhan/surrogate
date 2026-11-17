"""Generate simulation data for a linear quantum repeater chain."""

import argparse
from pathlib import Path
import sequence as sq
from sequence.utils.classical_delay import update_cchannel_delay
from sequence.utils.config_generator_cli import generate_config
from sequence.constants import MILLISECOND, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
from sequence.resource_management.reservation_rule_registry import EP_AWAIT, EP_REQUEST, ES_A, ES_B, ES_B_END
from request_app import RequestAppTimeToServe
from traffic import Traffic
from utils import HDF5, build_linear_variable_length, configure_simulation
from repeater_chain_surrogate import REPEATER_CHAIN_SURROGATE, RepeaterChainASurrogate



def generate_linear_config(args) -> dict:
    """
    Generate a configuration template for a linear quantum repeater chain.
    """
    
    # contains # 7, 2, 8, 6, 3
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

    # contains # 1, 4
    lengths = [distance / 1000 for distance in args.link_distances]
    nodes = len(lengths) + 1
    attenuation = args.attenuation
    cc_delay = 0
    memory_size = 5
    # output = f'linear_{nodes}.json'
    # directory = 'config'
    output = None
    directory = None
    stop_time = args.stop_time
    formalism = BELL_DIAGONAL_STATE_FORMALISM
    measurement_fidelity = 0.99
    gate_fidelity = 0.99
    g = build_linear_variable_length(lengths, attenuation)
    config, _ = generate_config(g, cc_delay, memory_size, output, directory, stop_time, formalism, template, measurement_fidelity, gate_fidelity)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # same arguments as the link
    parser.add_argument("-lds", "--link_distances", type=float, nargs="+", default=[5000.0, 5000.0],
                        help="1: distances in meters between adjacent nodes")
    parser.add_argument("-me", "--memory_efficiency", type=float, default=0.7, help="2: efficiency of the memory qubits")
    parser.add_argument("-de", "--detector_efficiency", type=float, default=0.95, help="3: efficiency of the detectors")
    parser.add_argument("-a", "--attenuation", type=float, default=0.0002, help="4: attenuation of the channel")
    parser.add_argument("-cdn", "--classical_delay_node", type=float, default=0.1, help="5: classical delay in ms for each node")
    # Additional arguments for the chain
    parser.add_argument("-sp", "--swapping_probability", type=float, default=0.9, help="6: probability of successful entanglement swapping")
    parser.add_argument("-if", "--initial_fidelity", type=float, default=0.95, help="7: initial fidelity of link entangled pairs")
    parser.add_argument("-ct", "--coherence_time", type=float, default=2.0, help="8: coherence time in seconds for each memory qubit")
    parser.add_argument("-cdh", "--classical_delay_hop", type=float, default=0.02, help="9: classical delay in ms for each intermediate hop")
    # simulation parameters
    parser.add_argument("-d", "--dir", type=Path, default=Path("chain"))
    parser.add_argument("-st", "--stop_time", type=float, default=2002, help="stop time in seconds")
    parser.add_argument("-el", "--event_log", action='store_true', help="enable event log")
    parser.add_argument("-sl", "--summary_log", action='store_true', help="enable summary log")
    parser.add_argument("-s", "--surrogate", action='store_true', help="enable surrogate mode")
    
    args = parser.parse_args()
    
    if args.surrogate:
        configure_simulation(REPEATER_CHAIN_SURROGATE, BELL_DIAGONAL_STATE_FORMALISM)
        diabled_rules = [EP_AWAIT, EP_REQUEST, ES_A, ES_B, ES_B_END]
    else:
        configure_simulation(SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM)
        diabled_rules = [EP_AWAIT, EP_REQUEST]
    
    config = generate_linear_config(args)
    topology = sq.RouterNetTopo(config)
    update_cchannel_delay(topology, args.classical_delay_node * 1000, args.classical_delay_hop * 1000) # ms to us
    tl = topology.tl

    # disable purification and swapping
    router: sq.QuantumRouter
    for router in topology.get_nodes_by_type(sq.RouterNetTopo.QUANTUM_ROUTER):
        for rule in diabled_rules:
            router.disable_rule(rule)
    # summary log
    number_of_nodes = len(args.link_distances) + 1
    link_distances = "-".join(f"{distance:g}" for distance in args.link_distances)
    file_name = (f"n={number_of_nodes}_s={args.surrogate}_lds={link_distances}_sp={args.swapping_probability:g}_"
                 f"if={args.initial_fidelity:g}_ct={args.coherence_time:g}_"
                 f"cdh={args.classical_delay_hop:g}_me={args.memory_efficiency:g}_de={args.detector_efficiency:g}_"
                 f"a={args.attenuation:g}_cdn={args.classical_delay_node:g}")
    log_filename = args.dir / "log" / f"{file_name}.log"
    h5_filename = args.dir / "h5" / f"{file_name}.h5"
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    # event log
    if args.event_log:
        sq.log.set_logger(__name__, tl, str(log_filename) + '.event')
        sq.log.set_logger_level('INFO')
        # modules = ['request_app', 'swapping_bds', 'single_heralded', 'resource_manager', 'single_heralded_surrogate']
        # modules = ['timeline', 'main_chain', 'repeater_chain_surrogate']
        modules = ['timeline', 'main_chain']
        for module in modules:
            sq.log.track_module(module)

    traffic = Traffic(f'linear_{number_of_nodes}', seed=0)
    entanglement_number = 1
    name_to_app = {}
    node_seed = 0
    for router in topology.get_nodes_by_type(sq.RouterNetTopo.QUANTUM_ROUTER):
        router.set_seed(node_seed)
        node_seed += 1
        app = RequestAppTimeToServe(router)
        app.entanglement_number = entanglement_number
        app.set_distributions(traffic.distributions[router.name])
        app.kickoff_the_first_request()
        name_to_app[router.name] = app

    for bsm_node in topology.get_nodes_by_type(sq.RouterNetTopo.BSM_NODE):
        bsm_node.set_seed(node_seed)
        node_seed += 1

    tl.init()
    if args.surrogate:
        RepeaterChainASurrogate.set_surrogate_parameter(tl, args.classical_delay_node, args.classical_delay_hop)
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

    parameters = {"number_of_nodes": number_of_nodes, "swapping_probability": args.swapping_probability,
                  "link_distances": args.link_distances, "memory_efficiency": args.memory_efficiency,
                  "detector_efficiency": args.detector_efficiency, "attenuation": args.attenuation,
                  "classical_delay_node": args.classical_delay_node, "classical_delay_hop": args.classical_delay_hop,
                  "initial_fidelity": args.initial_fidelity, "coherence_time": args.coherence_time}
    HDF5(h5_filename).save_chain_data(parameters, time_to_serve_list, fidelity_list)

    if args.summary_log:
        log_samples = list(zip(reservation_list, time_to_serve_list, fidelity_list))
        # if len(log_samples) > 20:
        #     log_samples = log_samples[:10] + log_samples[-10:]
        chain_log = open(log_filename, "w")
        for reservation, time_to_serve, fidelity in log_samples:
            chain_log.write(f"{reservation}, {time_to_serve}, {fidelity}\n")
        chain_log.write(f"Average time to serve: {sum(time_to_serve_list) / len(time_to_serve_list):.3f} ms\n")
        chain_log.write(f"Average fidelity: {sum(fidelity_list) / len(fidelity_list):.6f}\n")
        chain_log.close()
    sq.log.logger.info(f"Average time to serve: {sum(time_to_serve_list) / len(time_to_serve_list):.3f} ms. Total events simulated: {tl.run_counter}")
