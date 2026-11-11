'''
the main for generating data for a link
'''
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sequence as sq
from sequence.utils.classical_delay import update_cchannel_delay
from sequence.constants import SECOND, MILLISECOND, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
from sequence.resource_management.reservation_rule_registry import EP_AWAIT, EP_REQUEST, ES_A, ES_B, ES_B_END
import sequence.utils.log as log
from request_app import RequestAppTimeToServe
from traffic import Traffic
from utils import HDF5, configure_simulation
from single_heralded_surrogate import SINGLE_HERALDED_SURROGATE, SingleHeraldedASurrogate

configure_simulation(SINGLE_HERALDED_SURROGATE, BELL_DIAGONAL_STATE_FORMALISM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ld", "--link_distance", type=float, default=5000.0)
    parser.add_argument("-me", "--memory_efficiency", type=float, default=0.7)
    parser.add_argument("-de", "--detector_efficiency", type=float, default=0.96)
    parser.add_argument("-a", "--attenuation", type=float, default=0.0002)
    parser.add_argument("-cdn", "--classical_delay_node", type=float, default=0.1, help="classical delay in ms for each node")
    parser.add_argument("-d", "--dir", type=Path, default=Path("link"))
    parser.add_argument("-st", "--stop_time", type=float, default=3 * SECOND, help="stop time in seconds")

    args = parser.parse_args()

    config_file = "linear_2"
    config_file_path = f"config/{config_file}.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    config['stop_time'] = args.stop_time
    router_template = config['templates']['router_template']
    router_template['MemoryArray']['efficiency'] = args.memory_efficiency

    detectors = config['templates']['bsm_template']['SingleHeraldedBSM']['detectors']
    for detector in detectors:
        detector['efficiency'] = args.detector_efficiency

    for qchannel in config['qchannels']:
        qchannel['distance'] = args.link_distance / 2
        qchannel['attenuation'] = args.attenuation

    topology = sq.RouterNetTopo(config)
    tl = topology.tl
    update_cchannel_delay(topology, classical_delay_node=args.classical_delay_node * 1000, classical_delay_hop=0) # ms to us

    # disable purification and swapping
    router: sq.QuantumRouter
    diabled_rules = [EP_AWAIT, EP_REQUEST, ES_A, ES_B, ES_B_END]
    for router in topology.get_nodes_by_type(sq.RouterNetTopo.QUANTUM_ROUTER):
        for rule in diabled_rules:
            router.disable_rule(rule)

    # set log
    file_name = (f"{config_file}_ld={args.link_distance:g}_me={args.memory_efficiency:g}_de={args.detector_efficiency:g}_"
                 f"a={args.attenuation:g}_cdn={args.classical_delay_node:g}")
    log_filename = args.dir / "log" / f"{file_name}.log"
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    log.set_logger(__name__, tl, str(log_filename) + '.event')
    log.set_logger_level('DEBUG')
    modules = ['request_app', 'swapping_bds', 'single_heralded', 'resource_manager']
    for module in modules:
        log.track_module(module)

    # set traffic and applications
    traffic_seed = 0
    traffic = Traffic(config_file, seed=traffic_seed)
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

    # kickoff the simulation
    tl.init()
    SingleHeraldedASurrogate.set_surrogate_parameter_from_timeline(tl)
    tl.run()

    time_to_serve_dict = defaultdict(float)
    for _, app in name_to_app.items():
        time_to_serve_dict |= app.time_to_serve
    tts_list = []
    reservation_list = []
    for reservation, time_to_serve in sorted(time_to_serve_dict.items()):
        reservation_list.append(reservation)
        tts_list.append(time_to_serve / MILLISECOND)

    parameters = {"link_distance": args.link_distance, "memory_efficiency": args.memory_efficiency,
                  "detector_efficiency": args.detector_efficiency, "attenuation": args.attenuation,
                  "classical_delay_node": args.classical_delay_node}

    # save h5 file and other log data
    h5_filename = args.dir / "h5" / f"{file_name}.h5"
    HDF5(h5_filename).save_link_data(parameters, tts_list)

    log_samples = list(zip(reservation_list, tts_list))
    if len(log_samples) > 20:
        log_samples = log_samples[:10] + log_samples[-10:]

    mylog = open(log_filename, "w")
    for reservation, time_to_serve in log_samples:
        mylog.write(f"{reservation}, {time_to_serve}\n")
    mylog.write(f"Average time to serve: {sum(tts_list) / len(tts_list):.3f} ms\n")
    mylog.close()