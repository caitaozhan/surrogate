'''
the main for generating data for a link
'''

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.purification import BBPSSWProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.quantum_manager import QuantumManager
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import SECOND, MILLISECOND, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
import sequence.utils.log as log
from request_app import RequestAppTimeToServe
from traffic import Traffic
from utils import HDF5, update_cchannel_delay

EntanglementGenerationA.set_global_type(SINGLE_HERALDED)
EntanglementGenerationB.set_global_type(SINGLE_HERALDED)
BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ld", "--link_distance", type=float, default=5000.0)
    parser.add_argument("-me", "--memory_efficiency", type=float, default=0.7)
    parser.add_argument("-de", "--detector_efficiency", type=float, default=0.96)
    parser.add_argument("-a", "--attenuation", type=float, default=0.0002)
    parser.add_argument("-cdn", "--classical_delay_node", type=float, default=0.1, help="classical delay in ms for each node")
    parser.add_argument("-d", "--dir", type=Path, default=Path("link"))
    parser.add_argument("-st", "--stop_time", type=float, default=5002 * SECOND, help="stop time in seconds")

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

    topology = RouterNetTopo(config)
    update_cchannel_delay(topology, config_file_path, classical_delay_node=args.classical_delay_node, classical_delay_hop=0)

    tl = topology.tl

    log_filename = args.dir / (f"{config_file}_ld={args.link_distance:g}_me={args.memory_efficiency:g}_de={args.detector_efficiency:g}_"
                               f"a={args.attenuation:g}_cdn={args.classical_delay_node:g}.log")
    args.dir.mkdir(parents=True, exist_ok=True)
    # log.set_logger(__name__, tl, str(log_filename))
    # log.set_logger_level('DEBUG')
    # log.set_logger_level('INFO')
    # modules = ['test_swapping_order', 'request_app', 'swapping_bds', 'single_heralded', 'resource_manager', 'memory']
    # modules = ['main_link']

    # for module in modules:
    #     log.track_module(module)

    traffic_seed = 0
    traffic = Traffic(config_file, seed=traffic_seed)
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

    time_to_serve_dict = defaultdict(float)
    for _, app in name_to_app.items():
        time_to_serve_dict |= app.time_to_serve
    tts_list = []
    reservation_list = []
    for reservation, time_to_serve in sorted(time_to_serve_dict.items()):
        reservation_list.append(reservation)
        tts_list.append(time_to_serve / MILLISECOND)

    parameters = {
        "link_distance": args.link_distance,
        "memory_efficiency": args.memory_efficiency,
        "detector_efficiency": args.detector_efficiency,
        "attenuation": args.attenuation,
        "classical_delay_node": args.classical_delay_node,
    }
    HDF5(log_filename.with_suffix(".h5")).save_link_data(parameters, tts_list)

    log_samples = list(zip(reservation_list, tts_list))
    if len(log_samples) > 20:
        log_samples = log_samples[:10] + log_samples[-10:]

    mylog = open(log_filename, "w")
    for reservation, time_to_serve in log_samples:
        mylog.write(f"{reservation}, {time_to_serve}\n")
    mylog.write(f"Average time to serve: {sum(tts_list) / len(tts_list):.3f} ms\n")
    mylog.close()
