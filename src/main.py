import optparse
import os
import pickle

from tensorflow.keras import models
from helper import choose_action
from models.GNN_model import GNNModel
import sys

# we need to import some python modules from the $SUMO_HOME/tools directory
import neat
from sumo_env import SumoEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import tensorflow as tf
from scheduler.basic_random_scheduler import BasicRandomScheduler, BasicRandomSchedulerInfo
from scheduler.basic_color_based_scheduler import BasicColorBasedScheduler, BasicColorBasedSchedulerInfo
from scheduler.neat_scheduler import NeatScheduler, NeatSchedulerInfo
from scheduler.multi_detector_neat_sceduler import MultiDetectorNeatScheduler, MultiDetectorNeatSchedulerInfo
from scheduler.DQL_scheduler import DQLScheduler, DQLSchedulerInfo
from scheduler.multi_detector_DQL_scheduler import MultiDetectorDQLScheduler, MultiDetectorDQLSchedulerInfo

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
from helper import *
from configurations import N_TIMESTEPS
import torch


def basic_random_scheduler_loop(tl_ids, lane2detector):
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                lanes = set(traci.trafficlight.getControlledLanes(tl_id))
                detectors = [lane2detector[lane] for lane in lanes]
                detectors = [t for item in detectors for t in item]

                north_count, east_count, south_count, west_count = get_lane_stats(detectors)
                info = BasicRandomSchedulerInfo(tl_id, north_count, east_count, south_count, west_count)
                prediction = BasicRandomScheduler.predict(info)

                # add yellow lights before switching
                old_phase = traci.trafficlight.getPhase(tl_id)
                if old_phase % 2:  # already yellow
                    traci.trafficlight.setPhase(tl_id, 2 * prediction)
                elif old_phase == 2 * prediction:  # color remains unchanged
                    traci.trafficlight.setPhase(tl_id, 2*prediction)
                else:  # assign yellow
                    yellow_phase = (2 * prediction + 3) % 4
                    traci.trafficlight.setPhase(tl_id, yellow_phase)
        step += 1


def basic_color_based_scheduler_loop(tl_ids, lane2detector):
    step = 0
    scheduler = BasicColorBasedScheduler()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                red, green, cont = get_red_green_lanes(tl_id)
                if cont:
                    continue

                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, green)))
                info = BasicColorBasedSchedulerInfo(tl_id, red_stats, green_stats)
                set_tl_phases(scheduler, info, tl_id)

        step += 1


def neat_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = NeatScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                red, green, cont = get_red_green_lanes(tl_id)
                if cont:
                    continue

                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, green)))
                info = NeatSchedulerInfo(tl_id, red_stats, green_stats)
                set_tl_phases(scheduler, info, tl_id)

        step += 1


def multi_detector_neat_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = MultiDetectorNeatScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                red, green, cont = get_red_green_lanes(tl_id)
                if cont:
                    continue

                red_stats = get_multi_detector_lane_stats(lane2detector, red)
                green_stats = get_multi_detector_lane_stats(lane2detector, green)
                info = MultiDetectorNeatSchedulerInfo(tl_id, red_stats, green_stats)
                set_tl_phases(scheduler, info, tl_id)

        step += 1


def dql_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = DQLScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                red, green, cont = get_red_green_lanes(tl_id)
                if cont:
                    continue

                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, green)))
                info = DQLSchedulerInfo(tl_id, red_stats, green_stats)
                set_tl_phases(scheduler, info, tl_id)

        step += 1


def multi_detector_dql_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = MultiDetectorDQLScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % N_TIMESTEPS == 0:
            for tl_id in tl_ids:
                red, green, cont = get_red_green_lanes(tl_id)
                if cont:
                    continue

                red_stats = get_multi_detector_lane_stats(lane2detector, red)
                green_stats = get_multi_detector_lane_stats(lane2detector, green)
                info = MultiDetectorDQLSchedulerInfo(tl_id, red_stats, green_stats)
                set_tl_phases(scheduler, info, tl_id)

        step += 1


def multi_detector_gnn_scheduler_loop(sumo_config_path, model, net_file, gui=False):
    net = sumolib.net.readNet(net_file)
    edge_index = torch.LongTensor(get_edge_index(net).T)

    sumo_env = SumoEnv(sumo_config_path, multiple_detectors=True, gui=gui)
    observation = sumo_env.get_observation()

    while True:
        observation = torch.FloatTensor(observation)
        action = choose_action(model, observation, edge_index)
        observation, _, done = sumo_env.step(action)
        if done:
            break

    sumo_env.reset()
    print_statistics("MultiDetectorGNNScheduler")


def basic_sumo_loop():
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()


def run(sumo_binary, config_path, scheduler_type, net=None, training=False):
    traci.start([sumo_binary, "-c", config_path, "--tripinfo-output", "tripinfo.xml"])
    tl_ids = traci.trafficlight.getIDList()
    detector_ids = traci.lanearea.getIDList()
    lane2detector = get_lane_2_detector(detector_ids)

    if scheduler_type == "BasicRandomScheduler":
        basic_random_scheduler_loop(tl_ids, lane2detector)
    elif scheduler_type == "BasicColorBasedScheduler":
        basic_color_based_scheduler_loop(tl_ids, lane2detector)
    elif scheduler_type == "NeatScheduler":
        neat_scheduler_loop(tl_ids, lane2detector, net)
    elif scheduler_type == "MultiDetectorNeatScheduler":
        multi_detector_neat_scheduler_loop(tl_ids, lane2detector, net)
    elif scheduler_type == "DQLScheduler":
        dql_scheduler_loop(tl_ids, lane2detector, net)
    elif scheduler_type == "MultiDetectorDQLScheduler":
        multi_detector_dql_scheduler_loop(tl_ids, lane2detector, net)
    else:
        basic_sumo_loop()

    traci.close()
    sys.stdout.flush()

    if not training:
        print_statistics(scheduler_type)


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options, args


def main():
    options, args = get_options()
    config_path = args[0]
    scheduler_type = args[1] if len(args) > 1 else None
    model_file = args[2] if len(args) > 2 else None
    config_file = args[3] if len(args) > 3 else 'configurations/neat/neat_config.txt'

    net = None
    if model_file:
        if scheduler_type == 'NeatScheduler' or scheduler_type == 'MultiDetectorNeatScheduler':
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_file)

            with open(model_file, "rb") as f:
                genome = pickle.load(f)
                net = neat.nn.FeedForwardNetwork.create(genome, config)
        elif scheduler_type == 'DQLScheduler' or scheduler_type == 'MultiDetectorDQLScheduler':
            net = tf.keras.models.load_model(model_file, compile=False)
        elif scheduler_type == 'MultiDetectorGNNScheduler':
            net = GNNModel(
                input_dim=18,
                output_dim=2,
                num_layers=1,
                dropout=0.25
            ).to(device)
            net.load_state_dict(torch.load(model_file, map_location=device))
            net.eval()
            
    # check binary
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    if scheduler_type == "MultiDetectorGNNScheduler":
        multi_detector_gnn_scheduler_loop(config_path, net, config_file, not options.nogui)
    else:
        run(sumo_binary, config_path, scheduler_type, net)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
