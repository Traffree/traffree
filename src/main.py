import optparse
import os
import pickle
import sys
import re

# we need to import some python modules from the $SUMO_HOME/tools directory
import neat
from configurations.neat.config import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from scheduler.basic_random_scheduler import BasicRandomScheduler, BasicRandomSchedulerInfo
from scheduler.basic_color_based_scheduler import BasicColorBasedScheduler, BasicColorBasedSchedulerInfo
from scheduler.neat_scheduler import NeatScheduler, NeatSchedulerInfo
from scheduler.multi_detector_neat_sceduler import MultiDetectorNeatScheduler, MultiDetectorNeatSchedulerInfo

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import sumolib


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options, args


def get_lane_2_detector(detector_ids):
    lane2detector = {}
    for detector in detector_ids:
        lane = traci.lanearea.getLaneID(detector)
        detectors = lane2detector.get(lane, [])
        detectors.append(detector)
        lane2detector[lane] = detectors

    return lane2detector


def get_lane_stats(detectors):
    pattern = re.compile(r'(\D+)(\d+)(\D+)(\d+)')
    north_count, east_count, south_count, west_count = 0, 0, 0, 0
    for detector in detectors:
        name = detector.split("_")[1]  # e2det_A1A0_0 -> A1A0
        parts = re.match(pattern, name)

        from_col, from_row, to_col, to_row = parts.group(1, 2, 3, 4)
        jam_length = traci.lanearea.getJamLengthVehicle(detector)
        if from_col < to_col:
            west_count += jam_length
        elif from_col > to_col:
            east_count += jam_length
        elif from_row > to_row:
            north_count += jam_length
        elif from_row < to_row:
            south_count += jam_length

    return north_count, east_count, south_count, west_count


def basic_color_based_scheduler_loop(tl_ids, lane2detector):
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 19 == 0:
            for tl_id in tl_ids:
                red, green = set(), set()

                links = traci.trafficlight.getControlledLinks(tl_id)
                pattern = traci.trafficlight.getRedYellowGreenState(tl_id)
                duration = traci.trafficlight.getPhaseDuration(tl_id)
                if duration == 999:  # case of 2 straight roads joining
                    old_phase = traci.trafficlight.getPhase(tl_id)
                    traci.trafficlight.setPhase(tl_id, old_phase)
                    continue

                for idx, link in enumerate(links):
                    link_from = link[0][0]
                    if pattern[idx] == 'R' or pattern[idx] == 'r':
                        red.add(link_from)
                    elif pattern[idx] == 'G' or pattern[idx] == 'g':
                        green.add(link_from)

                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, green)))

                info = BasicColorBasedSchedulerInfo(tl_id, red_stats, green_stats)
                prediction = BasicColorBasedScheduler.predict(info)

                old_phase = traci.trafficlight.getPhase(tl_id)
                if prediction == 0:
                    # maintain green
                    traci.trafficlight.setPhase(tl_id, old_phase)
                else:
                    # switch to next phase (which is yellow followed by red)
                    new_phase = (old_phase + 1) % 4
                    traci.trafficlight.setPhase(tl_id, new_phase)
        step += 1


def basic_random_scheduler_loop(tl_ids, lane2detector):
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 19 == 0:
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


def neat_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = NeatScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 11 == 0:
            for tl_id in tl_ids:
                red, green = set(), set()

                links = traci.trafficlight.getControlledLinks(tl_id)
                pattern = traci.trafficlight.getRedYellowGreenState(tl_id)
                duration = traci.trafficlight.getPhaseDuration(tl_id)
                if duration == 999:  # case of 2 straight roads joining
                    old_phase = traci.trafficlight.getPhase(tl_id)
                    traci.trafficlight.setPhase(tl_id, old_phase)
                    continue

                for idx, link in enumerate(links):
                    link_from = link[0][0]
                    if pattern[idx] == 'R' or pattern[idx] == 'r':
                        red.add(link_from)
                    elif pattern[idx] == 'G' or pattern[idx] == 'g':
                        green.add(link_from)

                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(lane2detector, green)))

                info = NeatSchedulerInfo(tl_id, red_stats, green_stats)
                prediction = scheduler.predict(info)

                old_phase = traci.trafficlight.getPhase(tl_id)
                if prediction >= 0:
                    # maintain green
                    traci.trafficlight.setPhase(tl_id, old_phase)
                else:
                    # switch to next phase (which is yellow followed by red)
                    new_phase = (old_phase + 1) % 4
                    traci.trafficlight.setPhase(tl_id, new_phase)
        step += 1


def multi_detector_neat_scheduler_loop(tl_ids, lane2detector, net):
    step = 0
    scheduler = MultiDetectorNeatScheduler(net)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 11 == 0:
            for tl_id in tl_ids:
                red, green = set(), set()

                links = traci.trafficlight.getControlledLinks(tl_id)
                pattern = traci.trafficlight.getRedYellowGreenState(tl_id)
                duration = traci.trafficlight.getPhaseDuration(tl_id)
                if duration == 999:  # case of 2 straight roads joining
                    old_phase = traci.trafficlight.getPhase(tl_id)
                    traci.trafficlight.setPhase(tl_id, old_phase)
                    continue

                for idx, link in enumerate(links):
                    link_from = link[0][0]
                    if pattern[idx] == 'R' or pattern[idx] == 'r':
                        red.add(link_from)
                    elif pattern[idx] == 'G' or pattern[idx] == 'g':
                        green.add(link_from)

                red_stats = get_multi_detector_lane_stats(lane2detector, red)
                green_stats = get_multi_detector_lane_stats(lane2detector, green)

                info = MultiDetectorNeatSchedulerInfo(tl_id, red_stats, green_stats)

                prediction = scheduler.predict(info)

                old_phase = traci.trafficlight.getPhase(tl_id)
                if prediction >= 0:
                    # maintain green
                    traci.trafficlight.setPhase(tl_id, old_phase)
                else:
                    # switch to next phase (which is yellow followed by red)
                    new_phase = (old_phase + 1) % 4
                    traci.trafficlight.setPhase(tl_id, new_phase)
        step += 1


def get_multi_detector_lane_stats(lane2detector, lanes):
    detectors = [lane2detector[lane] for lane in lanes]
    detectors = [t for item in detectors for t in item]
    stats = [[0]*len(speed_thresholds) for _ in range(N_detectors_per_lane)]

    for detector in detectors:
        jam = traci.lanearea.getJamLengthVehicle(detector)
        _, idx, speed, _, _ = detector.split('_')
        speed_idx = speed_thresholds.index(speed)
        stats[int(idx)][speed_idx] += jam

    return stats


def basic_sumo_loop():
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()


def run(scheduler_type, net=None, training=False):
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
    else:
        basic_sumo_loop()

    traci.close()
    sys.stdout.flush()

    if not training:
        print_statistics(scheduler_type)


def get_statistics():
    waiting_time_array = []
    waiting_count_array = []
    stop_time_array = []
    time_loss_array = []
    for trip_info in sumolib.xml.parse('tripinfo.xml', ['tripinfo']):
        waiting_time_array.append(float(trip_info.waitingTime))
        waiting_count_array.append(float(trip_info.waitingCount))
        stop_time_array.append(float(trip_info.stopTime))
        time_loss_array.append(float(trip_info.timeLoss))

    return waiting_time_array, waiting_count_array, stop_time_array, time_loss_array


def print_statistics(scheduler_type):
    waiting_time_array, waiting_count_array, stop_time_array, time_loss_array = get_statistics()
    print("Scheduler Type: ", scheduler_type if scheduler_type is not None else "Default")
    print("Waiting time statistics")
    print("Max: ", max(waiting_time_array))
    print("Min: ", min(waiting_time_array))
    print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))
    # print("\n")

    print("Waiting count statistics")
    print("Max: ", max(waiting_count_array))
    print("Min: ", min(waiting_count_array))
    print("Avg: ", sum(waiting_count_array) / len(waiting_count_array))
    # print("\n")

    print("Stop time statistics")
    print("Max: ", max(stop_time_array))
    print("Min: ", min(stop_time_array))
    print("Avg: ", sum(stop_time_array) / len(stop_time_array))
    # print("\n")

    print("Time loss statistics")
    print("Max: ", max(time_loss_array))
    print("Min: ", min(time_loss_array))
    print("Avg: ", sum(time_loss_array) / len(time_loss_array))
    print("\n")


def main():
    options, args = get_options()
    config_path = args[0]
    scheduler_type = args[1] if len(args) > 1 else None
    model_file = args[2] if len(args) > 2 else None
    neat_config_file = args[3] if len(args) > 3 else 'configurations/neat/neat_config.txt'

    net = None
    if model_file:
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    neat_config_file)

        with open(model_file, "rb") as f:
            genome = pickle.load(f)
            net = neat.nn.FeedForwardNetwork.create(genome, config)

    # check binary
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumo_binary, "-c", config_path, "--tripinfo-output", "tripinfo.xml"])
    run(scheduler_type, net)


if __name__ == "__main__":
    main()
