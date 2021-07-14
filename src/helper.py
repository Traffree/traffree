import re

import numpy as np
import sumolib
import torch
import torch.nn.functional as F
import traci

from configurations.config import *
from scheduler.scheduler_interface import SchedulerInfoInterface, SchedulerInterface


def set_tl_phases(scheduler: SchedulerInterface, info: SchedulerInfoInterface, tl_id):
    prediction = scheduler.predict(info)

    old_phase = traci.trafficlight.getPhase(tl_id)
    if prediction == 0:
        # maintain green
        traci.trafficlight.setPhase(tl_id, old_phase)
    else:
        # switch to next phase (which is yellow followed by red)
        new_phase = (old_phase + 1) % 4
        traci.trafficlight.setPhase(tl_id, new_phase)


def get_red_green_lanes(tl_id):
    red, green = set(), set()

    links = traci.trafficlight.getControlledLinks(tl_id)
    pattern = traci.trafficlight.getRedYellowGreenState(tl_id)
    duration = traci.trafficlight.getPhaseDuration(tl_id)
    if duration == 999:  # case of 2 straight roads joining
        old_phase = traci.trafficlight.getPhase(tl_id)
        traci.trafficlight.setPhase(tl_id, old_phase)
        return None, None, True

    for idx, link in enumerate(links):
        link_from = link[0][0]
        if pattern[idx] == 'R' or pattern[idx] == 'r':
            red.add(link_from)
        elif pattern[idx] == 'G' or pattern[idx] == 'g':
            green.add(link_from)

    return red, green, False


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


def get_node_to_index(net):
    nodes = [node.getID() for node in net.getNodes()]
    indices = np.argsort(nodes)
    return {node : index for node, index in zip(nodes, indices)}


def get_edge_index(net):
    node2idx = get_node_to_index(net)

    edges = net.getEdges()
    edge_index = []
    for edge in edges:
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        edge_index.append([node2idx[from_node], node2idx[to_node]])
    
    return np.array(edge_index)


def choose_action(model, observation, edge_index, n_actions=2):
    # add batch dimension to the observation if only a single example was provided
    with torch.no_grad():
        logits = model(observation, edge_index)
        action = torch.multinomial(logits, num_samples=1)
        action = action.flatten()
        action = F.one_hot(action, n_actions)
        return action


def get_statistics():
    waiting_time_array = []
    waiting_count_array = []
    stop_time_array = []
    time_loss_array = []
    duration_array = []
    arrival_array = []
    for trip_info in sumolib.xml.parse('tripinfo.xml', ['tripinfo']):
        waiting_time_array.append(float(trip_info.waitingTime))
        waiting_count_array.append(float(trip_info.waitingCount))
        stop_time_array.append(float(trip_info.stopTime))
        time_loss_array.append(float(trip_info.timeLoss))
        duration_array.append(float(trip_info.duration))
        arrival_array.append(float(trip_info.arrival))

    waiting_time_pct = [100 * waiting / total for waiting, total in zip(waiting_time_array, duration_array)]
    return waiting_time_array, waiting_count_array, stop_time_array, time_loss_array, waiting_time_pct, arrival_array[-1]


def print_statistics(scheduler_type):
    waiting_time_array, waiting_count_array, stop_time_array, time_loss_array, waiting_time_pct, last_arrival = get_statistics()
    print("Scheduler Type: ", scheduler_type if scheduler_type is not None else "Default")
    print("Waiting time statistics")
    print("Max: ", max(waiting_time_array))
    print("Min: ", min(waiting_time_array))
    print("Avg: ", sum(waiting_time_array) / len(waiting_time_array))

    print("Waiting count statistics")
    print("Max: ", max(waiting_count_array))
    print("Min: ", min(waiting_count_array))
    print("Avg: ", sum(waiting_count_array) / len(waiting_count_array))

    print("Stop time statistics")
    print("Max: ", max(stop_time_array))
    print("Min: ", min(stop_time_array))
    print("Avg: ", sum(stop_time_array) / len(stop_time_array))

    print("Time loss statistics")
    print("Max: ", max(time_loss_array))
    print("Min: ", min(time_loss_array))
    print("Avg: ", sum(time_loss_array) / len(time_loss_array))

    print("Waiting time percentage statistics")
    print("Max: ", max(waiting_time_pct))
    print("Min: ", min(waiting_time_pct))
    print("Avg: ", sum(waiting_time_pct) / len(waiting_time_pct))

    print("---------------------")
    print("Simulation duration: ", last_arrival)
    print("\n")
