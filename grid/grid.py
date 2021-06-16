#!/usr/bin/python3

import optparse
import os
import sys

from scheduler.basic_color_based_scheduler import BasicColorBasedScheduler, BasicColorBasedSchedulerInfo

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options, args


def getLane2Detector(detector_ids):
    lane2detector = {}
    for detector in detector_ids:
        lane = traci.lanearea.getLaneID(detector)
        detectors = lane2detector.get(lane, [])
        detectors.append(detector)
        lane2detector[lane] = detectors

    return lane2detector


import re


def get_lane_stats(detectors):
    pattern = re.compile(r'(\D+)(\d+)(\D+)(\d+)')
    north_count, east_count, south_count, west_count = 0, 0, 0, 0
    for detector in detectors:
        name = detector.split("_")[1]  # e2det_A1A0_0 -> A1A0
        parts = re.match(pattern, name)

        from_col, from_row, to_col, to_row = parts.group(1, 2, 3, 4)
        jam_length = traci.lanearea.getJamLengthVehicle(detector)
        print(jam_length)
        if (from_col < to_col):
            west_count += jam_length
        elif (from_col > to_col):
            east_count += jam_length
        elif (from_row > to_row):
            north_count += jam_length
        elif (from_row < to_row):
            south_count += jam_length

    return north_count, east_count, south_count, west_count


# run with lane area detector
def run():
    step = 0
    # start with phase 2 where SN has green

    tl_ids = traci.trafficlight.getIDList()
    detector_ids = traci.lanearea.getIDList()
    lane2detector = getLane2Detector(detector_ids)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 19 == 0:
            for tl_id in tl_ids:
                red, green = set(), set()
                # yellow = set()  # we won't need to count yellow links cause there are none of them
                links = traci.trafficlight.getControlledLinks(tl_id)
                pattern = traci.trafficlight.getRedYellowGreenState(tl_id)
                for idx, link in enumerate(links):
                    link_from = link[0][0]
                    if pattern[idx] == 'R' or pattern[idx] == 'r':
                        red.add(link_from)
                    elif pattern[idx] == 'G' or pattern[idx] == 'g':
                        green.add(link_from)
                    # elif pattern[idx] == 'Y' or pattern[idx] == 'y':
                    #     yellow.add(link_from)

                red_stats = get_colored_lane_stats(lane2detector, red)
                green_stats = get_colored_lane_stats(lane2detector, green)
                # yellow_stats = get_colored_lane_stats(lane2detector, yellow)

                info = BasicColorBasedSchedulerInfo(tl_id, red_stats, green_stats)
                prediction = BasicColorBasedScheduler.predict(info)

                old_phase = traci.trafficlight.getPhase(tl_id)
                if prediction == 0:
                    # maintain green
                    traci.trafficlight.setPhase(old_phase)
                else:
                    # switch to next phase (which is yellow followed by red)
                    new_phase = (old_phase + 1) % 4
                    traci.trafficlight.setPhase(new_phase)

                # # old code
                # add yellow lights before switching
                # if old_phase % 2:  # already yellow
                #     traci.trafficlight.setPhase(tl_id, 2 * prediction)
                # elif old_phase == 2 * prediction:  # color remains unchanged
                #     traci.trafficlight.setPhase(tl_id, 2 * prediction)
                # else:  # assign yellow
                #     yellow_phase = (2 * prediction + 3) % 4
                #     traci.trafficlight.setPhase(tl_id, yellow_phase)

        step += 1

    traci.close()
    sys.stdout.flush()


# new helper
def get_colored_lane_stats(lane2detector, lanes):
    detectors = [lane2detector[lane] for lane in lanes]
    detectors = [t for item in detectors for t in item]
    jams = [traci.lanearea.getJamLengthVehicle(detector) for detector in detectors]
    # print(jams)
    return sum(jams)


# TODO: nothing to do, just remark: below is an old run method
# # run with lane area detector
# def run():
#     step = 0
#     # start with phase 2 where SN has green

#     tl_ids = traci.trafficlight.getIDList()
#     detector_ids = traci.lanearea.getIDList()
#     lane2detector = getLane2Detector(detector_ids)

#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()

#         if step % 19 == 0:
#             for tl_id in tl_ids:
#                 lanes = set(traci.trafficlight.getControlledLanes(tl_id))
#                 # print(lanes)

#                 # links = traci.trafficlight.getControlledLinks(tl)

#                 detectors = [lane2detector[lane] for lane in lanes]
#                 detectors = [t for item in detectors for t in item]
#                 # print(detectors)

#                 north_count, east_count, south_count, west_count = get_lane_stats(detectors)

#                 info = BasicRandomSchedulerInfo(tl_id, north_count, east_count, south_count, west_count)
#                 prediction = BasicRandomScheduler.predict(info)

#                 # add yellow lights before switching
#                 old_phase = traci.trafficlight.getPhase(tl_id)
#                 if old_phase % 2:  # already yellow
#                     traci.trafficlight.setPhase(tl_id, 2*prediction)
#                 elif old_phase == 2*prediction:  # color remains unchanged
#                     traci.trafficlight.setPhase(tl_id, 2*prediction)
#                 else:  # assign yellow
#                     yellow_phase = (2*prediction + 3) % 4
#                     traci.trafficlight.setPhase(tl_id, yellow_phase)

#         step += 1

#     traci.close()
#     sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options, args = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "grid.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])
    run()
