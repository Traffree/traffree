#!/usr/bin/python3

import os
import sys
import optparse

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scheduler.basic_random_scheduler import BasicRandomScheduler, BasicRandomSchedulerInfo
from scheduler.scheduler_interface import SchedulerInterface

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary, net   # Checks for the binary in environ vars
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
        name = detector.split("_")[1] # e2det_A1A0_0 -> A1A0
        parts = re.match(pattern, name)
        
        from_col, from_row, to_col, to_row = parts.group(1, 2, 3, 4)
        jam_length = traci.lanearea.getJamLengthVehicle(detector)

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
                lanes = set(traci.trafficlight.getControlledLanes(tl_id))
                # print(lanes)

                # links = traci.trafficlight.getControlledLinks(tl)

                detectors = [lane2detector[lane] for lane in lanes]
                detectors = [t for item in detectors for t in item]  
                # print(detectors)

                north_count, east_count, south_count, west_count = get_lane_stats(detectors)

                info = BasicRandomSchedulerInfo(tl_id, north_count, east_count, south_count, west_count)
                prediction = BasicRandomScheduler.predict(info)

                # add yellow lights before switching
                old_phase = traci.trafficlight.getPhase(tl_id)
                if old_phase % 2:  # already yellow
                    traci.trafficlight.setPhase(tl_id, 2*prediction)
                elif old_phase == 2*prediction:  # color remains unchanged
                    traci.trafficlight.setPhase(tl_id, 2*prediction)
                else:  # assign yellow
                    yellow_phase = (2*prediction + 3) % 4
                    traci.trafficlight.setPhase(tl_id, yellow_phase)

        step += 1

    traci.close()
    sys.stdout.flush()


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