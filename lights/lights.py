#!/usr/bin/python3

import os
import sys
import optparse

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scheduler_interface.scheduler import Scheduler, Info

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
    return options


# run with lane area detector
def run():
    step = 0
    # start with phase 2 where SN has green
    traci.trafficlight.setPhase("Ctl", 0)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 19 == 0:
            north_count = traci.lanearea.getJamLengthVehicle("det_NR")
            east_count = traci.lanearea.getJamLengthVehicle("det_ER")
            south_count = traci.lanearea.getJamLengthVehicle("det_SR")
            west_count = traci.lanearea.getJamLengthVehicle("det_WR")

            prediction = Scheduler.predict(Info("Ctl", north_count, east_count, south_count, west_count))
            traci.trafficlight.setPhase("Ctl", prediction)

        step += 1

    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "lights.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()