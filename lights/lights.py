#!/usr/bin/python3

import os
import sys
import optparse

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


# # run with induction loop
# def run():
#     step = 0
#     passed_cars_sofar = 0
#     # start with phase 2 where SN has green
#     traci.trafficlight.setPhase("Ctl", 0)
#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()
#         if traci.trafficlight.getPhase("Ctl") == 0:
#             # we are not already switching
#             passed_cars_sofar += traci.inductionloop.getLastStepVehicleNumber("det_WR")
#             print(passed_cars_sofar)
#             if passed_cars_sofar > 4:
#                 # vehicle from the nwest, switch
#                 traci.trafficlight.setPhase("Ctl", 1)
#             else:
#                 # keep green for SN
#                 traci.trafficlight.setPhase("Ctl", 0)

#         step += 1

#     traci.close()
#     sys.stdout.flush()


# run with lane area detector
def run():
    step = 0
    # start with phase 2 where SN has green
    traci.trafficlight.setPhase("Ctl", 0)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if traci.trafficlight.getPhase("Ctl") == 0:
            # we are not already switching
            jam_cars = traci.lanearea.getJamLengthVehicle("det_WR")
            # print(jam_cars)
            if jam_cars > 4:
                # vehicle from the nwest, switch
                traci.trafficlight.setPhase("Ctl", 1)
            else:
                # keep green for SN
                traci.trafficlight.setPhase("Ctl", 0)

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