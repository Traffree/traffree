import sys

import numpy as np
import traci
from numpy.lib.function_base import average
from sumolib import checkBinary  # Checks for the binary in environ vars

from configurations import N_TIMESTEPS
from helper import get_lane_2_detector, get_multi_detector_lane_stats


class SumoEnv:
    def __init__(self, config_path, multiple_detectors, gui=False):
        self.gui = gui
        self.config_path = config_path
        self.multiple_detectors = multiple_detectors
        self.start_sumo()
        self.tl_ids = sorted(traci.trafficlight.getIDList())
        detector_ids = traci.lanearea.getIDList()
        self.lane2detector = get_lane_2_detector(detector_ids)
        self.old_wait_times = {}
        self.car_id_to_tl = {}

    def start_sumo(self):
        if (self.gui):
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, "-c", self.config_path, "--waiting-time-memory", "100000", "--tripinfo-output", "tripinfo.xml"])
        traci.simulationStep()

    def reset(self):
        traci.close()
        sys.stdout.flush()

    def get_observation(self):
        next_observation = []
        for tl_id in self.tl_ids:
            red, green = set(), set()

            links = traci.trafficlight.getControlledLinks(tl_id)
            pattern = traci.trafficlight.getRedYellowGreenState(tl_id)

            for idx, link in enumerate(links):
                link_from = link[0][0]
                if pattern[idx] == 'R' or pattern[idx] == 'r':
                    red.add(link_from)
                else:
                    green.add(link_from)

            if self.multiple_detectors:
                red_stats = get_multi_detector_lane_stats(self.lane2detector, red)
                green_stats = get_multi_detector_lane_stats(self.lane2detector, green)
                arr = red_stats + green_stats
                next_observation.append([x for sub_arr in arr for x in sub_arr])
            else:
                red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(self.lane2detector, red)))
                green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(self.lane2detector, green)))
                next_observation.append([red_stats, green_stats])

        return np.array(next_observation)

    def get_reward(self):
        reward = []
        for tl_id in self.tl_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            car_ids = []
            for controlled_lane in controlled_lanes:
                car_ids.extend(traci.lane.getLastStepVehicleIDs(controlled_lane))

            tl_waiting_time = 0
            for car_id in car_ids:
                car_waiting_time = traci.vehicle.getAccumulatedWaitingTime(car_id)

                tl_waiting_time += car_waiting_time - self.old_wait_times.get(car_id, 0)
                if self.car_id_to_tl.get(car_id, 0) != tl_id:
                    self.old_wait_times[car_id] = car_waiting_time
                    self.car_id_to_tl[car_id] = tl_id
            reward.append((-tl_waiting_time / len(car_ids)) if car_ids else 0)
            
        return np.array(reward, dtype=float)

    def step(self, action):
        for idx, tl_id in enumerate(self.tl_ids):
            if traci.trafficlight.getPhaseDuration(tl_id) == 999:
                continue

            old_phase = traci.trafficlight.getPhase(tl_id)
            if action[idx][1] == 0:
                # maintain green
                traci.trafficlight.setPhase(tl_id, old_phase)
            else:
                # switch to next phase (which is yellow followed by red)
                new_phase = (old_phase + 1) % 4
                traci.trafficlight.setPhase(tl_id, new_phase)
        
        for i in range(N_TIMESTEPS):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                finish_reward = 1  # maybe increased reward can speed up process
                return None, np.array([finish_reward] * len(self.tl_ids)), True

        next_observation = self.get_observation()
        reward = self.get_reward()
        return next_observation, reward, False

