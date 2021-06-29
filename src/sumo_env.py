from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import sys
import numpy as np
from main import get_lane_2_detector, get_multi_detector_lane_stats


class SumoEnv:
    def __init__(self, config_path):
        self.config_path = config_path
        self.start_sumo()
        self.tl_ids = filter(lambda tl_id: traci.trafficlight.getPhaseDuration(tl_id) != 999,
                             traci.trafficlight.getIDList())
        detector_ids = traci.lanearea.getIDList()
        self.lane2detector = get_lane_2_detector(detector_ids)

    def start_sumo(self):
        sumo_binary = checkBinary('sumo-gui')
        traci.start([sumo_binary, "-c", self.config_path, "--tripinfo-output", "tripinfo.xml"])

    def reset(self):
        traci.close()
        sys.stdout.flush()
        self.start_sumo()

    def get_observation(self):
        next_observation = []
        for idx, tl_id in enumerate(self.tl_ids):
            red, green = set(), set()

            links = traci.trafficlight.getControlledLinks(tl_id)
            pattern = traci.trafficlight.getRedYellowGreenState(tl_id)

            for idx, link in enumerate(links):
                link_from = link[0][0]
                if pattern[idx] == 'R' or pattern[idx] == 'r':
                    red.add(link_from)
                elif pattern[idx] == 'G' or pattern[idx] == 'g':
                    green.add(link_from)

            red_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(self.lane2detector, red)))
            green_stats = sum(map(lambda arr: arr[-1], get_multi_detector_lane_stats(self.lane2detector, green)))
            next_observation.append([red_stats, green_stats])

        return np.array(next_observation)

    def step(self, action):
        for i in range(11):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                return None, 1, True

        for idx, tl_id in enumerate(self.tl_ids):
            old_phase = traci.trafficlight.getPhase(tl_id)
            if action[idx][1] == 0:
                # maintain green
                traci.trafficlight.setPhase(tl_id, old_phase)
            else:
                # switch to next phase (which is yellow followed by red)
                new_phase = (old_phase + 1) % 4
                traci.trafficlight.setPhase(tl_id, new_phase)

        next_observation = self.get_observation()
        reward = np.sum(next_observation, axis=1, dtype=float)
        return next_observation, reward, False

