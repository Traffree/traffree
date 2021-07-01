from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import sys
import numpy as np
from main import get_lane_2_detector, get_multi_detector_lane_stats


class SumoEnv:
    def __init__(self, config_path):
        self.config_path = config_path
        self.start_sumo()
        self.tl_ids = list(filter(lambda tl_id: traci.trafficlight.getPhaseDuration(tl_id) != 999, traci.trafficlight.getIDList()))
        detector_ids = traci.lanearea.getIDList()
        self.lane2detector = get_lane_2_detector(detector_ids)
        self.old_wait_times = {}
        self.car_id_to_tl = {}

    def start_sumo(self):
        sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, "-c", "--waiting-time-memory", "100000", self.config_path, "--tripinfo-output", "tripinfo.xml"])

    def reset(self):
        traci.close()
        sys.stdout.flush()
        self.start_sumo()

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
                elif pattern[idx] == 'G' or pattern[idx] == 'g':
                    green.add(link_from)

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
            reward.append(-tl_waiting_time)

        return np.array(reward, dtype=float)

    def step(self, action):
        for i in range(11):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                finish_reward = 1  # maybe increased reward can speed up process
                return None, np.array([finish_reward] * len(self.tl_ids)), True

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
        reward = self.get_reward()
        return next_observation, reward, False

