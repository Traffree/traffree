import random
import os
import sys
import time
import neat
import pickle
import main
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def eval_genomes(genomes, config):
    nets = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    for index in range(len(ge)):
        traci.start([checkBinary('sumo'), "-c", "abstract_networks/grid/grid.sumocfg", "--tripinfo-output", "tripinfo.xml", "-W"])
        main.run("NeatScheduler", nets[index], training=True)

        waiting_time_array = main.get_statistics()[0]  # TODO: what if we choose other metrics?
        ge[index].fitness = -sum(waiting_time_array)
        print(ge[index].fitness)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 50 generations.  # TODO: take from command line or from config
    winner_genome = p.run(eval_genomes, 3)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner_genome))
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    # TODO: take params from command line. generate appropriate winner.pkl name
    traci.start([checkBinary('sumo'), "-c", "abstract_networks/grid/grid.sumocfg", "--tripinfo-output", "tripinfo.xml", "-W"])
    main.run("NeatScheduler", winner_net)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner_genome, f)
        f.close()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run(config_path)

