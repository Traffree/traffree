import os
import pickle
import statistics
import sys
import time

import neat
import traci
from sumolib import checkBinary  # Checks for the binary in environ vars

import main


def eval_genomes(genomes, config):
    nets = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    for index in range(len(ge)):
        main.run(checkBinary('sumo'), sumo_config_file, scheduler_type, nets[index], training=True)

        waiting_time_array = main.get_statistics()[0]
        obj = sum(waiting_time_array) / len(waiting_time_array) + 0.1 * statistics.stdev(waiting_time_array)
        ge[index].fitness = -obj
        print(ge[index].fitness)


def show_final_stats(winner_genome, config):
    print('\nBest genome:\n{!s}'.format(winner_genome))
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    main.run(checkBinary('sumo'), sumo_config_file, scheduler_type, winner_net)


def save_best_model(winner_genome):
    model_file_name = f'saved_models/neat/{scheduler_type}_{time.strftime("%d.%m.%Y-%H:%M")}.pkl'
    with open(model_file_name, "wb") as f:
        pickle.dump(winner_genome, f)
        f.close()


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

    winner_genome = p.run(eval_genomes, generations_number)
    show_final_stats(winner_genome, config)
    save_best_model(winner_genome)


if __name__ == '__main__':
    args = sys.argv[1:]

    scheduler_type = args[0] if len(args) > 0 else 'NeatScheduler'
    neat_config_file = args[1] if len(args) > 1 else 'configurations/neat/neat_config.txt'
    sumo_config_file = args[2] if len(args) > 2 else 'abstract_networks/grid/grid.sumocfg'
    generations_number = int(args[3]) if len(args) > 3 else 3

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, neat_config_file)
    run(config_path)

