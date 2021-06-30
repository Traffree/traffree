python3 ../../main.py grid.sumocfg --nogui
python3 ../../main.py grid.sumocfg BasicColorBasedScheduler --nogui
#python3 ../../main.py grid.sumocfg BasicRandomScheduler --nogui  #TODO needs fix
python3 ../../main.py grid.sumocfg NeatScheduler ../../saved_models/neat/winner.pkl ../../configurations/neat/neat_config.txt --nogui
python3 ../../main.py grid.sumocfg MultiDetectorNeatScheduler ../../saved_models/neat/md_winner.pkl ../../configurations/neat/md_neat_config.txt --nogui
python3 ../../main.py grid.sumocfg DQLScheduler ../../saved_models/DQL/DQL_30.06.2021-20_49.h5 --nogui