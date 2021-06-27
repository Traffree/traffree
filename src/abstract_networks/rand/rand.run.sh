python3 ../../main.py rand.sumocfg --nogui
python3 ../../main.py u_rand.sumocfg BasicColorBasedScheduler --nogui
#python3 ../../main.py rand.sumocfg BasicRandomScheduler --nogui  #TODO needs fix
python3 ../../main.py u_rand.sumocfg NeatScheduler ../../saved_models/neat/winner.pkl ../../configurations/neat/neat_config.txt --nogui
python3 ../../main.py u_rand.sumocfg MultiDetectorNeatScheduler ../../saved_models/neat/md_winner.pkl ../../configurations/neat/md_neat_config.txt --nogui
