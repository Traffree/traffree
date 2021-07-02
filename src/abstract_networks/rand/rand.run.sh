python3 ../../main.py rand.sumocfg --nogui
python3 ../../main.py u_rand.sumocfg BasicColorBasedScheduler --nogui
#python3 ../../main.py rand.sumocfg BasicRandomScheduler --nogui
python3 ../../main.py u_rand.sumocfg NeatScheduler ../../saved_models/neat/winner.pkl ../../configurations/neat/neat_config.txt --nogui
python3 ../../main.py u_rand.sumocfg MultiDetectorNeatScheduler ../../saved_models/neat/md_winner.pkl ../../configurations/neat/md_neat_config.txt --nogui
python3 ../../main.py u_rand.sumocfg DQLScheduler ../../saved_models/DQL/DQL_02.07.2021-10:52.h5 --nogui
python3 ../../main.py u_rand.sumocfg MultiDetectorDQLScheduler ../../saved_models/DQL/multi_DQL_02.07.2021-16:04.h5 --nogui