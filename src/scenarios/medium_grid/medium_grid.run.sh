python3 ../../main.py normal/config.sumocfg --nogui
python3 ../../main.py normal/u_config.sumocfg BasicColorBasedScheduler --nogui
python3 ../../main.py normal/u_config.sumocfg NeatScheduler ../../saved_models/neat/winner.pkl ../../configurations/neat/neat_config.txt --nogui
python3 ../../main.py normal/u_config.sumocfg MultiDetectorNeatScheduler ../../saved_models/neat/md_winner.pkl ../../configurations/neat/md_neat_config.txt --nogui
python3 ../../main.py normal/u_config.sumocfg DQLScheduler ../../saved_models/DQL/DQL_17.07.2021-22:02.h5 --nogui
python3 ../../main.py normal/u_config.sumocfg MultiDetectorDQLScheduler ../../saved_models/DQL/multi_DQL_17.07.2021-23:18.h5 --nogui
python3 ../../main.py normal/u_config.sumocfg MultiDetectorGNNScheduler ../../saved_models/GNN/multi_GNN_offline_demo.pt u_map.net.xml --nogui