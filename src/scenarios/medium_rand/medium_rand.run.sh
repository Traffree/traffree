#python3 ../../main.py light/rand.sumocfg --nogui
#python3 ../../main.py light/u_rand.sumocfg BasicColorBasedScheduler --nogui
#python3 ../../main.py light/u_rand.sumocfg NeatScheduler ../../saved_models/neat/winner.pkl ../../configurations/neat/neat_config.txt
#python3 ../../main.py light/u_rand.sumocfg MultiDetectorNeatScheduler ../../saved_models/neat/md_winner.pkl ../../configurations/neat/md_neat_config.txt --nogui
python3 ../../main.py light/u_rand.sumocfg DQLScheduler ../../saved_models/DQL/DQL_17.07.2021-22:02.h5 --nogui
python3 ../../main.py light/u_rand.sumocfg MultiDetectorDQLScheduler ../../saved_models/DQL/multi_DQL_17.07.2021-23:18.h5 --nogui
#python3 ../../main.py light/u_rand.sumocfg MultiDetectorGNNScheduler ../../saved_models/GNN/try/multi_GNN_offline_16.07.2021-14_01.pt u_map.net.xml --nogui
