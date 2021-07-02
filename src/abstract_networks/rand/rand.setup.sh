./rand.net.sh
../../sumo_helpers/generateTLSE2Detectors.py -n rand.net.xml -o rand.add.xml
python3 ../../sumo_helpers/tl_organizer.py rand.net.xml u_rand.net.xml
../../sumo_helpers/randomTrips.py -n rand.net.xml -r rand.rou.xml -e 2000 -l --period 0.8
