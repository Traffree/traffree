./grid.net.sh
../../sumo_helpers/generateTLSE2Detectors.py -n grid.net.xml -o grid.add.xml
python3 ../../sumo_helpers/tl_organizer.py grid.net.xml u_grid.net.xml
../../sumo_helpers/randomTrips.py -n grid.net.xml -r grid.rou.xml -e 200 -l
