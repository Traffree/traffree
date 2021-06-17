./grid.net.sh
../abstract_networks/generateTLSE2Detectors.py -n grid.net.xml -o grid.add.xml
../open_maps/randomTrips.py -n grid.net.xml -r grid.rou.xml -e 2000 -l --period 0.8
