./grid.net.sh
../../generateTLSE2Detectors.py -n grid.net.xml -o grid.add.xml
../../randomTrips.py -n grid.net.xml -r grid.rou.xml -e 2000 -l
