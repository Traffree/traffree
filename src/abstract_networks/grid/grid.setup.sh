./grid.net.sh
../../generateTLSE2Detectors.py -n grid.net.xml -o grid.add.xml
python3 ../../tl_organizer.py grid.net.xml u_grid.net.xml
../../randomTrips.py -n grid.net.xml -r grid.rou.xml -e 200 -l
