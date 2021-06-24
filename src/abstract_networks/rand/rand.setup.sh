./rand.net.sh
../../generateTLSE2Detectors.py -n rand.net.xml -o rand.add.xml
python3 ../../tl_organizer.py rand_net.xml u_rand.net.xml
../../randomTrips.py -n rand.net.xml -r rand.rou.xml -e 2000 -l --period 0.8
