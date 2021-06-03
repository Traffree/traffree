# Random Trips simulation on real map.
[Source](https://www.youtube.com/watch?v=LWecm_rCPJw&ab_channel=RodrigueTchamna)


### Search and Download Open Street Map ([OSM](https://www.openstreetmap.org/))
Find your prefered map and export .osm file. (In our case delisi_map.osm).

### Convert the Map into SUMO Network
```bash
netconvert --osm-files delisi_map.osm -o delisi.net.xml
```
### Add trip and route to the network using build-in Python scripts randomTrips.py
```bash
python randomTrips.py -n delisi.net.xml -r delisi.rou.xml -e 2000 -l
```

### Create SUMO config file (delisi.sumocfg) with the following content
```bash
<configuration> 

<input> 
<net-file value="delisi.net.xml"/> 
<route-files value="delisi.rou.xml"/> 
</input> 

<time> 
<begin value="0"/> 
<end value="2000"/> 
</time> 

</configuration>
```
### Building

```bash
sumo -c delisi.sumocfg
```

Or with GUI:

```bash
sumo-gui -c delisi.sumocfg
