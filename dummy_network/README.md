# Instruction

### Generating .net.xml file

```bash
netconvert --node-files my_nodes.nod.xml -- edge-files my_edge.edg.xml -t my_type.type.xml -o my_net.net.xml
```

### Building

```bash
sumo -c my_config_file.sumocfg
```

Or with GUI:

```bash
sumo-gui -c my_config_file.sumocfg
```

