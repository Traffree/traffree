# Simple rerouting demo

[Source](https://www.youtube.com/watch?v=pwix3dZgCwA&ab_channel=TienanLi)

## Description

This demo enforces single vehicle to change its route.

### Declaring SUMO_HOME variable

```bash
export SUMO_HOME=<your path here>
```

### Which path to choose for SUMO_HOME ?

```bash
whereis sumo
```

You can choose one of them for example /usr/share/sumo

### Building

In case of denied permissions:

```bash
chmod +x demo.py
```

And then:

```bash
./demo.py
```

Pops up SUMO GUI. Set appropriate time delay and run simulation.


