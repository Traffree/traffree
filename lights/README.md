# Simple traffic light demo

## Description

This demo changes traffic light color based on simple rule.

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
chmod +x lights.py
```

And then:

```bash
./lights.py
```

Pops up SUMO GUI. Set appropriate time delay and run simulation.
Note: Be patient. The last car appears at 200ms.


