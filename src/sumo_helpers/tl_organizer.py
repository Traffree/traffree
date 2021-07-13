import optparse
import os
import sys

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def update_durations(logic, max_dur, yellow_dur):
    res = ''
    for phase in logic:
        new_phase = phase.split('"')
        if 'G' in new_phase[3]:  # or 'g' in new_phase[3]
            new_phase[1] = max_dur
        else:
            new_phase[1] = yellow_dur
        res += '"'.join(new_phase)
    return res


def get_new_logic(logic, max_dur='1000', yellow_dur="6", dumb_dur="999"):
    new_phases = ''.join(update_durations(logic, max_dur, yellow_dur))
    if len(logic) == 8:
        new_phases = '\t\t<phase duration="' + max_dur + '" state="GGgrrrGGgrrr"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="yyyrrryyyrrr"/>\n' \
                     '\t\t<phase duration="' + max_dur + '" state="rrrGGgrrrGGg"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="rrryyyrrryyy"/>\n'
    elif len(logic) == 6:
        new_phases = '\t\t<phase duration="' + max_dur + '" state="GgrrGG"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="yyrryy"/>\n' \
                     '\t\t<phase duration="' + max_dur + '" state="rrGGrr"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="rryyyr"/>\n'
    elif len(logic) == 3:
        new_phases = '\t\t<phase duration="' + dumb_dur + '" state="GG"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="yy"/>\n' \
                     '\t\t<phase duration="' + dumb_dur + '" state="GG"/>\n' \
                     '\t\t<phase duration="' + yellow_dur + '" state="yy"/>\n'
    elif len(logic) == 4:
        first_phase = logic[0]
        first_phase_pattern = first_phase.split('"')[3]
        if first_phase_pattern in ('GG', 'GGGG'):
            new_phases = ''.join(update_durations(logic, dumb_dur, yellow_dur))
        elif len(first_phase_pattern) == 6:
            new_phases = '\t\t<phase duration="' + max_dur + '" state="GgrrGG"/>\n' \
                         '\t\t<phase duration="' + yellow_dur + '" state="yyrryy"/>\n' \
                         '\t\t<phase duration="' + max_dur + '" state="rrGGrr"/>\n' \
                         '\t\t<phase duration="' + yellow_dur + '" tate="rryyyr"/>\n'
    return new_phases


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    net = open(in_file, 'r')
    new_xml = ''
    while True:
        line = net.readline()
        new_xml += line
        if line.count('<tlLogic') > 0:
            cur_phases = []
            while True:
                phase = net.readline()
                if phase.count('</tlLogic>') > 0:
                    new_xml += get_new_logic(cur_phases)
                    new_xml += phase
                    break
                cur_phases.append(phase)

        if not line:
            break

    net.close()
    new_net = open(out_file, 'w')
    new_net.write(new_xml)
    new_net.close()


if __name__ == "__main__":
    main()
