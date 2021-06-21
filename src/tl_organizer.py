import os
import sys

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def update_durations(logic, max_dur):
    res = ''
    for phase in logic:
        new_phase = phase.split('"')
        if 'G' in new_phase[3]:  # or 'g' in new_phase[3]
            new_phase[1] = max_dur
        # else duration is 3
        res += '"'.join(new_phase)
    return res


def get_new_logic(logic, max_dur='1000'):
    new_phases = ''.join(update_durations(logic, max_dur))
    if len(logic) == 8:
        new_phases = '\t\t<phase duration="' + max_dur + '" state="GGgrrrGGgrrr"/>\n' \
                     '\t\t<phase duration="3"  state="yyyrrryyyrrr"/>\n' \
                     '\t\t<phase duration="' + max_dur + '" state="rrrGGgrrrGGg"/>\n' \
                     '\t\t<phase duration="3"  state="rrryyyrrryyy"/>\n'
    elif len(logic) == 6:
        new_phases = '\t\t<phase duration="' + max_dur + '" state="GgrrGG"/>\n' \
                     '\t\t<phase duration="3"  state="yyrryy"/>\n' \
                     '\t\t<phase duration="' + max_dur + '" state="rrGGrr"/>\n' \
                     '\t\t<phase duration="3"  state="rryyyr"/>\n'
    elif len(logic) == 3:
        new_phases = '\t\t<phase duration="999" state="GG"/>\n' \
                     '\t\t<phase duration="3"  state="yy"/>\n' \
                     '\t\t<phase duration="999" state="GG"/>\n' \
                     '\t\t<phase duration="3"  state="yy"/>\n'
    elif len(logic) == 4:
        first_phase = logic[0]
        first_phase_pattern = first_phase.split('"')[3]
        if first_phase_pattern == 'GG':
            new_phases = ''.join(update_durations(logic, '999'))
        elif len(first_phase_pattern) == 6:
            new_phases = '\t\t<phase duration="' + max_dur + '" state="GgrrGG"/>\n' \
                         '\t\t<phase duration="3"  state="yyrryy"/>\n' \
                         '\t\t<phase duration="' + max_dur + '" state="rrGGrr"/>\n' \
                         '\t\t<phase duration="3"  state="rryyyr"/>\n'
    return new_phases


def main():

    net = open('rand.net.xml', 'r')
    new_xml = ''
    while True:
        line = net.readline()
        new_xml += line
        if line.count('<tlLogic') > 0:
            cur_phases = []
            while True:
                phase = net.readline()
                if phase.count('</tlLogic>') > 0:
                    new_xml += get_new_logic(cur_phases, max_dur='36')
                    new_xml += phase
                    break
                cur_phases.append(phase)

        if not line:
            break

    net.close()
    new_net = open('u_rand.net.xml', 'w')
    new_net.write(new_xml)
    new_net.close()


if __name__ == "__main__":
    main()
