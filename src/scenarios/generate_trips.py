import subprocess
import sys
import os


def generate_trips(path, duration=2000, period_min=2, period_max=6, variants=20):

    for period in range(period_min, period_max):
        period_str = f"0.{period}"

        for i in range(variants):
            route_duration = duration if period > 3 else duration // 2
            route_tag = f"{route_duration}_{period_str.replace('.', '')}_{i}"
            subprocess.run([
                "python3", 
                "../sumo_helpers/randomTrips.py",
                "-n", f"{path}/u_map.net.xml", 
                "-r", f"{path}/training/route_{route_tag}.rou.xml",
                "-e", f"{route_duration}",
                "-l",
                "--period", f"{period_str}",
                "--random"
            ])
            with open(f"{path}/training/cfg_{route_tag}.sumocfg", "w") as config_file:
                content = f"""
                    <configuration>
                        <input>
                            <net-file value="../u_map.net.xml"/>
                            <route-files value="route_{route_tag}.rou.xml"/>
                            <additional-files value="../map.add.xml"/>
                        </input>
                    </configuration>
                """
                config_file.write(content)
    
    for file_name in os.listdir(f"{path}/training/"):
        if '.alt.' in file_name:
            os.remove(f"{path}/training/{file_name}")

if __name__ == "__main__":
    generate_trips(sys.argv[1])