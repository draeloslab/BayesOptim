import sys

filename = sys.argv[1]
total = 0.0

with open(filename, 'r') as file:
    for line in file:
        if line.startswith("Total time:"):
            time = float(line.split()[2])
            total += time

print(f"Total time: {total} s")