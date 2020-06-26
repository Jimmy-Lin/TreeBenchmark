import sys
import os  

# from collectors.benchmark import benchmark
from collectors.configurations import configurations
from collectors.scale import samples, features
from plots.radar import plot_radar
from plots.tradeoff import plot_tradeoff
from plots.scalability import plot_scalability

if len(sys.argv) < 2:
    "No command specified."
    exit(1)

command = sys.argv[1]
if command == "benchmark":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py benchmark [datasets,...] [algorithms,...]")
        exit(1)
    # benchmark(sys.argv[2].split(","), sys.argv[3].split(","))
elif command == "radar":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py radar [datasets,...] [algorithms,...]")
        exit(1)
    for dataset in sys.argv[2].split(","):
        plot_radar(dataset, sys.argv[3].split(","))
elif command == "configurations":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py configurations [datasets,...] [algorithms,...]")
        exit(1)
    configurations(sys.argv[2].split(","), sys.argv[3].split(","))
elif command == "tradeoff":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py tradeoff [datasets,...] [algorithms,...]")
        exit(1)
    for dataset in sys.argv[2].split(","):
        plot_tradeoff(dataset, sys.argv[3].split(","))
elif command == "scale":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py scale [datasets,...] [algorithms,...]")
        exit(1)
    for dataset in sys.argv[2].split(","):
        samples(dataset, sys.argv[3].split(","))
        features(dataset, sys.argv[3].split(","))
elif command == "scalability":
    if len(sys.argv) < 3:
        print("Usage python3 python/run.py scalability [datasets,...] [algorithms,...]")
        exit(1)
    for dataset in sys.argv[2].split(","):
        plot_scalability(dataset, sys.argv[3].split(","))
