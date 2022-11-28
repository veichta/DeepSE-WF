import argparse
import configparser
import itertools
import logging
import os
import sys
from os import makedirs
from os.path import isdir, join
from time import strftime

import adaptive as ap
import constants as ct
import numpy as np
import overheads as oh
from pparser import Trace, dump, parse
from tqdm import tqdm

logger = logging.getLogger("wtfpad")
# parameter

MON_SITE_NUM = 100
MON_INST_NUM = 60
UNMON_SITE_NUM = 0
OPEN_WORLD = 0


def init_directories(config):
    # Create a results dir if it doesn't exist yet
    if not isdir(ct.RESULTS_DIR):
        makedirs(ct.RESULTS_DIR)

    # Define output directory
    timestamp = strftime("%m%d_%H%M")
    output_dir = join(ct.RESULTS_DIR, f"wtfpad_{timestamp}")
    logger.info(f"Creating output directory: {output_dir}")

    # make the output directory
    makedirs(output_dir)

    return output_dir


def main():
    # parser config and arguments
    args, config = parse_arguments()

    if not os.path.exists(args.traces_path):
        raise FileNotFoundError(f"Path {args.traces_path} not found")

    logger.info(f"Arguments: {args}, Config: {config}")

    # Init run directories
    output_dir = init_directories(args.section)

    # Instantiate a new adaptive padding object
    wtfpad = ap.AdaptiveSimulator(config)

    # Run simulation on all traces
    latencies, bandwidths = [], []
    flist = os.listdir(args.traces_path)
    flist = [f for f in flist if "-" in f]
    flist.sort()

    for fname in tqdm(flist):
        trace = parse(join(args.traces_path, fname))

        # logger.info(f"Simulating trace: {fname}")
        simulated = wtfpad.simulate(Trace(trace))

        # dump simulated trace to results directory
        dump(simulated, join(output_dir, fname))

        # calculate overheads
        bw_ovhd = oh.bandwidth_ovhd(simulated, trace)
        bandwidths.append(bw_ovhd)
        logger.debug(f"Bandwidth overhead: {bw_ovhd}")

        lat_ovhd = oh.latency_ovhd(simulated, trace)
        latencies.append(lat_ovhd)
        logger.debug(f"Latency overhead: {lat_ovhd}")

    if OPEN_WORLD:
        for i in range(UNMON_SITE_NUM):
            fname = str(i)
            if os.path.exists(join(args.traces_path, fname)):
                trace = parse(join(args.traces_path, fname))
                logger.info(f"Simulating trace: {fname}")
                simulated = wtfpad.simulate(Trace(trace))
                # dump simulated trace to results directory
                dump(simulated, join(output_dir, fname))

                # calculate overheads
                bw_ovhd = oh.bandwidth_ovhd(simulated, trace)
                bandwidths.append(bw_ovhd)
                logger.debug(f"Bandwidth overhead: {bw_ovhd}")

                lat_ovhd = oh.latency_ovhd(simulated, trace)
                latencies.append(lat_ovhd)
                logger.debug(f"Latency overhead: {lat_ovhd}")
            else:
                logger.warn(f"File {fname} does not exist!")

    logger.info(f"Latency overhead: {np.median([l for l in latencies if l > 0.0])}")

    logger.info(f"Bandwidth overhead: {np.median([b for b in bandwidths if b > 0.0])}")


def parse_arguments():
    # Read configuration file
    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)

    parser = argparse.ArgumentParser(
        description="It simulates adaptive padding on a set of web traffic traces."
    )

    parser.add_argument(
        "traces_path",
        metavar="<traces path>",
        help="Path to the directory with the traffic traces to be simulated.",
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="section",
        metavar="<config name>",
        help="Adaptive padding configuration.",
        choices=conf_parser.sections(),
        default="normal_rcv",
    )

    parser.add_argument(
        "--log",
        type=str,
        dest="log",
        metavar="<log path>",
        default="stdout",
        help="path to the log file. It will print to stdout by default.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        dest="loglevel",
        metavar="<log level>",
        help="logging verbosity level.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Get section in config file
    config = conf_parser._sections[args.section]

    # Use default values if not specified
    config = dict(config, **conf_parser._sections["normal_rcv"])

    # logging config
    config_logger(args)

    return args, config


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != "stdout":
        log_file = open(args.log, "w")
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(-1)
