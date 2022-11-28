import argparse
import configparser
import datetime
import logging
import os
import sys
import time
from os import makedirs
from os.path import join
from time import strftime

import constants as ct
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("Front")


def init_directories(section):
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(ct.RESULTS_DIR):
        makedirs(ct.RESULTS_DIR)

    # Define output directory
    timestamp = strftime("%m%d_%H%M")

    section_str = str(section).capitalize()

    output_dir = join(ct.RESULTS_DIR, f"Front_{section_str}_{timestamp}")
    makedirs(output_dir)

    return output_dir


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


def parse_arguments():

    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)

    parser = argparse.ArgumentParser(
        description="It simulates adaptive padding on a set of web traffic traces."
    )

    parser.add_argument(
        "p",
        metavar="<traces path>",
        help="Path to the directory with the traffic traces to be simulated.",
    )
    parser.add_argument(
        "-format", metavar="<suffix of a file>", default="", help="suffix of a file."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="section",
        metavar="<config name>",
        help="Adaptive padding configuration.",
        choices=conf_parser.sections(),
        default="default",
    )

    parser.add_argument(
        "--log",
        type=str,
        dest="log",
        metavar="<log path>",
        default="stdout",
        help="path to the log file. It will print to stdout by default.",
    )

    args = parser.parse_args()
    config = dict(conf_parser._sections[args.section])
    config_logger(args)
    return args, config


def load_trace(fdir):
    with open(fdir, "r") as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype("float")
    return np.array(t)


def dump(trace, fname):
    global output_dir
    outfile = join(output_dir, fname)
    with open(outfile, "w") as fo:
        for packet in trace:
            line = "{:.4f}".format(packet[0]) + "\t" + f"{int(packet[1])}" + ct.NL
            fo.write(line)
        fo.close()


def simulate(fdir):
    if not os.path.exists(fdir):
        raise FileNotFoundError(fdir)
    # logger.debug("Simulating trace {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    trace = load_trace(fdir)
    trace = RP(trace)
    # print(trace)
    fname = fdir.split("/")[-1]
    dump(trace, fname)


def RP(trace):
    # format: [[time, pkt],[...]]
    # trace, cpkt_num, spkt_num, cwnd, swnd
    global client_dummy_pkt_num
    global server_dummy_pkt_num
    global min_wnd
    global max_wnd
    global start_padding_time
    global client_min_dummy_pkt_num
    global server_min_dummy_pkt_num

    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    if client_min_dummy_pkt_num != client_dummy_pkt_num:
        client_dummy_pkt = np.random.randint(client_min_dummy_pkt_num, client_dummy_pkt_num)
    else:
        client_dummy_pkt = client_dummy_pkt_num
    if server_min_dummy_pkt_num != server_dummy_pkt_num:
        server_dummy_pkt = np.random.randint(server_min_dummy_pkt_num, server_dummy_pkt_num)
    else:
        server_dummy_pkt = server_dummy_pkt_num
    logger.debug("client_wnd:", client_wnd)
    logger.debug("server_wnd:", server_wnd)
    logger.debug("client pkt:", client_dummy_pkt)
    logger.debug("server pkt:", server_dummy_pkt)

    first_incoming_pkt_time = trace[np.where(trace[:, 1] < 0)][0][0]
    last_pkt_time = trace[-1][0]

    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    client_timetable = client_timetable[
        np.where(start_padding_time + client_timetable[:, 0] <= last_pkt_time)
    ]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:, 0] += first_incoming_pkt_time
    server_timetable = server_timetable[
        np.where(start_padding_time + server_timetable[:, 0] <= last_pkt_time)
    ]

    # print("client_timetable")
    # print(client_timetable[:10])
    client_pkts = np.concatenate(
        (client_timetable, 888 * np.ones((len(client_timetable), 1))), axis=1
    )
    server_pkts = np.concatenate(
        (server_timetable, -888 * np.ones((len(server_timetable), 1))), axis=1
    )

    noisy_trace = np.concatenate((trace, client_pkts, server_pkts), axis=0)
    noisy_trace = noisy_trace[noisy_trace[:, 0].argsort(kind="mergesort")]
    return noisy_trace


def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))
    # print(wnd, num)
    # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
    timestamps = sorted(np.random.rayleigh(wnd, num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps), 1))


if __name__ == "__main__":
    global client_dummy_pkt_num
    global server_dummy_pkt_num
    global client_min_dummy_pkt_num
    global server_min_dummy_pkt_num
    global max_wnd
    global min_wnd
    global start_padding_time
    args, config = parse_arguments()

    if not os.path.exists(args.p):
        raise FileNotFoundError(f"Path {args.p} not found")

    logger.info(args)

    client_min_dummy_pkt_num = int(config.get("client_min_dummy_pkt_num", 1))
    server_min_dummy_pkt_num = int(config.get("server_min_dummy_pkt_num", 1))
    client_dummy_pkt_num = int(config.get("client_dummy_pkt_num", 300))
    server_dummy_pkt_num = int(config.get("server_dummy_pkt_num", 300))
    start_padding_time = int(config.get("start_padding_time", 0))
    max_wnd = float(config.get("max_wnd", 10))
    min_wnd = float(config.get("min_wnd", 10))
    MON_SITE_NUM = int(config.get("mon_site_num", 10))
    MON_INST_NUM = int(config.get("mon_inst_num", 10))
    UNMON_SITE_NUM = int(config.get("unmon_site_num", 100))
    print(f"client_min_dummy_pkt_num:{client_min_dummy_pkt_num}")
    print(f"server_min_dummy_pkt_num:{server_min_dummy_pkt_num}")
    print(f"client_dummy_pkt_num: {client_dummy_pkt_num}")
    print(f"server_dummy_pkt_num: {server_dummy_pkt_num}")

    print(f"max_wnd: {max_wnd}\nmin_wnd: {min_wnd}")
    print(f"start_padding_time: {start_padding_time}")

    flist = os.listdir(args.p)
    flist = [os.path.join(args.p, f) for f in flist if "-" in f]

    flist.sort()

    # Init run directories
    output_dir = init_directories(section=args.section)
    logger.info(f"Traces are dumped to {output_dir}")
    start = time.time()

    for i, f in tqdm(enumerate(flist), total=len(flist)):
        logger.debug(f"Simulating {f}")
        simulate(f)

    logger.info(f"Time: {time.time() - start}")
