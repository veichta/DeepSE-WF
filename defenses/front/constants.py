from os.path import abspath, dirname, join, pardir

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))
RESULTS_DIR = "../data/defended/"  # TODO: Put your results path here

# Files
CONFIG_FILE = join(f"{BASE_DIR}/front", "config.ini")

# Logging format
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

# Characters
CSV_SEP = ";"
TRACE_SEP = "\t"
NL = "\n"  # new line

# MPU
TOR_CELL_SIZE = 512
MTU = 1

# Directions
IN = -1
OUT = 1
DIR_NAMES = {IN: "in", OUT: "out"}
DIRECTIONS = [OUT, IN]

# AP states
GAP = 0x00
BURST = 0x01
WAIT = 0x02

# Mappings
EP2DIRS = {"client": OUT, "server": IN}
MODE2STATE = {"gap": GAP, "burst": BURST}

# Histograms
INF = float("inf")
