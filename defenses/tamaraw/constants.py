from os.path import abspath, dirname, join, pardir

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))
RESULTS_DIR = "../data/defended/"  # TODO: Put your results path here

# Files
CONFIG_FILE = join(f"{BASE_DIR}/tamaraw", "config.ini")

# Logging format
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
