# Constants for trade history
ROLLING_WINDOW_IN_DAYS = 60

# Constants for the scoring function
VOLUME_FEE = 0.01
ROI_MIN = 0.0 # originally 0.02?
VOLUME_MIN = 1000
VOLUME_DECAY = 0.9
RAMP = 0.1 # originally 0.1
RHO_CAP = 0.1 # originally 0.1
KAPPA_NEXT = 0.02

# Build-up period constants for miner eligibility
MIN_EPOCHS_FOR_ELIGIBILITY = 2  # Must trade for 2+ days
MIN_TRADES_FOR_ELIGIBILITY = 2  # Must have 2+ trades
MIN_ACTIVE_EPOCHS = 2  # Must be active in 2+ epochs

# Weighting parameters
# Percentage we allocate to score well-performing miners so they don't get deregistered
GENERAL_POOL_WEIGHT_PERCENTAGE = 0.20
MINER_WEIGHT_PERCENTAGE = 1 - GENERAL_POOL_WEIGHT_PERCENTAGE

TOTAL_MINER_ALPHA_PER_DAY = 2952 # 7200 alpha per day for entire subnet * 0.41 (41% for miners)
EPOCH_DISTRIBUTION_COUNT = 20 # 20 epochs per day

# Subnet owner burn UID
BURN_UID = 210
# Subnet owner excess miner weight UID
EXCESS_MINER_WEIGHT_UID = 0
EXCESS_MINER_MIN_WEIGHT = 0.001