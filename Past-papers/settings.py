import logging
from os.path import join

LOG_LEVEL = logging.INFO
# The BitMEX API requires permanent API keys. Go to https://testnet.bitmex.com/app/apiKeys to fill these out.

# The BitMEX API requires permanent API keys. Go to https://testnet.bitmex.com/app/apiKeys to fill these out.
API_KEY = "EFFZOvOW_WJq00T-fyqXSODU" #"-9NdCmNitAGUaCAXVYoFIbdv" #  #"vI_PDNYSFP7d_jsClDUfW6EF" #"9FR7reF9F71NDZG_BDoMsfm9" #"ZJ7ZG0bDrem884wQkNnvv2PB"
API_SECRET = "Y0HgsMr3YRvSxtaNYvRAbEL0ofp-aARxwg6Au7cXto6qTZtV"

class settingsETHUSD:
    # For running ETH to USD
    # API URL.
    BASE_URL = "https://www.bitmex.com/api/v1/" # Once you're ready, uncomment this.

    # The BitMEX API requires permanent API keys. Go to https://testnet.bitmex.com/app/apiKeys to fill these out.
    API_KEY = "EFFZOvOW_WJq00T-fyqXSODU" #"-9NdCmNitAGUaCAXVYoFIbdv" #  #"vI_PDNYSFP7d_jsClDUfW6EF" #"9FR7reF9F71NDZG_BDoMsfm9" #"ZJ7ZG0bDrem884wQkNnvv2PB"
    API_SECRET = "Y0HgsMr3YRvSxtaNYvRAbEL0ofp-aARxwg6Au7cXto6qTZtV"

    SYMBOL = 'ETHUSD'

    # BTCUSD
    #  XBTUSD

    # Wait times between orders / errors
    API_REST_INTERVAL = 1
    API_ERROR_INTERVAL = 10
    TIMEOUT = 7

    # Available levels: logging.(DEBUG|INFO|WARN|ERROR)
    LOG_LEVEL = logging.INFO

    # If any of these files (and this file) changes, reload the bot.
    WATCHED_FILES = [join('market_maker', 'market_maker.py'), join('market_maker', 'bitmex.py'), join('market_maker', 'settings.py')]

    # always amend orders
    RELIST_INTERVAL = 0.00

    # hyperparameters
    GAMMA = 100 #8000?
    K = 50 # 400 - 200
    D = 450
    THETA = 100
    ETA = 0.004
    ETA2 = 0.0006
    MAX_POS = THETA*2
    REG_COEF = 100
    PERC = 0.02
    THRESHOLD = 0.75
    SAD_THRESHOLD = 1.5
    MIN_VOLA = 0.002

    DRY_RUN = False
    POST_ONLY = True

    # Not necessary:
    #=================
    ORDER_PAIRS = 6

    # ORDER_START_SIZE will be the number of contracts submitted on level 1
    # Number of contracts from level 1 to ORDER_PAIRS - 1 will follow the function
    # [ORDER_START_SIZE + ORDER_STEP_SIZE (Level -1)]
    ORDER_START_SIZE = 100
    ORDER_STEP_SIZE = 100

    # Distance between successive orders, as a percentage (example: 0.005 for 0.5%)
    INTERVAL = 0.005

    # Minimum spread to maintain, in percent, between asks & bids
    MIN_SPREAD = 0.01

    # If True, market-maker will place orders just inside the existing spread and work the interval % outwards,
    # rather than starting in the middle and killing potentially profitable spreads.
    MAINTAIN_SPREADS = True

    # This number defines far much the price of an existing order can be from a desired order before it is amended.
    # This is useful for avoiding unnecessary calls and maintaining your ratelimits.
    #
    # Further information:
    # Each order is designed to be (INTERVAL*n)% away from the spread.
    # If the spread changes and the order has moved outside its bound defined as
    # abs((desired_order['price'] / order['price']) - 1) > settings.RELIST_INTERVAL)
    # it will be resubmitted.
    #
    # 0.01 == 1%
    RELIST_INTERVAL = 0.01

    CHECK_POSITION_LIMITS = False
    MIN_POSITION = -10000
    MAX_POSITION = 100

    #========================


    # Might be necessary
    #=======================
    LOOP_INTERVAL = 5

    # Wait times between orders / errors
    API_REST_INTERVAL = 1
    API_ERROR_INTERVAL = 10
    TIMEOUT = 7

    # If we're doing a dry run, use these numbers for BTC balances
    DRY_BTC = 50
    ORDERID_PREFIX = "mm_bitmex_"
    CONTRACTS = ['ETHUSD']
    # ===========================


class settingsXBTUSD:
    # For running BitCoin to USD 
    # API URL.
    BASE_URL = "https://www.bitmex.com/api/v1/" # Once you're ready, uncomment this.

    # The BitMEX API requires permanent API keys. Go to https://testnet.bitmex.com/app/apiKeys to fill these out.
    API_KEY = "EFFZOvOW_WJq00T-fyqXSODU" #"-9NdCmNitAGUaCAXVYoFIbdv" #  #"vI_PDNYSFP7d_jsClDUfW6EF" #"9FR7reF9F71NDZG_BDoMsfm9" #"ZJ7ZG0bDrem884wQkNnvv2PB"
    API_SECRET = "Y0HgsMr3YRvSxtaNYvRAbEL0ofp-aARxwg6Au7cXto6qTZtV"

    SYMBOL = 'XBTUSD'

    # BTCUSD
    #  XBTUSD

    # Wait times between orders / errors
    API_REST_INTERVAL = 1
    API_ERROR_INTERVAL = 10
    TIMEOUT = 7

    # Available levels: logging.(DEBUG|INFO|WARN|ERROR)
    LOG_LEVEL = logging.INFO

    # If any of these files (and this file) changes, reload the bot.
    WATCHED_FILES = [join('market_maker', 'market_maker.py'), join('market_maker', 'bitmex.py'), join('market_maker', 'settings.py')]

    # always amend orders
    RELIST_INTERVAL = 0.00

    # hyperparameters
    GAMMA = 8000 #8000? #1200
    K = 8000/(10**25) #30 # 4000 - 0.033
    D = 9000
    THETA = 100
    T = 10
    ETA = 0.004
    ETA2 = 0.0006
    MAX_POS = THETA*3
    REG_COEF = 50
    PERC = 0.012
    THRESHOLD = 5
    SAD_THRESHOLD = 50
    MIN_VOLA = 0.00

    DRY_RUN = False
    POST_ONLY = True

    # Not necessary:
    #=================
    ORDER_PAIRS = 6

    # ORDER_START_SIZE will be the number of contracts submitted on level 1
    # Number of contracts from level 1 to ORDER_PAIRS - 1 will follow the function
    # [ORDER_START_SIZE + ORDER_STEP_SIZE (Level -1)]
    ORDER_START_SIZE = 100
    ORDER_STEP_SIZE = 100

    # Distance between successive orders, as a percentage (example: 0.005 for 0.5%)
    INTERVAL = 0.005

    # Minimum spread to maintain, in percent, between asks & bids
    MIN_SPREAD = 0.01

    # If True, market-maker will place orders just inside the existing spread and work the interval % outwards,
    # rather than starting in the middle and killing potentially profitable spreads.
    MAINTAIN_SPREADS = True

    # This number defines far much the price of an existing order can be from a desired order before it is amended.
    # This is useful for avoiding unnecessary calls and maintaining your ratelimits.
    #
    # Further information:
    # Each order is designed to be (INTERVAL*n)% away from the spread.
    # If the spread changes and the order has moved outside its bound defined as
    # abs((desired_order['price'] / order['price']) - 1) > settings.RELIST_INTERVAL)
    # it will be resubmitted.
    #
    # 0.01 == 1%
    RELIST_INTERVAL = 0.01

    CHECK_POSITION_LIMITS = False
    MIN_POSITION = -10000
    MAX_POSITION = 100

    #========================


    # Might be necessary
    #=======================
    LOOP_INTERVAL = 5

    # Wait times between orders / errors
    API_REST_INTERVAL = 1
    API_ERROR_INTERVAL = 10
    TIMEOUT = 7

    # If we're doing a dry run, use these numbers for BTC balances
    DRY_BTC = 50
    ORDERID_PREFIX = "mm_bitmex_"
    CONTRACTS = ['XBTUSD']
    # ===========================


class SettingsDict:
    OPTIONS = {
        'ETHUSD': settingsETHUSD(),
        'XBTUSD': settingsXBTUSD()
    }
