from __future__ import absolute_import
from time import sleep
import sys
import datetime
from os.path import getmtime
import random
import requests
import atexit
import signal

from market_maker import bitmex
#from market_maker import settings
from market_maker.settings import SettingsDict
from market_maker.utils import log, constants, errors, math
import time
import math as math_lib

import threading
import numpy as np
from decimal import *
import pandas as pd 
import ccxt

# Chosing the correct settings file based on the parameter
if len(sys.argv) == 1:
    settings = SettingsDict.OPTIONS['ETHUSD']
else:
    settings = SettingsDict.OPTIONS[str(sys.argv[1])]


# Used symbols
SYMBOLS = ('ETHUSD', 'XBTUSD')

print(settings)

from influxdb import InfluxDBClient

# Used for reloading the bot - saves modified times of key files
import os
watched_files_mtimes = [(f, getmtime(f)) for f in settings.WATCHED_FILES]


#
# Helpers
#
logger = log.setup_custom_logger('root')

GAMMA = settings.GAMMA
K = settings.K
D = settings.D
THETA = settings.THETA
ETA = settings.ETA
ETA2 = settings.ETA2
MAX_POS = settings.MAX_POS
THRESHOLD = THETA*2.5
OF_LIMIT = 500000 # TODO: add this to settings

class ExchangeInterface:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        # if len(sys.argv) > 1:
        #     self.symbol = sys.argv[1]
        # else:
        self.symbol = settings.SYMBOL
        self.bitmex = bitmex.BitMEX(base_url=settings.BASE_URL, symbol=self.symbol,
                                    apiKey=settings.API_KEY, apiSecret=settings.API_SECRET,
                                    orderIDPrefix=settings.ORDERID_PREFIX, postOnly=settings.POST_ONLY,
                                    timeout=settings.TIMEOUT)

    def cancel_order(self, order):
        tickLog = self.get_instrument()['tickLog']
        logger.info("Canceling: %s %d @ %.*f" % (order['side'], order['orderQty'], tickLog, order['price']))
        while True:
            try:
                self.bitmex.cancel(order['orderID'])
                sleep(settings.API_REST_INTERVAL)
            except ValueError as e:
                logger.info(e)
                sleep(settings.API_ERROR_INTERVAL)
            else:
                break

    def cancel_all_orders(self):
        if self.dry_run:
            return

        logger.info("Resetting current position. Canceling all existing orders.")
        tickLog = self.get_instrument()['tickLog']

        # In certain cases, a WS update might not make it through before we call this.
        # For that reason, we grab via HTTP to ensure we grab them all.
        orders = self.bitmex.http_open_orders()

        for order in orders:
            logger.info("Canceling: %s %d @ %.*f" % (order['side'], order['orderQty'], tickLog, order['price']))

        if len(orders):
            self.bitmex.cancel([order['orderID'] for order in orders])

        sleep(settings.API_REST_INTERVAL)

    def get_portfolio(self):
        contracts = settings.CONTRACTS
        portfolio = {}
        for symbol in contracts:
            position = self.bitmex.position(symbol=symbol)
            instrument = self.bitmex.instrument(symbol=symbol)

            if instrument['isQuanto']:
                future_type = "Quanto"
            elif instrument['isInverse']:
                future_type = "Inverse"
            elif not instrument['isQuanto'] and not instrument['isInverse']:
                future_type = "Linear"
            else:
                raise NotImplementedError("Unknown future type; not quanto or inverse: %s" % instrument['symbol'])

            if instrument['underlyingToSettleMultiplier'] is None:
                multiplier = float(instrument['multiplier']) / float(instrument['quoteToSettleMultiplier'])
            else:
                multiplier = float(instrument['multiplier']) / float(instrument['underlyingToSettleMultiplier'])

            portfolio[symbol] = {
                "currentQty": float(position['currentQty']),
                "futureType": future_type,
                "multiplier": multiplier,
                "markPrice": float(instrument['markPrice']),
                "spot": float(instrument['indicativeSettlePrice'])
            }

        return portfolio

    def calc_delta(self):
        """Calculate currency delta for portfolio"""
        portfolio = self.get_portfolio()
        spot_delta = 0
        mark_delta = 0
        for symbol in portfolio:
            item = portfolio[symbol]
            if item['futureType'] == "Quanto":
                spot_delta += item['currentQty'] * item['multiplier'] * item['spot']
                mark_delta += item['currentQty'] * item['multiplier'] * item['markPrice']
            elif item['futureType'] == "Inverse":
                spot_delta += (item['multiplier'] / item['spot']) * item['currentQty']
                mark_delta += (item['multiplier'] / item['markPrice']) * item['currentQty']
            elif item['futureType'] == "Linear":
                spot_delta += item['multiplier'] * item['currentQty']
                mark_delta += item['multiplier'] * item['currentQty']
        basis_delta = mark_delta - spot_delta
        delta = {
            "spot": spot_delta,
            "mark_price": mark_delta,
            "basis": basis_delta
        }
        return delta

    def get_delta(self, symbol=None):
        total_contracts = 0
        # if symbol == "XBTUSD":
        #     total_contracts = self.get_position(symbol)['currentQty']
        # else:
        for symbol in SYMBOLS:
            total_contracts += self.get_position(symbol)['currentQty']
        print('Total Contracts: {}'.format(total_contracts))

        return total_contracts

    def get_instrument(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        return self.bitmex.instrument(symbol)

    def get_margin(self):
        if self.dry_run:
            return {'marginBalance': float(settings.DRY_BTC), 'availableFunds': float(settings.DRY_BTC)}
        return self.bitmex.funds()

    def get_orders(self):
        if self.dry_run:
            return []
        return self.bitmex.open_orders()

    def get_highest_buy(self):
        buys = [o for o in self.get_orders() if o['side'] == 'Buy']
        if not len(buys):
            return {'price': -2**32}
        highest_buy = max(buys or [], key=lambda o: o['price'])
        return highest_buy if highest_buy else {'price': -2**32}

    def get_lowest_sell(self):
        sells = [o for o in self.get_orders() if o['side'] == 'Sell']
        if not len(sells):
            return {'price': 2**32}
        lowest_sell = min(sells or [], key=lambda o: o['price'])
        return lowest_sell if lowest_sell else {'price': 2**32}  # ought to be enough for anyone

    def get_position(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        return self.bitmex.position(symbol)

    def get_ticker(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        return self.bitmex.ticker_data(symbol)

    def is_open(self):
        """Check that websockets are still open."""
        return not self.bitmex.ws.exited

    def check_market_open(self):
        instrument = self.get_instrument()
        if instrument["state"] != "Open" and instrument["state"] != "Closed":
            raise errors.MarketClosedError("The instrument %s is not open. State: %s" %
                                           (self.symbol, instrument["state"]))

    def check_if_orderbook_empty(self):
        """This function checks whether the order book is empty"""
        instrument = self.get_instrument()
        if instrument['midPrice'] is None:
            raise errors.MarketEmptyError("Orderbook is empty, cannot quote")

    def amend_bulk_orders(self, orders):
        if self.dry_run:
            return orders
        return self.bitmex.amend_bulk_orders(orders)

    def create_bulk_orders(self, orders):
        if self.dry_run:
            return orders
        return self.bitmex.create_bulk_orders(orders)

    def cancel_bulk_orders(self, orders):
        if self.dry_run:
            return orders
        return self.bitmex.cancel([order['orderID'] for order in orders])

class OrderManager:
    def __init__(self):
        self.exchange = ExchangeInterface(settings.DRY_RUN)
        # Once exchange is created, register exit handler that will always cancel orders
        # on any error.
        atexit.register(self.exit)
        signal.signal(signal.SIGTERM, self.exit)

        logger.info("Using symbol %s." % self.exchange.symbol)

        if settings.DRY_RUN:
            logger.info("Initializing dry run. Orders printed below represent what would be posted to BitMEX.")
        else:
            logger.info("Order Manager initializing, connecting to BitMEX. Live run: executing real trades.")

        self.start_time = datetime.datetime.now()
        self.instrument = self.exchange.get_instrument()
        self.starting_qty = self.exchange.get_delta(self.exchange.symbol)
        self.running_qty = None

        self.cur_volatility = None
        self.act_volatility = None
        self.cur_qty = None
        self.prev_qty = None
        self.streak = 0
        self.prev_len = 0
        self.cur_len = 0
        self.idle = 0
        self.first = True
        self.sleep_ctr = 0
        self.general_ctr = 0
        self.ctr = 0
        self.to_record_vola = True
        self.vola_adjust = 1.
        self.slope_ctr = 0
        self.plus = True

        exchange = ccxt.bitmex()
        logger.info('Connected to CCXT')
        date_N_days_ago = (datetime.datetime.now() - datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        since = time.mktime(datetime.datetime.strptime(date_N_days_ago, "%Y-%m-%d %H:%M:%S").timetuple())*1000
        if self.exchange.symbol == "ETHUSD":
            df = exchange.fetch_ohlcv('ETH/USD', timeframe = '1m', since=since, limit=500)
            print ('---'*20)
            print ('CONNECT TO CCXT ETHUSD')
            print ('---'*20)
        elif self.exchange.symbol == "XBTUSD":
            df = exchange.fetch_ohlcv('BTC/USD', timeframe = '1m', since=since, limit=500)
            print ('---'*20)
            print ('CONNECT TO CCXT XBTUSD')
            print ('---'*20)
        df = pd.DataFrame(df)
        df.columns = ["Timestamp", "Open", "High", "Low", "tick", "Volume"]

        self.client = InfluxDBClient(host='localhost', port=8086)
        self.client.switch_database('example')

        self.df = pd.DataFrame({'tick': df.tick.values.tolist()})
        logger.info('Retrieved data from CCXT! Len df: {}'.format(len(self.df)))

        self.of = None

        self.reset()

    def reset(self):
        self.exchange.cancel_all_orders()
        self.sanity_check()
        self.print_status()

        # Create orders and converge.
        self.one_loop()

    def print_status(self):
        """Print the current MM status."""

        margin = self.exchange.get_margin()
        position = self.exchange.get_position()
        self.running_qty = self.exchange.get_delta(self.exchange.symbol)
        tickLog = self.exchange.get_instrument()['tickLog']
        self.start_XBt = margin["marginBalance"]

        logger.debug("Current XBT Balance: %.6f" % XBt_to_XBT(self.start_XBt))
        logger.debug("Current Contract Position: %d" % self.running_qty)
        if settings.CHECK_POSITION_LIMITS:
            logger.debug("Position limits: %d/%d" % (settings.MIN_POSITION, settings.MAX_POSITION))
        if position['currentQty'] != 0:
            logger.debug("Avg Cost Price: %.*f" % (tickLog, float(position['avgCostPrice'])))
            logger.debug("Avg Entry Price: %.*f" % (tickLog, float(position['avgEntryPrice'])))
        logger.debug("Contracts Traded This Run: %d" % (self.running_qty - self.starting_qty))
        logger.debug("Total Contract Delta: %.4f XBT" % self.exchange.calc_delta()['spot'])

    def get_ticker(self):
        ticker = self.exchange.get_ticker()
        tickLog = self.exchange.get_instrument()['tickLog']

        # Midpoint, used for simpler order placement.
        self.start_position_mid = ticker["mid"]
        #print (self.start_position_mid)
        logger.info("%s Ticker: New Mid %f" % (self.instrument['symbol'], self.start_position_mid))
        return ticker

    def calc_res_price(self, mid, qty, vola, max_pos=MAX_POS):
        #print (qty) 
        VAR = max(vola, 0.0005)
        logger.info('Qty: {}, GAMMA: {}, VAR: {}, D: {}'.format(qty, GAMMA, VAR, D))
        r = mid - (qty*GAMMA*VAR*D)/MAX_POS
        #if abs(self.slope_ctr) >= 6:
        #    if self.exchange.symbol == "ETHUSD":
        #        r += 0.05*self.slope_ctr
        #    elif self.exchange.symbol == "XBTUSD":
        #        r += 0.5*self.slope_ctr
        # if self.exchange.symbol == "ETHUSD":
        #     r -= np.sign(qty)*(abs(qty)/THRESHOLD)*0.05*10 #0.05: min tick; 10: arbitrary constant.
        spread = max(0.1, GAMMA*VAR*D + np.log10(1+GAMMA/K))
        return r, spread

    def get_qty(self, qty, vola):
        #buy_qty = THETA*np.exp(-ETA2*qty) if qty < 0 else THETA*np.exp(-ETA*qty)
        #sell_qty = THETA*np.exp(ETA2*qty) if qty > 0 else THETA*np.exp(ETA*qty)
        single_qty = self.single_qty
        if vola >  0.8:
            self.vola_adjust = 0
        elif abs(qty) > 250 or abs(single_qty) > 250:
            self.vola_adjust = 0.25
        elif vola > 0.025 or abs(qty) >= 150 or abs(single_qty) >= 150:
            self.vola_adjust = 0.5
        else:
            self.vola_adjust = 1.

        if qty > 0: 
            buy_qty = THETA if qty < THETA*5 else 0
            sell_qty = THETA if qty < THETA*4 else THETA*2
            buy_qty = buy_qty*self.vola_adjust
            if abs(single_qty) <= THETA/2:
                sell_qty += abs(single_qty)

        else:
            buy_qty = THETA if -qty < THETA*4 else THETA*2
            sell_qty = THETA if -qty < THETA*5 else 0
            sell_qty = sell_qty*self.vola_adjust
            if qty == 0:
                buy_qty = buy_qty*self.vola_adjust
            if abs(single_qty) <= THETA/2:
                buy_qty += abs(single_qty)

        # if qty < 0:
        #     if (abs(qty)/THETA) < 1.5:
        #         buy_qty = abs(qty)
        #     else:
        #         buy_qty = THETA*np.exp(-ETA2*qty)
        #     sell_qty = THETA*np.exp(ETA*qty)
        # else:
        #     if (abs(qty)/THETA) < 1.5:
        #         sell_qty = -qty
        #     else:
        #         sell_qty = THETA*np.exp(ETA2*qty)
        #     buy_qty = THETA*np.exp(-ETA*qty)
        return int(round(buy_qty)), int(round(sell_qty))

    def truncate(self, number, digits) -> float:
        stepper = pow(10.0, digits)
        return math_lib.trunc(stepper * number) / stepper

    def get_order_imbalance(self):
        of = 0
        try:
            rs = self.client.query('SELECT * FROM BITMEX WHERE time >= now()-5s AND time <= now();')
            # BUG FIX -> Update sides: buy = ask   and sell = bid
            ask_orders = list(rs.get_points(tags={'side':'buy', 'pair':str(self.exchange.symbol)}))
            bid_orders = list(rs.get_points(tags={'side':'sell', 'pair':str(self.exchange.symbol)}))
            of = sum([bid_orders[i]['amount'] for i in range(len(bid_orders))]) - sum([ask_orders[i]['amount'] for i in range(len(ask_orders))])

            print ("="*10)
            print ('Of of {0} = {1}'.format(str(self.exchange.symbol), of))
            print ("="*10)
            self.add_to_influx("order_imbalance"+"+"+str(self.exchange.symbol), float(of))
        except Exception as e:
            print(e)
            of = 0
        return of

    def add_to_influx(self, feed, amount):
        pass
        # json_body = [{'measurement': 'BITMEX', 
        #             'tags': {'pair': 'XBTUSD', 'feed': feed}, 
        #             'time': datetime.datetime.now(), 
        #             'fields': {'timestamp': 'connection', 'amount': amount, 'price': 0.0}}]
        # self.client.write_points(json_body)

    def offload_influx(self):
        self.client.query('DELETE FROM BITMEX WHERE time < now();')

    def adjust_order_imbalance(self, of):
        if self.exchange.symbol == "XBTUSD":
            return min(40, of/(2*OF_LIMIT))
        if self.exchange.symbol == "ETHUSD":
            return min(4, of/OF_LIMIT)

    def one_loop(self):
        #self.ctr += 1
        self.general_ctr += 1
        ticker = self.get_ticker()
        pos = self.exchange.get_delta(self.exchange.symbol)
        ord_list = self.exchange.get_orders()
        self.cur_qty = self.exchange.get_delta(self.exchange.symbol)
        self.single_qty = self.exchange.get_position(self.exchange.symbol)['currentQty']
        self.cur_of = self.get_order_imbalance()

        if self.first == True and pos != 0:
            self.first = False
            self.ctr = 1
        if self.first == True and len(ord_list) != 0:
            self.first = False
            self.ctr = 1

        if self.first:
            self.offload_influx()
            logger.info('Offload Influx -- On start.')

        if self.ctr == 1:
            self.ctr = 0
            self.df = self.df.append(pd.DataFrame({'tick': [ticker['mid']]}), ignore_index = True)
            if len(self.df) > 40:
                # self.write = True
                self.df = self.df.iloc[-120:]
                self.df['ret'] = ((self.df['tick'] - self.df['tick'].shift())/self.df['tick'].shift())*100
                self.df['mean'] = self.df['ret'].rolling(36).apply(np.mean)
                self.df['ma10'] = self.df['tick'].rolling(5).apply(np.mean)
                #print (((self.df['ret'] - self.df['mean'])**2).rolling(36).apply(np.mean))
                self.df['vola'] = ((self.df['ret'] - self.df['mean'])**2).rolling(36).apply(np.mean)
                self.df['vola'] = self.df['vola'].apply(np.sqrt)

                # self.df.loc[self.df.ma10.shift(1) - self.df.ma10 > 0, 'slope'] = -1
                # self.df.loc[self.df.ma10 - self.df.ma10.shift(1) > 0, 'slope'] = 1
                # self.df.loc[self.df.slope.shift(-1) == -1, 'slope'] = -1
                # self.df.loc[self.df.slope.shift(-1) == 1, 'slope'] = 1
                # self.df.loc[self.df.ma10.shift(1) - self.df.ma10 == 0, 'slope'] = 0
                # self.df.loc[self.df.slope.shift(-1) == 0, 'slope'] = 0

                self.cur_volatility = self.df.iloc[-1].vola
                print ("Volatility: ", self.cur_volatility)
                self.slope_ctr += np.sign(self.df.ma10.tolist()[-1] - self.df.ma10.tolist()[-2]) if self.df.ma10.tolist()[-1] != self.df.ma10.tolist()[-2] else -0.5*np.sign(self.slope_ctr)
                print ('*'*10)
                print ('Symbol - {}, Current Slope -  {}'.format(self.exchange.symbol, self.slope_ctr))
                print ('*'*10)
                logger.info('Five secs -- Volatility: {}, Trade Volatility: {}, Order Imbalance: {}'.format(self.cur_volatility, self.act_volatility, self.cur_of))
            if self.first:
                print (self.df.tail(5))
                print (self.df.iloc[-1], self.df.iloc[-1].tick, self.df.iloc[-2], self.df.iloc[-2].tick)
            if self.to_record_vola:
                self.act_volatility = self.cur_volatility
                self.to_record_vola = False

        if len(ord_list) > 2:
            self.exchange.cancel_all_orders()

        if self.general_ctr == 180:
            logger.info('RAN FOR 15 mins. Clean database')
            self.general_ctr = 0
            self.offload_influx()

        if abs(self.slope_ctr) >= 6 and self.plus == True:
            self.slope_ctr += np.sign(self.slope_ctr)*8
            self.plus = False
        if abs(self.slope_ctr) <= 1:
            self.plus = True

        # if (self.df.iloc[-1].tick == self.df.iloc[-2].tick) and (self.df.iloc[-3].tick == self.df.iloc[-2].tick):
        #     print ('Repetition! RESTART TRIGGERING: ', [self.df.iloc[-1].tick, self.df.iloc[-2].tick, self.df.iloc[-3].tick])
        #     logger.info('Repetition! RESTART TRIGGERING')
        #     self.restart()

        # no need for automatic restart for now
        # if self.general_ctr == 2880:
        #     print ('RAN FOR 12 HRS! RESTART TRIGGERING')
        #     logging.info('RAN FOR 12 HRS! RESTART TRIGGERING')
        #     self.general_ctr = 0
        #     self.restart()

        self.cur_len = len(ord_list)
        if (self.cur_len == self.prev_len) and (self.cur_len > 0): # could incur errors
            self.idle += 1
        elif (self.cur_len < self.prev_len):
            print ("RAPID FILLED: ", self.cur_len, self.prev_len)
            self.streak += 1
            self.idle = 0 #wont use idle for now
        else:
            self.idle = 0

        print ('==='*10)

        print ("{}: {}".format(self.exchange.symbol, self.cur_volatility))
        print ('==='*10)
        logger.debug('Subminute -- Mid price {}; Position Size: {}; OrderList: {}; OrderLength: {}'.format(ticker['mid'], pos, ord_list, len(ord_list)))
        #logger.info("Current Order list: ", ord_list, self.cur_len)

        if self.act_volatility != None: #abrupt change in volatility
        # AND MID NOT NEARBY LIMIT ORDER PRICE
            cond1 = self.cur_volatility > self.act_volatility*1.25
            cond2 = self.cur_volatility < self.act_volatility*.75
        else:
            cond1 = cond2 = False

        cond3 = (self.cur_volatility != None) and (self.first) # no order placed before + enough data to calc volatility
        cond4 = (ord_list != None) and (ord_list != []) and (len(ord_list) < 2) #and (self.cur_len < self.prev_len) # 1 order just filled --> left 1 order on the other side
        #cond5 = (self.idle == 60) # if orders don't get filled for too long
        cond5 = False
        cond6 = (ord_list == [] and self.first == False) # no orders after the first trade
        cond7 = (ord_list != None) and (ord_list != []) and (len(ord_list) < 2) and (ord_list[0]['side'] == 'Buy') and (pos != 0) and (pos > 0) # 1 order left + on the same side of the pos
        cond8 = (ord_list != None) and (ord_list != []) and (len(ord_list) < 2) and (ord_list[0]['side'] == 'Sell') and (pos != 0) and (pos < 0) # 1 order left + on the same side of the pos
        cond9 = (len(ord_list) >= 10)

        cond10 = False
        cond11 = self.prev_qty != None and self.cur_qty != self.prev_qty

        if self.of != None and self.cur_of != None and self.of != 0 and self.cur_of != 0:
            cond10 = ((self.cur_of - self.of) >= 30000) and (max(self.cur_of, self.of)/ min(self.cur_of, self.of) >= 1.2)
        elif abs(self.cur_of > 0) and self.of == 0:
            cond10 = True

        if self.streak == 3:
            logger.debug('Sleep to prevent successive market orders.') ## TODO: not working correctly
            cond4 = False
            self.streak = 0
            self.sleep_ctr += 1

        if abs(self.slope_ctr) >= 6 and np.sign(pos) != np.sign(self.slope_ctr):
            cond1 = False
            cond2 = False
            cond3 = False
            cond4 = False
            cond5 = False
            cond6 = False
            cond7 = False
            cond8 = False
            cond9 = False
            cond10 = False
            logger.info('Caught in Trend. Stop Trading!')
        if abs(self.slope_ctr) >= 6 and np.sign(pos) == np.sign(self.slope_ctr):
            if (abs(self.slope_ctr) % 2 == 0) or ((abs(self.slope_ctr) - 0.5) % 2 == 0):
                logger.info('Intermittently checking on inventory.')
                cond4 = True

        logger.debug('assess conditions: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond10))
        if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8 or cond9 or cond10 or cond11:
            if cond3:
                logger.info('First Trade!')
                self.first = False
            if cond4 or cond1 or cond2 or cond7 or cond8 or cond9 or cond10 or cond11: # TODO: case work to specify which factor influences the revise
                logger.info('Revise')
                #self.client.Order.Order_cancelAll().result()
            if cond5:
                logger.info('Idle')
                self.idle = 0
            r, spread = self.calc_res_price(ticker["mid"], pos, self.cur_volatility)
            other_r, other_spread = self.calc_res_price(ticker["mid"], self.single_qty, self.cur_volatility, MAX_POS*5)
            print ('Real mid: ', r)
            print ('Spread: ', spread)
            buy_qty, sell_qty = self.get_qty(pos, self.cur_volatility)
            other_buy_qty, other_sell_qty = self.get_qty(self.single_qty, self.cur_volatility)
            buy_qty = max(buy_qty, other_buy_qty)
            sell_qty = max(sell_qty, other_sell_qty)

            if buy_qty != 0 or sell_qty != 0:
                self.place_orders(other_spread, other_r, spread, r, ticker["mid"], buy_qty, sell_qty, pos)
                self.act_volatility = self.cur_volatility
                self.cur_len += bool(buy_qty) + bool(sell_qty)
                logger.info('Orders post: {}, {}, {}, {}'.format(r, spread, buy_qty, sell_qty))
        else:
            pass

        self.prev_len = self.cur_len
        self.prev_qty = self.cur_qty

    def round_to(self, n, precision):
        correction = 0.5 if n >= 0 else -0.5
        return int( n/precision+correction ) * precision

    def round_to_05(self, n):
        return self.round_to(n, 0.05)

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return float('.'.join([i, (d+'0'*n)[:n]]))

    def place_orders(self, other_spread, other_mid, spread, mid, exchange_mid, buy_qty, sell_qty, pos):
        """Create order items for use in convergence."""

        buy_orders = []
        sell_orders = []
        # Create orders from the outside in. This is intentional - let's say the inner order gets taken;
        # then we match orders from the outside in, ensuring the fewest number of orders are amended and only
        # a new order is created in the inside. If we did it inside-out, all orders would be amended
        # down and a new order would be created at the outside.

        getcontext().prec = 4
        # if pos > 0: # change so that we always send POSTONLY orders
        #     buy = {'orderQty': buy_qty, 'price': self.round_to_05(float(Decimal(mid) - Decimal(spread)/Decimal(2))), 'side': 'Buy', 'execInst': 'ParticipateDoNotInitiate'}
        #     sell = {'orderQty': sell_qty, 'price': self.round_to_05(float(Decimal(mid) + Decimal(spread)/Decimal(2))), 'side': 'Sell', 'execInst': 'ParticipateDoNotInitiate'}
        # elif pos < 0:
        #     buy = {'orderQty': buy_qty, 'price': self.round_to_05(float(Decimal(mid) - Decimal(spread)/Decimal(2))), 'side': 'Buy', 'execInst': 'ParticipateDoNotInitiate'}
        #     sell = {'orderQty': sell_qty, 'price': self.round_to_05(float(Decimal(mid) + Decimal(spread)/Decimal(2))), 'side': 'Sell', 'execInst': 'ParticipateDoNotInitiate'}
        buy_price = max(self.round_to_05(float(Decimal(mid) - Decimal(spread)/Decimal(2))), self.round_to_05(float(Decimal(other_mid) - Decimal(other_spread)/Decimal(2))))
        sell_price = min(self.round_to_05(float(Decimal(mid) + Decimal(spread)/Decimal(2))), self.round_to_05(float(Decimal(other_mid) + Decimal(other_spread)/Decimal(2))))

        print('Before adjustments: Buy: {}; Sell: {}; Mid: {}'.format(buy_price,sell_price, float(Decimal(exchange_mid))))

        if self.exchange.symbol == 'ETHUSD':
            if self.truncate(buy_price, 2) >= exchange_mid:
                logger.info('Buy -- Revise Price for POSTONLY')
                buy_price = float(exchange_mid - 0.05)
            if self.truncate(sell_price, 2) <= exchange_mid:
                logger.info('Sell -- Revise Price for POSTONLY')
                sell_price = float(exchange_mid + 0.05)

        if self.exchange.symbol == 'XBTUSD':
            if self.truncate(buy_price, 2) >= exchange_mid:
                logger.info('Buy -- Revise Price for POSTONLY')
                buy_price = float(exchange_mid - 0.5)
            if self.truncate(sell_price, 2) <= exchange_mid:
                logger.info('Sell -- Revise Price for POSTONLY')
                sell_price = float(exchange_mid + 0.5)


        print('After adjustments: Buy: {}; Sell: {}; Mid: {}'.format(buy_price,sell_price, float(Decimal(exchange_mid))))


        if self.cur_of > 0:
            sell_price = self.round_to_05(sell_price + self.adjust_order_imbalance(self.cur_of))
        else:
            buy_price = self.round_to_05(buy_price + self.adjust_order_imbalance(self.cur_of))

        if self.exchange.symbol == "XBTUSD":
            buy = {'orderQty': buy_qty, 'price': self.round_to(buy_price, 0.5), 'side': 'Buy', 'execInst': 'ParticipateDoNotInitiate'}
            sell = {'orderQty': sell_qty, 'price': self.round_to(sell_price, 0.5), 'side': 'Sell', 'execInst': 'ParticipateDoNotInitiate'}

        elif self.exchange.symbol == "ETHUSD":
            buy = {'orderQty': buy_qty, 'price': buy_price, 'side': 'Buy', 'execInst': 'ParticipateDoNotInitiate'}
            sell = {'orderQty': sell_qty, 'price': sell_price, 'side': 'Sell', 'execInst': 'ParticipateDoNotInitiate'}

        if buy_qty == 0:
            sell_orders.append(sell)
        elif sell_qty == 0:
            buy_orders.append(buy)
        else:
            sell_orders.append(sell)
            buy_orders.append(buy)
        print ('Buy: {}; Sell: {}'.format(buy['price'], sell['price']))
        logger.info('Buy: {}; Sell: {}'.format(buy['price'], sell['price']))

        self.of = self.cur_of
        os.system("""say 'Please be alerted. New orders posted to bitmex.' """)
        return self.converge_orders(buy_orders, sell_orders, spread, mid, exchange_mid, buy_qty, sell_qty, pos)

    def converge_orders(self, buy_orders, sell_orders, spread, mid, exchange_mid, buy_qty, sell_qty, pos):
        """Converge the orders we currently have in the book with what we want to be in the book.
           This involves amending any open orders and creating new ones if any have filled completely.
           We start from the closest orders outward."""

        tickLog = self.exchange.get_instrument()['tickLog']
        to_amend = []
        to_create = []
        to_cancel = []
        buys_matched = 0
        sells_matched = 0
        existing_orders = self.exchange.get_orders()

        # Check all existing orders and match them up with what we want to place.
        # If there's an open one, we might be able to amend it to fit what we want.
        for order in existing_orders:
            try:
                if order['side'] == 'Buy':
                    desired_order = buy_orders[buys_matched]
                    buys_matched += 1
                else:
                    desired_order = sell_orders[sells_matched]
                    sells_matched += 1

                amend_cond_1 = self.truncate(float(desired_order['orderQty']),2) != self.truncate(float(order['leavesQty']), 2)
                amend_cond_2 = self.truncate(float(desired_order['price']), 2) != self.truncate(float(order['price']), 2)
                
                if amend_cond_1:
                    print (self.truncate(float(desired_order['price']), 2), self.truncate(float(order['price']), 2))
                if amend_cond_2:
                    print(self.truncate(float(desired_order['orderQty']),2), self.truncate(float(order['leavesQty']), 2))
                logger.debug('Desired Orders: {}'.format(desired_order))
                logger.debug('Amend Conditions: {}, {}'.format(amend_cond_1, amend_cond_2))
                # Found an existing order. Do we need to amend it?
                if (amend_cond_1) or (amend_cond_2):
                    try:
                        to_amend.append({'orderID': order['orderID'], 'orderQty': desired_order['orderQty'],
                                        'price': desired_order['price'], 'side': order['side'], 'execInst': desired_order['execInst']})
                    except:
                        logger.info('Unknown error to amending.')

            except IndexError:
                # Will throw if there isn't a desired order to match. In that case, cancel it.
                to_cancel.append(order)

        while buys_matched < len(buy_orders):
            to_create.append(buy_orders[buys_matched])
            buys_matched += 1

        while sells_matched < len(sell_orders):
            to_create.append(sell_orders[sells_matched])
            sells_matched += 1

        logger.debug('To Amend: {}; Len amend: {} \n To Create: {}, Len Create: {}'.format(to_amend, len(to_amend), to_create, len(to_create)))

        if len(to_amend) > 0:
            for amended_order in reversed(to_amend):
                reference_order = [o for o in existing_orders if o['orderID'] == amended_order['orderID']][0]
                logger.info("Amending %4s: %d @ %.*f to %d @ %.*f (%+.*f)" % (
                    amended_order['side'],
                    reference_order['leavesQty'], tickLog, reference_order['price'],
                    (amended_order['orderQty'] - reference_order['cumQty']), tickLog, amended_order['price'],
                    tickLog, (amended_order['price'] - reference_order['price'])
                ))
            # This can fail if an order has closed in the time we were processing.
            # The API will send us `invalid ordStatus`, which means that the order's status (Filled/Canceled)
            # made it not amendable.
            # If that happens, we need to catch it and re-tick.
            try:
                self.exchange.amend_bulk_orders(to_amend)
            except requests.exceptions.HTTPError as e:
                errorObj = e.response.json()
                if errorObj['error']['message'] == 'Invalid ordStatus':
                    logger.warn("Amending failed. Waiting for order data to converge and retrying.")
                    sleep(5)
                    return self.place_orders(spread, mid, exchange_mid, buy_qty, sell_qty, pos)
                else:
                    logger.error("Unknown error on amend: %s. Exiting" % errorObj)
                    sys.exit(1)

        if len(to_create) > 0:
            logger.info("Creating %d orders:" % (len(to_create)))
            for order in reversed(to_create):
                logger.info("%4s %d @ %.*f" % (order['side'], order['orderQty'], tickLog, order['price']))
            self.exchange.create_bulk_orders(to_create)

        # Could happen if we exceed a delta limit
        if len(to_cancel) > 0:
            logger.info("Canceling %d orders:" % (len(to_cancel)))
            for order in reversed(to_cancel):
                logger.info("%4s %d @ %.*f" % (order['side'], order['leavesQty'], tickLog, order['price']))
            self.exchange.cancel_bulk_orders(to_cancel)

    def sanity_check(self):
        """Perform checks before placing orders."""

        # Check if OB is empty - if so, can't quote.
        self.exchange.check_if_orderbook_empty()

        # Ensure market is still open.
        self.exchange.check_market_open()

    def check_file_change(self):
        """Restart if any files we're watching have changed."""
        for f, mtime in watched_files_mtimes:
            if getmtime(f) > mtime:
                print('Files Changed')
                self.exit() # The main file starts it

    def check_connection(self):
        """Ensure the WS connections are still open."""
        return self.exchange.is_open()

    def exit(self):
        logger.info("Shutting down.")
        try:
            #self.exchange.cancel_all_orders() ## TODO: no longer close orders.
            self.exchange.bitmex.exit()
        except errors.AuthenticationError as e:
            logger.info("Was not authenticated; could not cancel orders.")
        except Exception as e:
            logger.info("Unable to cancel orders: %s" % e)

        print ('Dropping database.')
        self.client.drop_database('example')
        print('Final Scripts Completed')
        os._exit(1) # Ends the script without raising exception
        # sys.exit()

    def run_loop(self):
        threading.Timer(5.0, self.run_loop).start()
        sys.stdout.write("-----\n")
        sys.stdout.flush()

        self.ctr += 1

        self.check_file_change()
        #sleep(settings.LOOP_INTERVAL)

            # This will restart on very short downtime, but if it's longer,
            # the MM will crash entirely as it is unable to connect to the WS on boot.
        if not self.check_connection():
            logger.error("Realtime data connection unexpectedly closed, restarting.")
            self.add_to_influx('connection', float(0))
            self.exit()
            #self.restart()
        elif self.check_connection():
            self.add_to_influx('connection', float(1))

        self.sanity_check()  # Ensures health of mm - several cut-out points here
        self.print_status()  # Print skew, delta, etc

        self.one_loop()  # Creates desired orders and converges to existing orders

    def restart(self):
        logger.info("Restarting the market maker...")
        #os.execv(sys.executable, [sys.executable] + sys.argv)
        os.system("python3.6 -m market_maker.market_maker")
        sys.exit()

def XBt_to_XBT(XBt):
    return float(XBt) / constants.XBt_TO_XBT

# todo: helper function: convert bitmex symbol to ccxt symbol
def run():
    logger.info('BitMEX Market Maker Version: %s\n' % constants.VERSION)

    om = OrderManager()
    #print("print attributes: ", om.__dict__)
    # Try/except just keeps ctrl-c from printing an ugly stacktrace
    try:
        om.run_loop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()

if __name__ == "__main__":
    current_hour = datetime.datetime.now().hour
    os.environ['TZ'] = 'Asia/Saigon'
    time.tzset() # only available in Unix
run()
