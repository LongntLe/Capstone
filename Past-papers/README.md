## Overview
This folder contains my code for running the Avellaneda & Stoikov model live on BitMEX.

The files contains in this folder are not to be run alone. They need to be supplied to [BitMEX Sample Market Maker](https://github.com/BitMEX/sample-market-maker), where they will overwrite their default settings and classes. The guidelines for running the market maker, as well as customed strategies are contained in the mentioned repo. 

## Adaptations for the cryptocurrency market

To apply the model to the cryptocurrency market, as well as to improve its practicality, I have made some changes to the true price and spread calculation functions:

* Add a term to the true mid calculation formula, that takes into account the current return in relation to past returns in a short window. If the most recent return is more than 1 standard deviation away from the mean return in the past minute, shifting the projected mid price to favor liquidating current inventory.
* Add new functions to the OrderManager class to truncate and round order size and price such that they satisfy the conditions for order size and price set by BitMEX (i.e., BTC order price needs to be rounded by 0.5 and order size must be an integer).
* Scaling the mid and position formulas by a max_pos constant so the bot would prioritize liquidating inventory if a max pos is reached.
