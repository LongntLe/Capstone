## Overview

This folder contains my codes for each variation of MPC presented in my paper

* Toy MPC: native MPC with a linear regression to project price movement
* Stochastic toy MPC: stochastic formulation of the objective function, but instead of finding a universal set of control U, I find an optimal set of control u for each projected time series, then average out the sets of control.
* Scenario-based MPC: stochastic formulation of the objective function, to find only one control that satisfies multiple projected time series at the same time.
