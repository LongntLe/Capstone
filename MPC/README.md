## Overview

This folder contains my codes for each variation of MPC presented in my paper

* Toy MPC: native MPC with a linear regression to project price movement
* Stochastic toy MPC: stochastic formulation of the objective function, but instead of finding a universal set of control U, I find an optimal set of control u for each projected time series, then average out the sets of control.
* Scenario-based MPC: stochastic formulation of the objective function, to find only one control that satisfies multiple projected time series at the same time.

This folder also contains a iPython notebook to run the models interactively, as well as to try the idea of adding a position layer to the MPC. For the notebook, please import the data from data (instead of running the data import line as-is).

self-#critique: I did not generalize frequently used command sequences such as plotting to be functions. I also did not specify variables for users to modify parameters -- parameters are hard-coded into the code. Apologies that I did not incorporate these into the code on time.
