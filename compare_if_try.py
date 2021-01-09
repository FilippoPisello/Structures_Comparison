import time
from typing import List
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import  pandas as pd
import numpy as np

class Comparison:
    """
    Class to compare the expected execution time of n functions given two
    different inputs.

    ----------------
    While its use can be more general, this class was designed to test the
    performance of if/else structures vs try/except structures. In particular,
    it is under test how the expected execution time changes given the probability
    of usage of one (if/try) or the other (else/except) branch.
    """

    decimals = [x/100 for x in range(101)]

    def __init__(self, inputs: List, functions: List, labels: List, loops=10**7):
        self.inputs = inputs
        self.functions = functions
        self.labels = labels
        self.loops = loops
        self.avg_times = self.avg_times()

    # --------------------------------------------------------------------------
    # 1 - Main Elements
    # --------------------------------------------------------------------------
    # 1.1 - Methods used in arguments
    # --------------------------
    def avg_times(self):
        """
        Returns a list with execution times for function for bad and good scenario.

        ------------
        The output is a list of len 2. The first subelement is the list of
        average exectution times for each function, given the preferred input.
        The second subelement is the same for the worse input.
        """
        return [[self.avg_fun_time(fun, inp, self.loops) for fun in self.functions]
                for inp in self.inputs]

    # --------------------------
    # 1.2 - Main Methods
    # --------------------------
    def comparison_df(self):
        """
        Returns dataframe with average worst and best execution time of the given
        functions.
        """
        df_dict = {"Function" : self.labels,
                   "Best Exp Time" : self.avg_times[0],
                   "Worst Exp Time" : self.avg_times[1]}
        return pd.DataFrame(df_dict)

    def exp_time_df(self):
        """
        Returns dataframe with a probability column and an expected execution
        time column for each provided function.
        """
        df_dict = {"P(worst case)" : self.decimals}
        time_cols = [self.exp_time(self.avg_times[1][i], self.avg_times[0][i])
                     for i in range(len(self.functions))]
        for label, data in zip(self.labels, time_cols):
            df_dict[label] = data
        df = pd.DataFrame(df_dict)
        return df

    def exp_time_graph(self, filename=None):
        """
        Returns the plot of the functions performance given the changing
        probability.
        """
        # Initiate figure object
        fig = plt.figure(figsize=(8 * 1.5, 4.5 * 1.5), dpi=80)
        # Define labels
        plot = fig.add_subplot()
        plot.set_title("Execution Time Comparison", fontsize=20)
        plot.set_xlabel("Probability of slowest branch", fontsize=15)
        plot.set_ylabel("Execution Time", fontsize=15)

        # Plot the data
        df = self.exp_time_df()
        for label in self.labels:
            plot.plot(df["P(worst case)"], df[label], "-", label=label)

        # Customize axis
        plot.yaxis.set_major_locator(ticker.MaxNLocator(10))
        plot.set_xticks(np.arange(0, 1.1, step=0.1))
        plot.set_xlim([0, 1])

        # Add further details
        fig.legend(loc="lower center", ncol=2, fontsize=9)
        plot.grid(True, axis="both", ls="--")

        if filename is not None:
            fig.savefig(filename)
        return plot

    # --------------------------------------------------------------------------
    # 2 - Worker Methods
    # --------------------------------------------------------------------------
    # 2.1 - Methods used in properties
    # --------------------------
    @staticmethod
    def avg_fun_time(function, argument, loops):
        """
        Returns the average runtime of a function over a given number of loops.
        """
        start_time = time.time()
        for counter in range(loops):
            function(argument)
        return (time.time() - start_time)/loops

    @classmethod
    def exp_time(self, worst_case_time: float, best_case_time: float) -> List:
        """
        Given two potential outcomes, it returns a list with the expected value
        computed at the variation of the probability.

        ------------------
        The function assumes that the two provided outcomes are complete and
        mutually exclusive. The exp value is calculated as follows:
        [p(A) * Value A] + [(1 - p(B)) * Value B], where p ranges from 0 to 1
        with range 100.
        """
        return [p*worst_case_time + (1-p)*best_case_time for p in self.decimals]