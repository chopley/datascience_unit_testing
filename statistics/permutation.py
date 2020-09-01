class permutation_test:
    import copy
    import random
    import numpy as np
    import time

    @classmethod
    def test_collection_rate_function(cls, a, b, kwargs):
        """
        Helper function for get_collection_rate
        """
        import pandas as pd
        import numpy as np

        time_column = kwargs["time_column"]
        df_a = pd.DataFrame(a.groupby(time_column).sum()).reset_index()
        df_b = pd.DataFrame(b.groupby(time_column).sum()).reset_index()
        df_a["collection_rate_a"] = 100 * (
            df_a["collected_during"] / df_a["billed_during"]
        )
        df_b["collection_rate_b"] = 100 * (
            df_b["collected_during"] / df_b["billed_during"]
        )
        df_c = df_a.merge(df_b, left_on=["time_passed"], right_on=["time_passed"])
        df_c["diff"] = np.abs(df_c["collection_rate_a"] - df_c["collection_rate_b"])
        test_stat = df_c["diff"].sum()
        return test_stat

    @classmethod
    def get_collection_rate(cls, dataframe, n_randomizations, kwargs):
        """
        Function to handle returning permutation test for collection rate
        First calculate the collection rate for each time period, and sum the absolute values
        for the un-randomized case- the test statistics is the difference between these values for the
        two experimental conditions
        Then shuffle, and recalculate the above.
        """
        import time
        import sys
        import copy
        import numpy as np

        group_columns = kwargs["group_columns"]
        group_values = kwargs["group_values"]
        time_column = kwargs["time_column"]

        keep_columns = group_columns
        keep_columns.append(time_column)
        keep_columns.append("collected_during")
        keep_columns.append("billed_during")
        print(keep_columns)
        # only keep the columns we need to speed up the shuffling...
        dataframe = dataframe[keep_columns].copy()
        # get the a experimental condition as defined by the group_column:group_value
        a = dataframe[(dataframe[group_columns[0]] == group_values[0])].copy()
        # get the b experimental condition as defined by the group_column:group_value
        b = dataframe[(dataframe[group_columns[0]] == group_values[1])].copy()
        gT = permutation_test.test_collection_rate_function(a, b, kwargs)

        pS = copy.copy(
            dataframe[
                (dataframe[group_columns[0]] == group_values[0])
                | (dataframe[group_columns[0]] == group_values[1])
            ]
        )
        pD = []
        start_time = time.time()

        for i in range(0, n_randomizations):
            pS = pS.sample(frac=1).reset_index(drop=True)
            a = pS[0 : int(len(pS) / 2)].copy()
            b = pS[int(len(pS) / 2) :].copy()
            pD.append(permutation_test.test_collection_rate_function(a, b, kwargs))
            time_now = time.time()
            time_cycle = (time_now - start_time) / (i + 1)
            estimated_time_remaining = (n_randomizations - (i + 1)) * time_cycle
            f = (
                "Time [ms] per shuffle: "
                + str(np.round(time_cycle * 1000))
                + " Estimated time remaining: "
                + str(np.round(estimated_time_remaining, 0))
            )
            sys.stdout.write("\r" + str(f))
            sys.stdout.flush()

        p_val = len(np.where(pD >= gT)[0]) / n_randomizations
        return (gT, p_val, pD)

    def permutation_test_statistic_time_series(
        dataframe, n_randomizations, test_function, **kwargs
    ):
        (gT, p_val, pD) = test_function(dataframe, n_randomizations, kwargs)

        return (gT, p_val, pD)

def generate_simulated_repayments(percentages, probabilities, amounts):
    """
    Function to simulate a repayment probability distribution
    Parameters:
    :arg1 percentages(list) - list of ratio of repayment percentage- e.g. fully repaid is 1 and no repayment =0
    :arg2 probabilities(list) - list of the associated probability of all the repayment percentages- this must sum to 1
    :arg3 amounts(list) - list of amounts that need to have the repayment distribution applied to them

    Returns:
    :list of repayment amounts based on the indicated distributions from the input arguments
    Examples:
    Simulate a repayment rate of 60% of invoices fully paid
    ---- repayments = generate_simulated_repayments([0,1],[0.4,0.6],billed_amounts)
    Simulate a repayment rate of
    - 1. 60% of invoices fully paid
    - 2. 30% of invoices unpaid
    - 3. 10% of invoices half-paid
    ----repayments = generate_simulated_repayments([0,0.5,1],[0.3,0.5,0.6],billed_amounts)
    """
    from numpy.random import choice
    repayment = []
    for amount in amounts:
        draw = choice(percentages, 1, p=probabilities)
        repayment.append(amount * draw[0])
    return repayment