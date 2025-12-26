from datetime import datetime

def xirr(cashflows, dates, guess=0.1):
    """
    Newton-Raphson XIRR implementation.
    cashflows: list of floats (negative = investment, positive = redemption)
    dates: list of datetime objects
    """
    if len(cashflows) != len(dates):
        raise ValueError("Cashflows and dates must have same length")

    # Convert dates to day offsets
    days = [(d - dates[0]).days for d in dates]

    def npv(rate):
        return sum(cf / ((1 + rate) ** (day / 365)) for cf, day in zip(cashflows, days))

    def d_npv(rate):
        return sum(
            -(day / 365) * cf / ((1 + rate) ** (1 + day / 365))
            for cf, day in zip(cashflows, days)
        )

    rate = guess
    for _ in range(100):
        f = npv(rate)
        df = d_npv(rate)
        if df == 0:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < 1e-6:
            return new_rate
        rate = new_rate

    return rate
