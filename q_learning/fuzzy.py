def fuzzify_rising(value: float, start: float, end: float) -> float:
    if value > end:
        return 1.0
    elif value < start:
        return 0.0
    else:
        return (value - start) / (end - start)


def fuzzify_falling(value: float, start: float, end: float) -> float:
    return f_not(fuzzify_rising(value, start, end))


def fuzzify_triangle(value: float, start: float, mid: float, end: float) -> float:
    if value < mid:
        return fuzzify_rising(value, start, mid)
    else:
        return fuzzify_falling(value, mid, end)


def f_not(value: float) -> float:
    return 1.0 - value


def defuzzify(value, zero, one):
    if zero > one:
        return defuzzify(f_not(value), one, zero)
    else:
        return zero + value * (one - zero)

