def format_indian(n):
    n = float(n)
    if abs(n) >= 1_00_00_000:
        return f"{n/1_00_00_000:.2f} Cr"
    elif abs(n) >= 1_00_000:
        return f"{n/1_00_000:.2f} L"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{n:.0f}"
