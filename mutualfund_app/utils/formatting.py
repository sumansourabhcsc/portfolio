import csv

def format_indian(n):
    n = float(n)
    if abs(n) >= 1_00_00_000:
        return f"{n/1_00_00_000:.2f} Cr"
    elif abs(n) >= 1_00_000:
        return f"{n/1_00_000:.2f} L"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{n:.0f}"


def detect_delimiter(sample_bytes: bytes) -> str:
    """Detect CSV delimiter (comma or semicolon)."""
    sample = sample_bytes.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        # fallback
        return ";" if sample.count(";") > sample.count(",") else ","
