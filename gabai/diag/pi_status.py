"""Raspberry Pi status helpers. Standard library only, so they also run on the Pi."""

THROTTLE_BITS = {
    "under_voltage_now": 0x1, "freq_capped_now": 0x2, "throttled_now": 0x4, "soft_temp_limit_now": 0x8,
    "under_voltage_since_boot": 0x10000, "freq_capped_since_boot": 0x20000, "throttled_since_boot": 0x40000,
    "soft_temp_limit_since_boot": 0x80000,
}


def decode_throttled(value):
    """`vcgencmd get_throttled` value ("0x50005" or an int) -> {flag: bool}, or None when it was not reported."""
    if value is None or str(value).strip() == "":
        return None
    try:
        x = int(str(value), 16) if isinstance(value, str) else int(value)
    except ValueError:
        return None
    return {k: bool(x & bit) for k, bit in THROTTLE_BITS.items()}


def active_flags(value):
    """Names of the flags that are set right now or since boot, as a list ([] when clear, None when unknown)."""
    d = decode_throttled(value)
    return None if d is None else [k for k, v in d.items() if v]
