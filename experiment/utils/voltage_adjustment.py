def calcucalculate_voltage_adjustments(curr_psr_error, prev_psr_error):
    """
    Calculate voltage adjustments based on current and previous power splitting ratio errors.
    Args:
        curr_psr_error (dict): Current power splitting ratio errors.
        prev_psr_error (dict): Previous power splitting ratio errors.
    Returns:
        dict: Voltage adjustments for each MZI.
    """
    voltage_adjustments = {}
    for mzi in curr_psr_error.keys():
        # Simple proportional-derivative control for voltage adjustment
        kp = 0.1  # Proportional gain
        kd = 0.05  # Derivative gain

        curr_error = curr_psr_error[mzi]
        prev_error = prev_psr_error.get(mzi, 0)

        adjustment = kp * curr_error + kd * (curr_error - prev_error)
        voltage_adjustments[mzi] = adjustment

    return voltage_adjustments
