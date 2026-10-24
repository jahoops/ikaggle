def normalize_value(raw_value, norm_offset, norm_scale):
    """
    Normalize a raw value to the [0, 1] range using the provided offset and scale.
    
    Arguments:
        raw_value (float): The original value.
        norm_offset (float): The minimum (offset) value of the original scale.
        norm_scale (float): The range (max - min) of the original scale.
        
    Returns:
        float: Normalized value between 0 and 1.
    """
    return (raw_value - norm_offset) / norm_scale

def denormalize_value(norm_value, norm_offset, norm_scale):
    """
    Convert a normalized value [0, 1] back to its original scale using the provided offset and scale.
    
    Arguments:
        norm_value (float): The normalized value between 0 and 1.
        norm_offset (float): The minimum (offset) value of the original scale.
        norm_scale (float): The range (max - min) of the original scale.
        
    Returns:
        float: The value in the original scale.
    """
    return norm_value * norm_scale + norm_offset
