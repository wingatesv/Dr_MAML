import math

def half_trapezoidal_step_scheduler(total_epochs, current_epoch, max_step, min_step, max_step_width, half_right=True):

    max_step_start_epoch = math.floor(total_epochs - max_step_width*total_epochs) if half_right else math.floor(max_step_width*total_epochs)

    max_step_size = total_epochs - max_step_start_epoch if half_right else max_step_start_epoch

    step_freq = math.floor(max_step_start_epoch / (max_step - 1)) if half_right else math.floor((total_epochs - max_step_start_epoch) / (max_step-1))

    total_step_freq = (max_step - 1) * step_freq

    if total_step_freq < (total_epochs - max_step_size):
        step_diff = total_epochs - total_step_freq - max_step_size
        max_step_size += step_diff

    if half_right:
      # Determine the current step based on the current epoch
      if current_epoch < total_step_freq:
          current_step = min_step + current_epoch // step_freq
      else:
          current_step = max_step
    else:
      # Determine the current step based on the current epoch
      if current_epoch < max_step_size:
          current_step = max_step
      else:
          current_step = (max_step - 1) - (current_epoch - max_step_size) // step_freq

    return current_step
    
def new_trapezoidal_step_scheduler(total_epochs, current_epoch, max_step, min_step, max_step_width):
    step_diff = 0
    # Calculate the midpoint of total_epochs and max_step_width
    epoch_midpoint = total_epochs // 2
    max_step_width_midpoint = max_step_width / 2

    # Calculate the frequency of half max step and the starting and ending epochs of max step frequency
    half_max_step_freq = math.floor(total_epochs * max_step_width_midpoint)
    max_step_freq_start_epoch = epoch_midpoint - half_max_step_freq
    max_step_freq_end_epoch = epoch_midpoint + half_max_step_freq

    # Calculate the size of max step and total step frequency
    max_step_size = max_step_freq_end_epoch - max_step_freq_start_epoch
    total_step_freq_size = total_epochs - max_step_size

    # Calculate the size of half step frequency and the frequency of step
    half_step_freq_size = total_step_freq_size // 2
    step_freq = math.floor(half_step_freq_size / (max_step - 1))

    # Calculate the total step frequency and adjust max step size if necessary
    total_step_freq = (max_step - 1) * step_freq * 2
    if total_step_freq < (total_epochs - max_step_size):
        step_diff = total_epochs - total_step_freq - max_step_size
        max_step_size = max_step_size + step_diff if max_step_width != 0 else max_step_size

    # Calculate the new half step frequency size
    new_half_step_freq_size = total_step_freq // 2

    if max_step_width == 0 and step_diff > 0:
         # Determine the current step based on the current epoch
        if current_epoch < new_half_step_freq_size:
            current_step = min_step + current_epoch // step_freq
        elif current_epoch < (new_half_step_freq_size + step_diff):
            current_step = (max_step - 1)
        else:
            current_step = (max_step - 1) - (current_epoch - step_freq * (max_step - 1) - step_diff) // step_freq       
    else:
        # Determine the current step based on the current epoch
        if current_epoch < new_half_step_freq_size:
            current_step = min_step + current_epoch // step_freq
        elif current_epoch < (new_half_step_freq_size + max_step_size):
            current_step = max_step
        else:
            current_step = (max_step - 1) - (current_epoch - step_freq * (max_step - 1) - max_step_size) // step_freq

    return current_step

def trapezoidal_step_scheduler(total_epochs, current_epoch, max_adaptation_step, min_adaptation_step,  include_max_phase=True, half=False):
    # Guardrails to ensure correct types and values for the parameters
    assert isinstance(total_epochs, int), "total_epochs must be an integer"
    assert isinstance(current_epoch, int), "current_epoch must be an integer"
    assert isinstance(max_adaptation_step, int), "max_adaptation_step must be an integer"
    assert isinstance(min_adaptation_step, int), "min_adaptation_step must be an integer"
    assert isinstance(include_max_phase, bool), "include_max_phase must be a boolean"
    assert isinstance(half, bool), "half must be a boolean"
    
    assert min_adaptation_step > 0, "min_adaptation_step must be greater than 0"
    assert max_adaptation_step > min_adaptation_step, "max_adaptation_step must be greater than min_adaptation_step"

    # Check total_epochs based on the value of half and include_max_phase
    if not half:
      assert total_epochs >= 12 if include_max_phase else  total_epochs % max_adaptation_step == 0, "total_epochs must be at least 12 if include_max_phase, else total_epochs must be divisible by max_adaptation_step"

    # Determine the trapezoidal_region based on the value of half and include_max_phase
    trapezoidal_region = 2 if half or not include_max_phase else 3

    # Calculate the period and frequencies per step
    period = total_epochs // trapezoidal_region
    freq_per_step = math.floor(period / (max_adaptation_step - 1)) if include_max_phase else math.floor(period / (max_adaptation_step))
    freq_max_step = total_epochs - (trapezoidal_region - 1) * (freq_per_step * (max_adaptation_step - 1))

    # Determine the current step based on the current epoch and whether the max adaptation step phase is included
    if include_max_phase:
        if current_epoch < freq_per_step * (max_adaptation_step - 1):
            return min_adaptation_step + current_epoch // freq_per_step
        elif current_epoch < freq_per_step * (max_adaptation_step - 1) + freq_max_step:
            return max_adaptation_step
        else:
            return (max_adaptation_step - 1) - (current_epoch - freq_per_step * (max_adaptation_step - 1) - freq_max_step) // freq_per_step
    else:
        if current_epoch < freq_per_step * (max_adaptation_step):
            return min_adaptation_step + current_epoch // freq_per_step
        else:
            return max_adaptation_step - (current_epoch - freq_per_step * (max_adaptation_step)) // freq_per_step
