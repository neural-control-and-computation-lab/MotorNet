import torch as th

def policy_mod(phase):
# During "growing_up" phase: train everything
    if phase == 'growing_up':
        input_freeze = 0
        output_freeze = 0
        optimizer_mod = 'Adam'
        learning_rate = 1e-3
# In other phases: freeze both input and output connections
    else:
        input_freeze = 1
        output_freeze = 1
        # optimizer_mod = 'SGD'
        # learning_rate = 1e-5
        optimizer_mod = 'Adam'
        learning_rate = 1e-3

    return input_freeze, output_freeze, optimizer_mod, learning_rate


# Apply a curl force field
def applied_load(endpoint_vel, k, mode = 'CW'):
    # Curved Force
    if mode == 'CW':
        curl_matrix = th.tensor([[0., -1.], [1., 0.]])  # Clockwise
    else:
        curl_matrix = th.tensor([[0., 1.], [-1., 0.]])   # Counterclockwise
    force_field = k * endpoint_vel @ curl_matrix

    return force_field





