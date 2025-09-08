import torch as th
import matplotlib.pyplot as plt

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


# The next two functions are for comparing output weights between two different phases
def eval_weights(saveLoc, phase):
    state = th.load(saveLoc + f'weights_{phase}')
    for name, param in state.items():
        print(name, param.shape)
    return state



def compare_weights(state1, state2):
    Wout1, bout1 = state1['Y'], state1['bY']
    Wout2, bout2 = state2['Y'], state2['bY']
    Wout1_masked, bout1_masked = state1['mask_Y'], state1['mask_bY']
    Wout2_masked, bout2_masked = state2['mask_Y'], state2['mask_bY']

    plt.hist(Wout1.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight value')
    plt.ylabel('Number of weights')
    plt.title('Distribution of output weights - phase 1')
    plt.show()

    plt.hist(Wout2.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight value')
    plt.ylabel('Number of weights')
    plt.title('Distribution of output weights - phase 2')
    plt.show()

    # Elementwise difference
    diff_Wout = Wout1 - Wout2
    diff_bout = bout1 - bout2

    print("Weight difference (mean abs):", diff_Wout.abs().mean().item())
    print("Weight difference (max abs):", diff_Wout.abs().max().item())
    print("Bias difference:", diff_bout)

    plt.hist(diff_Wout.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight difference')
    plt.ylabel('Number of weights')
    plt.title('Distribution of output weight differences')
    plt.show()

    plt.imshow(diff_Wout.cpu().numpy())
    plt.colorbar(label="Absolute weight difference")
    plt.show()

    return Wout1, Wout2, Wout1_masked, Wout2_masked, diff_Wout, diff_bout



if __name__ == "__main__":
    saveLoc = '/Users/pounemirzazadeh/Motornet/Modular'
    # Compare weights between the following two phases. This is especially useful to verify that freezing worked as expected
    phase1 = 'growing_up'
    state1 = eval_weights(saveLoc, phase1)
    phase2 = 'NF1'
    state2 = eval_weights(saveLoc, phase2)
    compare_weights(state1, state2)




