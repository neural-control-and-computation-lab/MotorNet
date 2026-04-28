import torch as th
import matplotlib.pyplot as plt

def policy_mod(phase):
# During "growing_up" phase: train everything
    if phase == 'growing_up':
        input_freeze = 0
        output_freeze = 0
        optimizer_mod = 'Adam'
        learning_rate = 1e-3 #1e-3
# In other phases: freeze both input and output connections
    else:
        input_freeze = 1
        output_freeze = 1
        # optimizer_mod = 'SGD'
        # learning_rate = 1e-5
        optimizer_mod = 'Adam'
        learning_rate = 1e-3#1e-3

    return input_freeze, output_freeze, optimizer_mod, learning_rate


# Apply a curl force field
def applied_load(endpoint_vel, k, mode = 'CW'):
    # Curved Force
    if mode == 'CW':
        curl_matrix = th.tensor([[0., -1.], [1., 0.]])  # Clockwise
    elif mode == 'CCW':
        curl_matrix = th.tensor([[0., 1.], [-1., 0.]])   # Counterclockwise
    else:
        curl_matrix = th.tensor([[0., 0.], [0., 0.]])
    force_field = k * endpoint_vel @ curl_matrix

    return force_field


# The next two functions are for comparing output weights between two different phases
def eval_weights(saveLoc, phase, exp, batch_number):
    weights = th.load(saveLoc + f'weightsDic_{phase}_{exp}')
    print(weights.keys())
    state = weights[f'{batch_number}']
    # state = th.load(saveLoc + f'weights_{phase}_{exp}')
    for name, param in state.items():
        print(name, param.shape)
    return state



def compare_weights(state1, state2):
    weight = 'Wh'
    bias = 'bh'
    type = 'hidden'
    Wout1, bout1 = state1[f'{weight}'], state1[f'{bias}']
    Wout2, bout2 = state2[f'{weight}'], state2[f'{bias}']
    Wout1_masked, bout1_masked = state1[f'mask_{weight}'], state1[f'mask_{bias}']
    Wout2_masked, bout2_masked = state2[f'mask_{weight}'], state2[f'mask_{bias}']

    plt.hist(Wout1.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight value')
    plt.ylabel('Number of weights')
    plt.title(f'Distribution of {type} weights - phase 1')
    plt.show()

    plt.hist(Wout2.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight value')
    plt.ylabel('Number of weights')
    plt.title(f'Distribution of {type} weights - phase 2')
    plt.show()

    # Elementwise difference
    diff_Wout = Wout1 - Wout2
    diff_bout = bout1 - bout2

    print("Weight difference (mean abs):", diff_Wout.abs().mean().item())
    print("Weight difference (max abs):", diff_Wout.abs().max().item())
    #print("Bias difference:", diff_bout)

    plt.hist(diff_Wout.flatten().cpu().numpy(), bins=50, color='skyblue')
    plt.xlabel('Absolute weight difference')
    plt.ylabel('Number of weights')
    plt.title(f'Distribution of {type} weight differences')
    plt.xticks(fontsize=6)
    plt.show()

    plt.imshow(diff_Wout.cpu().numpy())
    plt.colorbar(label="Absolute weight difference")
    plt.show()

    return Wout1, Wout2, Wout1_masked, Wout2_masked, diff_Wout, diff_bout



if __name__ == "__main__":
    saveLoc = '/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version/task_0/'
    # Compare weights between the following two phases. This is especially useful to verify that freezing worked as expected
    phase1 = 'FF1'
    exp1 = 'center_out'
    batch_number1 = '0'
    state1 = eval_weights(saveLoc, phase1, exp1, batch_number1)
    phase2 = 'FF1'
    exp2 = 'center_out'
    batch_number2 = '1'
    state2 = eval_weights(saveLoc, phase2, exp2, batch_number2)
    compare_weights(state1, state2)





