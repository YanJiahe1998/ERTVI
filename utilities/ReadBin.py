import numpy as np

def read_forward_and_jacobian(filename):
    with open(filename, 'rb') as f:
        # Read the number of measurements and parameters
        ##### read none #####
        np.fromfile(f, dtype=np.int32, count=1)
        #####################
        num_ind_meas = np.fromfile(f, dtype=np.int32, count=1)[0]
        num_param = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Allocate arrays for the data
        fcalc = np.zeros(num_ind_meas, dtype=np.float64)
        elec = np.zeros((4, num_ind_meas), dtype=np.int32)
        Jacobian = np.zeros((num_ind_meas,num_param), dtype=np.float64)

        # Read the 'elec' data
        for j in range(4):
            ##### read none #####
            np.fromfile(f, dtype=np.int32, count=2)
            #####################
            elec[j, :] = np.fromfile(f, dtype=np.int32, count=num_ind_meas)
            
        # Read the 'fcalc' data
        ##### read none #####
        np.fromfile(f, dtype=np.float64, count=1)
        #####################
        fcalc = np.fromfile(f, dtype=np.float64, count=num_ind_meas)
        
        # Read the 'Jacobian' data
        for j in range(num_param):
            ##### read none #####
            np.fromfile(f, dtype=np.float64, count=1)
            #####################
            Jacobian[:,j] = np.fromfile(f, dtype=np.float64, count=num_ind_meas)
            # print(Jacobian[:,j])
        
    # # Save the results to a text file 'R2_forward.dat'
    # with open('R2_forward.dat', 'w') as f:
    #     for i in range(num_ind_meas):
    #         f.write(f"{i+1:8d} {' '.join(f'{elec[j,i]:5d}' for j in range(4))} {fcalc[i]:20.8e}\n")
    
    # # Save the Jacobian to a text file 'R2_Jacobian.dat'
    # with open('R2_Jacobian.dat', 'w') as f:
    #     for j in range(num_param):
    #         f.write(f"{j+1}\n")
    #         np.savetxt(f, Jacobian[:,j], fmt="%20.8e")
            
    print("all ok")

    return num_ind_meas, num_param, elec, fcalc, Jacobian