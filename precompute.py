import numpy as np
import sid
import pickle
import time

def generate_matter_potential_data(flavors, T_domain, poverT_domain):
    data_dict = {}
    
    start = time.time()
    for flavor in flavors:
        print("Computing for flavor:", flavor)
        print("Time elapsed:", time.time() - start)
        flavor_data = np.zeros((len(poverT_domain), len(T_domain)))
        for i, poverT in enumerate(poverT_domain):
            if i % 10 == 0:
                print("Computing for poverT:", poverT)
                print("Time elapsed:", time.time() - start)
            flavor_data[i, :] = sid.matter_potential_integral(poverT * T_domain, T_domain, flavor)
        
        data_dict[flavor] = flavor_data
    
    return data_dict

T_domain = np.logspace(0, 6, 200)
poverT_domain = np.linspace(0.1, 20, 200)
flavors = ['electron', 'muon', 'tau']

data_dict = generate_matter_potential_data(flavors, T_domain, poverT_domain)
data_dict['T_domain'] = T_domain
data_dict['poverT_domain'] = poverT_domain

# Save the data dictionary to a pickle file
pickle_filename = 'data/matter_potential_data.pkl'
with open(pickle_filename, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)

print(f"Data saved to {pickle_filename}")
