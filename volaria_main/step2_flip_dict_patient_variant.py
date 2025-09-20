"""
Load files in the repository (dictionaries) and create a master new dict that's flipped:
    { patient_id : {"11": [variant_id,...], "10": [variant_id,...]} }

Autodetects cohort by sample IDs in the VCF header:
  - GTEx: selects samples containing 'GTEX'
  - CureGN: selects samples containing 'KO'

CLI:
  python step2_flip_dict_patient_variant.py <loc_dicts> <save_file> <original_file>
    <loc_dicts>   directory containing *.pkl shards produced earlier
    <save_file>   output pickle path for flipped dict
    <original_file> a representative VCF to read sample IDs from
    
"""

import numpy as np
import pickle
import sys
import os

loc_dicts = str(sys.argv[1])
print('loc_dicts', loc_dicts)

save_file = str(sys.argv[2])
print('save_file', save_file)

original_file = str(sys.argv[3])
print('original_file', original_file)

if not loc_dicts.endswith('/'):
    loc_dicts = loc_dicts + '/'

# read sample header line (first non-##)
first_line = True
with open(original_file) as f:
    for line in f:
        if line[0:2] == '##':
            continue
        if first_line:
            first_line = False
            patient_order = line.rsplit()
            break

# autodetect cohort by sample IDs in header
tag = None
if any('GTEX' in s for s in patient_order):
    tag = 'GTEX'
elif any('KO' in s for s in patient_order):
    tag = 'KO'
else:
    raise ValueError("Could not detect cohort from header (no 'GTEX' or 'KO' in sample IDs).")

patients_pos = [True if tag in s else False for s in patient_order]
patients_list = np.array(patient_order)[patients_pos]
print('patient_list', patients_list[:3])
print('N patients:', len(patients_list))

patients_list = [str(p) for p in patients_list]

# flip into {patient: {'11': [...], '10': [...]}}
patient_dict = {}
for p in patients_list:
    patient_dict[p] = {'11': [], '10': []}

print('example keys:', list(patient_dict.keys())[:3])

# aggregate shards
for i, file in enumerate(os.listdir(loc_dicts)):
    print(file)
    if '.pkl' not in file:
        print('Skipping file', file)
        continue
    with open(loc_dicts + file, 'rb') as f:
        try:
            var_dict = pickle.load(f)
        except Exception:
            print('****** Error loading the file, skipping ' + file)
            continue

    for k, var in enumerate(list(var_dict.keys())):
        # keys '11', '10', 'dot' are expected in shard dicts
        pats11 = var_dict[var]['11']
        var_str = str(var)
        for p in pats11:
            try:
                patient_dict[p]['11'].append(var_str)
            except Exception:
                patient_dict[p] = {}
                patient_dict[p]['11'] = [var_str]

        pats10 = var_dict[var]['10']
        for p in pats10:
            try:
                patient_dict[p]['10'].append(var_str)
            except Exception:
                patient_dict[p] = {}
                patient_dict[p]['10'] = [var_str]

print('len file', len(patient_dict))
print('Example keys:', list(patient_dict.keys())[:3])

with open(save_file, 'wb') as f:
    pickle.dump(patient_dict, f)
