#coding=utf-8
import os
import numpy as np

# Merge_dnaid according geneid
def merge_dnaid(L1000_geneid_file_path, L1000_map_file_path, dna_id, x_data):
    dna_id_new = []
    x_data_new = []
    # Load L1000 gene_id
    with open(L1000_geneid_file_path) as geneid_file:
        file_lines = geneid_file.readlines()
    del file_lines[0]
    L1000_geneid = [file_line[: -2] for file_line in file_lines]
    # Load mapping file of L1000 gene_id and dna_id
    with open(L1000_map_file_path) as L1000_map_file:
        file_lines = L1000_map_file.readlines()
    # Remove all useless lines
    for i in range(28):
        del file_lines[0]
    del file_lines[len(file_lines) - 1]
    # Construct dictionary with the key of geneid and the value of dnaid
    dict_L1000_geneid_dnaid = {}
    list_lines = [file_line.split('\t') for file_line in file_lines]
    for list_line in list_lines:
        if list_line[3] in L1000_geneid:
            if not dict_L1000_geneid_dnaid.has_key(list_line[3]):
                dict_L1000_geneid_dnaid[list_line[3]] = [dna_id.index(list_line[0])]
            else:
                dict_L1000_geneid_dnaid[list_line[3]].append(dna_id.index(list_line[0]))
    for each_key in dict_L1000_geneid_dnaid:
        dna_id_new.append(each_key)
    # Merge dnaid according to geneid
    for each_x_data in x_data:
        each_x_data_new = []
        for each_key in dict_L1000_geneid_dnaid:
            list_value = dict_L1000_geneid_dnaid[each_key]
            sum_feature = 0.0
            for each_value in list_value:
                sum_feature += float(each_x_data[each_value])
            new_feature = sum_feature / len(list_value)
            each_x_data_new.append(new_feature)
        x_data_new.append(each_x_data_new)
    return dna_id_new, x_data_new

if __name__ == "__main__":
    # Root path
    dna_id_path = "./dna_id_file.txt"
    L1000_geneid_file_path = "./Gene_ID.txt"
    L1000_map_file_path = "./GPL96.annot"

    # Get dna_id
    with open(dna_id_path) as file_p:
        file_lines = file_p.readlines()
    dna_id = file_lines[0].split('\t')

    '''
    # Merge data from the same drug
    map_file_path = os.path.join(file_root_path, "trt_cp.info")
    durg_sample_id, drug_data = merge_data_from_same_drug(map_file_path, drug_sample_id, drug_data)
    print "Merging data is done..."
    '''
    # xbxs = ['PC3', 'VCAP', 'A375', 'A549', 'HA1E', 'HCC515', 'HEPG2']
    #xbxs = ['PC3', 'VCAP', 'A375', 'A549', 'HA1E', 'HCC515']
    xbxs = ['HEPG2']
    for xbx in xbxs:
        print "welcome to", xbx
        tp_data_file_path = "./real_negative_922/tp_data_"+xbx+".npy"
        tp_data_npy = np.load(tp_data_file_path)
        print "    load npy done, npy.shape", tp_data_npy.shape
        tp_data = []
        for line in tp_data_npy:
            tp_data.append(list(line))
        print "    change to list done, list.shape:", len(tp_data), len(tp_data[0])
        # Merge L1000 dna_id according to gene_id
        dna_id, tp_data = merge_dnaid(L1000_geneid_file_path, L1000_map_file_path, dna_id, tp_data)
        print "    merge done..."
        tp_data = np.array(tp_data)
        np.save("./real_negative_922/tp_data_"+xbx+"_978.npy", tp_data)
        print "    save to npy done!\n"
  
