import pandas as pd
import numpy as np
from ase.units import Hartree, eV, Bohr, Ang, mol, kcal
import seaborn as sns

def get_error_distribution(data, size,  rmsd_type, rmsd_cutoff,cal_type = "MAE", error_type = "abs", target = "target", predict = "predict"):
    '''
    get error distribution for different size of data
    
    '''
    if rmsd_cutoff:
        data_new = data[data[rmsd_type] < rmsd_cutoff]
    else:
        data_new = data.copy()
        rmsd_cutoff = "None"
        rmsd_type = "None"
    data_new[target] = data_new[target]/(kcal/mol)
    
    outfile = open("Platinum_error_distribution_" + str(rmsd_type) + "_"  + str(rmsd_cutoff) + "_" + cal_type + "_" + error_type + "_" + size + ".csv","w")
    outfile.write("molecule_id," + cal_type + "_mean," + cal_type + "_std\n")
    
    if error_type == "rel":
        ### update dateset to remove molecules with only one conf ###
        remove_list = []
        for i in set(data_new["molecule_id"]):
            data_tmp = data_new[data_new["molecule_id"] == i]
            if data_tmp.shape[0] == 1:
                remove_list.append(i)
        print(str(len(remove_list)) + " molecules have been removed since they only have one conformation.")
        data_new = data_new[~data_new["molecule_id"].isin(remove_list)]
    
        ### calculate relative target and prediction ###
        data_min = data_new.groupby("molecule_id").agg({target:np.min,predict + "_1":np.min, predict + "_2":np.min,
                                                       predict + "_3":np.min, predict + "_4":np.min, predict + "_5":np.min})
        data_min.rename(columns = {target:target + "_min", predict + "_1" : predict + "_1_min", predict + "_2" : predict + "_2_min",
                                  predict + "_3" : predict + "_3_min", predict + "_4" : predict + "_4_min", predict + "_5" : predict + "_5_min"}, inplace = True)
        data_min.reset_index(inplace = True)
        data_new = pd.merge(data_new, data_min, on = "molecule_id")
        target_new = "diff_" + target
        predict_new = "diff_" + predict
        data_new[target_new] = data_new[target] - data_new[target + "_min"]
        for i in range(1,6):
            data_new[predict_new + "_" + str(i)] = data_new[predict + "_" + str(i)] - data_new[predict + "_" + str(i) + "_min"]
        target = target_new
        predict = predict_new
        
        
    print(str(data_new.shape[0]) + " conformations have been used to calculate error.")
    for i in set(data_new["molecule_id"]):
        data_tmp = data_new[data_new["molecule_id"] == i]
        yt = data_tmp[target]
        v_list = []
        for num in range(1,6):
            yp = data_tmp[predict + "_" + str(num)]/(kcal/mol)
            v = method(cal_type,yt,yp)
            v_list.append(v)
        v_mean = np.mean(v_list)
        v_std = np.std(v_list)
        outfile.write(str(i) + "," + str(v_mean) + ","+ str(v_std) + "\n")
    outfile.close()
def get_Platinum_size(size1, size2):
    data = pd.merge(df_Platinum_result,df_total,on = df_total.index)
    data = data[(data["molecule_id"] <= size2)&(data["molecule_id"] > size1)]
    print(data.shape)
    data.drop(["key_0"], axis = 1, inplace = True)
    return data

def method(cal_type, target, predict):
    if cal_type == "MAE":
        value = np.abs(target - predict).mean()
    elif cal_type == "RMSE":
        value = np.sqrt(((target-predict)**2).mean())

    return value

if __name__ == "__main__":
    df_Platinum_result = pd.read_csv("../../data/raw/confs_result/test_x1_result_x11_1_values_Platinum.csv")
    df_Platinum_result.rename(columns = {"target":"target_latest","predict":"predict_1"},inplace = True)
    for i in ["2","3","4","5"]:
        df_Platinum_result_new = pd.read_csv("../../data/raw/confs_result/test_x1_result_x11_" + i + "_values_Platinum.csv")
        df_Platinum_result_new.rename(columns = {"predict":"predict_" + i},inplace = True)
        df_Platinum_result = pd.merge(df_Platinum_result,df_Platinum_result_new, on = df_Platinum_result.index)
        df_Platinum_result.drop(["key_0","target"], axis = 1, inplace = True)
    df_Platinum_result.head()
    df_Platinum_result.shape
    df_Platinum_result = df_Platinum_result[0:4076]    
    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
    df_total.rename(columns={"idx":"index"},inplace = True)
    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
    df_total = df_total[0:4076]
    data_10 = get_Platinum_size(0,25)
    get_error_distribution(data_10, "10", None, None, target = "target_latest")
    get_error_distribution(data_10, "10", None, None, target = "target_latest", error_type = "rel")
