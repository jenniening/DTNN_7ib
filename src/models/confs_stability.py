import pandas as pd
import numpy as np
from ase.units import Hartree, eV, Bohr, Ang, mol, kcal
import seaborn as sns

def get_result(test_type, test_infor, test_index_flag = False, test_index = None):
    if test_type:
        test_type = "_" + test_type
    else:
        test_type = ""
        
    i = test_index

    if test_index_flag:
        test_index_1 = pd.read_csv("../../data/raw/confs_result/all_RMSD_test_" + str(i) + ".csv")
        test_index_2 = pd.read_csv("../../data/raw/confs_result/all_RMSD_test_live_" + str(i) + ".csv")
        test_index = pd.concat([test_index_1,test_index_2])
        test_index.reset_index(drop = True, inplace = True)
        num = test_index.shape[0]
        test_infor = pd.merge(test_index[["index","NumRotBonds"]], test_infor, on = ["index"])
    else:
        num = test_infor.shape[0]
            
    data_result = test_infor.copy()
    for result_type in ["x11","MMFF"]:
        data_result_new = pd.read_csv("../../data/raw/confs_result/test_x1_result_" + result_type + "_" + str(i) + "_values" + test_type + ".csv")[0:num]
        data_result_new["target"] = data_result_new["target"]/(kcal/mol)
        data_result_new["predict"] = data_result_new["predict"]/(kcal/mol)
        data_result_new.rename(columns = {"target":"target_latest", "predict":"predict_" + result_type}, inplace = True)
        if "target_latest" not in data_result.columns:
            data_result = pd.merge(data_result_new[["target_latest","predict_" + result_type]], data_result, on = data_result.index)
        else:
            data_result = pd.merge(data_result_new[["predict_" + result_type]], data_result, on = data_result.index)


        data_result.drop("key_0", axis = 1, inplace = True)
            
    return data_result

def method(cal_type, target, predict):
    if cal_type == "MAE":
        value = np.abs(target - predict).mean()
    elif cal_type == "RMSE":
        value = np.sqrt(((target-predict)**2).mean())
    
    return value

def get_error(data, rmsd_cutoff = None, rmsd_type = "rmsd2", data_type = "total", cal_type = "MAE",  error_type = "abs", target = "target", predict = "predict", sr = False):
    '''
    get absolute error
    
    data: input dataset, should includes molecule_id, rmsd1, rmsd2, rmsd3, target_value, predict_value,
    
    rmsd_cutoff: rmsd cutoff, only consider the subset with rmsd less then rmsd_cutoff
    rmsd_type: rmsd values will be used, rmsd1 --> rmsd between MMFF and QM, rmsd2 --> rmsd between MMFF and QM_MMFF, rmsd3 --> rmsd between QM and QM_MMFF
    
    data_type: total --> average error for all conformations, each --> average error for all conformations of each molecule, then average error for all molecule
    cal_type: MAE --> mean absolute error, RMSE --> root mean square error
    error_type: absolute error or relative error
    
    target: target column name
    predict: prediction column name
    
    sr: whether calculate success rate 
    
    '''
    if rmsd_cutoff:
        data_new = data[data[rmsd_type] < rmsd_cutoff]
    else:
        data_new = data.copy()
        
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
        data_min = data_new.groupby("molecule_id").agg({target:np.min,predict:np.min})
        data_min.rename(columns = {target:target + "_min", predict: predict + "_min"}, inplace = True)
        data_min.reset_index(inplace = True)
        data_new = pd.merge(data_new, data_min, on = "molecule_id")
        target_new = "diff_" + target
        predict_new = "diff_" + predict
        data_new[target_new] = data_new[target] - data_new[target + "_min"]
        data_new[predict_new] = data_new[predict] - data_new[predict + "_min"]
        target = target_new
        predict = predict_new
        
        
    print(str(data_new.shape[0]) + " conformations have been used to calculate error.")
    if data_type == "total":
        yt = data_new[target]
        yp = data_new[predict]
        value = method(cal_type, yt,yp) 
    else:
        value_list = []
        for i in set(data_new["molecule_id"]):
            data_tmp = data_new[data_new["molecule_id"] == i]
            yt = data_tmp[target]
            yp = data_tmp[predict]
            v = method(cal_type,yt,yp)
            value_list.append(v)
        value = np.mean(value_list)
        
    if sr:
        lowest_index = data_new[data_new[target] == 0]["index"]
        n_total = len(set(data_new["molecule_id"]))
        data_tmp = data_new.where(data_new[data_new[predict] == 0]["index"].isin(lowest_index))
        data_tmp.dropna(inplace = True)
        n_sr = len(set(data_tmp["molecule_id"]))
        sr = round(float(n_sr)/float(n_total),3)
        
        return sr
    else:
        return round(value,3)

def get_performance(confs_type, error_type, data_type, rmsd_type, cal_type):
    '''
    confs_type: confs or Platinum
    result_type: x11(TL_confs), MMFF(TL_MMFF), energy_abs(MMFF)
    error_type:abs, rel
    data_type: total, each
    rmsd_type: rmsd1, rmsd2, rmsd3
    sr: whether to calculate success rate, only in rel error
    
    '''
    outfile = open("eval_result_" + confs_type + "_" + error_type + "_" + cal_type + "_" + data_type + "_" + rmsd_type + ".csv", "w")
    rmsd_list = [None, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #rmsd_list = [0.6]
    outfile.write("rmsd_cutoff,x11_mean,x11_std,MMFF_mean,MMFF_std,energy_abs_mean,energy_abs_std\n")
    for rmsd_cutoff in rmsd_list:
        print(rmsd_cutoff, rmsd_type)
        total_error = []
        for result_type in ["x11","MMFF","energy_abs"]:
            error_list = []
            for i in range(1,6):
                if confs_type == "confs":
                    df_total = pd.read_csv("../../data/raw/eMol9_RMSD.csv")
                    data_test_confs= get_result(None,df_total,True, i)
                elif confs_type == "Platinum":
                    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
                    df_total.rename(columns={"idx":"index"},inplace = True)
                    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
                    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
                    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
                    df_total = df_total[0:4076]
                    data_test_confs = get_result("Platinum", df_total, False, i)
                elif confs_type == "Platinum_10":
                    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
                    df_total.rename(columns={"idx":"index"},inplace = True)
                    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
                    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
                    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
                    df_total = df_total[0:4076]
                    data_test_confs = get_result("Platinum", df_total, False, i)
                    data_test_confs = data_test_confs[(data_test_confs["molecule_id"] <=25)]
                    data_test_confs.reset_index(drop= True, inplace = True)
                elif confs_type == "Platinum_10_removelarge":
                    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
                    df_total.rename(columns={"idx":"index"},inplace = True)
                    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
                    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
                    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
                    df_total = df_total[0:4076]
                    data_test_confs = get_result("Platinum", df_total, False, i)
                    data_test_confs = data_test_confs[(data_test_confs["molecule_id"] <=25)& (data_test_confs["molecule_id"] != 11) &(data_test_confs["molecule_id"] != 25) ]
                    data_test_confs.reset_index(drop= True, inplace = True)
                elif confs_type == "Platinum_11":
                    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
                    df_total.rename(columns={"idx":"index"},inplace = True)
                    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
                    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
                    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
                    df_total = df_total[0:4076]
                    data_test_confs = get_result("Platinum", df_total, False, i)
                    data_test_confs = data_test_confs[(data_test_confs["molecule_id"] > 25)& (data_test_confs["molecule_id"] <=52)]
                    data_test_confs.reset_index(drop= True, inplace = True)
                elif confs_type == "Platinum_12":
                    df_total = pd.read_csv("../../data/raw/Gaussian_properties_allRMSD.csv")
                    df_total.rename(columns={"idx":"index"},inplace = True)
                    df_total_1 = pd.read_csv("../../data/raw/RMSD.csv")
                    df_total = pd.merge(df_total, df_total_1[["molecule_id","index","rmsd1","rmsd2","rmsd3","mmffenergy"]], on = ["index"])
                    df_total.rename(columns={"mmffenergy":"energy_abs"}, inplace = True)
                    df_total = df_total[0:4076]
                    data_test_confs = get_result("Platinum", df_total, False, i)
                    data_test_confs = data_test_confs[(data_test_confs["molecule_id"] > 53)& (data_test_confs["molecule_id"] <=74)]
                    data_test_confs.reset_index(drop= True, inplace = True)
                    
                if result_type == "energy_abs":
                    if cal_type == "sr":
                        error = get_error(data_test_confs,rmsd_cutoff,rmsd_type,data_type, "MAE", error_type, target = "target_latest", predict = result_type, sr = True)
                    else:
                        
                        error = get_error(data_test_confs,rmsd_cutoff,rmsd_type,data_type, cal_type, error_type, target = "target_latest", predict = result_type, sr = False)
                else:
                    if cal_type == "sr":
                        error = get_error(data_test_confs,rmsd_cutoff,rmsd_type,data_type, "MAE", error_type, target = "target_latest", predict = "predict_" + result_type, sr = True)


                    else:
                        error = get_error(data_test_confs,rmsd_cutoff,rmsd_type,data_type, cal_type, error_type, target = "target_latest", predict = "predict_" + result_type, sr = False)
                error_list.append(error)
            mean_error = np.mean(error_list)
            std_error = np.std(error_list)
            total_error.append(str(mean_error))
            total_error.append(str(std_error))
        print(total_error)
        outfile.write(str(rmsd_cutoff) + "," + ",".join(total_error) + '\n')
    outfile.close()
    
    return None

if __name__ == "__main__":
    ### abs MAE rmsd2
    get_performance("confs","abs","total","rmsd2", "MAE")
