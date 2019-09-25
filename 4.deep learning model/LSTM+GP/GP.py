from GP_crop_v3 import *

predict_year = 2010
path = 'save_2010result_prediction.npz'
RMSE_GP,ME_GP,Average_GP=GaussianProcess(predict_year,path)
print ('RMSE_GP',RMSE_GP)
print ('ME_GP',ME_GP)
print ('Average_GP',Average_GP)