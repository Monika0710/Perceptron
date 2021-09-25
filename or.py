from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
from utils.all_utils import save_model,save_plot

def main(data,eta,epoch,model_File,Plot_File):

    
    df = pd.DataFrame(data)

    X,y=prepare_data(df)


    
    model= Perceptron(eta=eta, epoch=epoch)
    model.fit(X, y)

    # _ = model_AND.total_loss()

    save_model(model,filename=model_File)

    save_plot(df,Plot_File,model)


if __name__ =='__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10


    main(OR,ETA,EPOCHS,"or.model","or.png")

