from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
from utils.all_utils import save_model,save_plot


OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

X,y=prepare_data(df)


ETA = 0.3 # 0 OR 1
EPOCHS = 10

model_OR = Perceptron(eta=ETA, epoch=EPOCHS)
model_OR.fit(X, y)

# _ = model_OR.total_loss()

save_model(model_OR,filename="OR.model")

save_plot(df,"OR.png",model_OR)