"""
This is the software's entry point.

Usage:
    
    >> python main    
 """

from src.nodes.models import bayes

# setup parameters
DATA_PATH = "data/data01_direction4priors/data/"
SUBJECT = "sub01"
PRIOR_SHAPE = "vonMisesPrior"
PRIOR_MODE = 225
OBJ_FUN = "maxLLH"
READOUT = "map"

if __name__ == "__main__":

    # fit
    output = bayes.fit(
        subject=SUBJECT,
        data_path=DATA_PATH,
        prior_shape=PRIOR_SHAPE,
        prior_mode=PRIOR_MODE,
        readout=READOUT,
        objfun=OBJ_FUN,
    )
    print(output)
