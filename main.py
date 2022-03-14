"""
example:

[fitP,fitPbkp,R2,sdata,fitPt,negLogl,negLoglbkp,Logl_pertrialBestfit,...
      output] = SLfitBayesianModel({'sub02'},...
     [80 40 20 1.74 4.77 10.74 34.25 NaN 0.001 15 NaN],...
     'experiment','vonMisesPrior',...
     'filename','datafit01_vonMisesPrior_BayesWithCard_maxLL_example',...
     'MAPReadout',...
     'MaxLikelihoodFit');
 """


from src.nodes.models import bayes

# setup parameters
DATA_PATH = "data/data01_direction4priors/data/"
SUBJECT = "sub01"
PRIOR_SHAPE = "vonMisesPrior"
OBJ_FUN = "maxLLH"
READOUT = "map"

if __name__ == "__main__":

    # fit
    output = bayes.fit(
        subject=SUBJECT,
        data_path=DATA_PATH,
        prior_shape=PRIOR_SHAPE,
        readout=READOUT,
        objfun=OBJ_FUN,
    )
    print(output)
