from DatasetManager import StrokeDataset
from Models.tree import DecisionTree, RandomForest
from Models.linear import LogisticRegression, LinearRegression
from Models.support_vector_machine import SupportVectorMachine

RESAMPLES = ['ROSE', 'SMOTE']
SMOKES = ['smoke', 'no smoke', 'fill nan in smokes']

def main():
    RESAMPLE = RESAMPLES[0]
    SMOKE = SMOKES[1]
    dataset = StrokeDataset(resample_method=RESAMPLE, smoke=SMOKE)

    DST = DecisionTree(dataset)
    DST.run()
    DST.plot_results('DST', resample_method=RESAMPLE, smoke=SMOKE)

    RF = RandomForest(dataset)
    RF.run()
    RF.plot_results('RF', resample_method=RESAMPLE, smoke=SMOKE)

    LogR = LogisticRegression(dataset)
    LogR.run()
    LogR.plot_results('LogR', resample_method=RESAMPLE, smoke=SMOKE)

if __name__ == "__main__":
    main()
