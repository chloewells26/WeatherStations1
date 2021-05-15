import matplotlib.pyplot as plt
import RandomForestsDataCompletion as rfc  # to reference these variables say rfc.variablename

column_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for x in column_numbers:
    actual = rfc.ActualData.iloc[:, x].tail(int(len(rfc.ActualData.iloc[:, x])*(4260/10648)))
    plt.plot(rfc.PredictionData.iloc[:, x], "g-", label='Prediction', alpha=0.5)
    plt.plot(actual.values, "b-", label='Actual', alpha=0.5)
    plt.title('Predictions compared to actual')
    plt.ylabel('units')
    plt.xlabel('Data Points')
    plt.legend()
    plt.show()




