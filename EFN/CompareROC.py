import matplotlib.pyplot as plt
import numpy as np

file_dict = {
  '1Jets_0BTags': 'ROC_curve_1Jet_0Tag.txt',
  '4Jets_0BTags': 'ROC_curve_4Jet_0Tag.txt',
  '4Jets_2BTags': 'ROC_curve_4Jet_2Tag.txt',
  '4Jets_4BTags': 'ROC_curve_4Jet_4Tag.txt'
}

for f in file_dict:
  fpr, tpr = np.loadtxt(file_dict[f])
  plt.plot(fpr, tpr, label=f)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('EFN ROC curve')
plt.legend(loc="best")

plt.savefig("ROC_compare.png")
plt.close()
