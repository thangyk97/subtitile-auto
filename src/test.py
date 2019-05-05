import librosa
import matplotlib.pyplot as plt

y1, sr1 = librosa.load('../data/orange_uk.wav')
y2, sr2 = librosa.load('../data/orange_ustra.wav')

plt.subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
plt.imshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
plt.imshow(mfcc2)

plt.figure()
from dtw import dtw
from numpy.linalg import norm
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)

plt.imshow(cost.T, origin='lower', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))

print(dist)
print(cost)
print(acc_cost)
print(path)


plt.show()