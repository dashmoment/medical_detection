import numpy as np
import pickle
import matplotlib.pyplot as plt

def generate_img(sampleList, sample_length):
    
    img = np.zeros((int(np.sqrt(sample_length)),int(np.sqrt(sample_length)),1))
   
    for _, data in enumerate(sampleList):
        
        xx = int(data//np.sqrt(sample_length))
        yy = int(data%np.sqrt(sample_length))
        img[xx,yy] = 0.5
    img[xx,yy] = 1
    
    return img

def generate_label(sampleList, sample_length):
    
    label_p = np.zeros((int(np.sqrt(sample_length)),int(np.sqrt(sample_length))))
    label_n = np.ones((int(np.sqrt(sample_length)),int(np.sqrt(sample_length))))

    label = np.dstack([label_p, label_n])
    xx = int(sampleList[-1]//np.sqrt(sample_length))
    yy  = int(sampleList[-1]%np.sqrt(sample_length))
    label[xx,yy,:] = [1,0]

    return label


nSmaples = 5000
sample_length = 16384
num_neg = 4
num_pos = 1

samples = []
labels = []

for s in range(nSmaples):
    data_point = []
    for n in range(num_pos + num_neg):
        neg_data = np.random.randint(sample_length)
        data_point.append(neg_data)
        
    img = generate_img(data_point, sample_length)
    label = generate_label(data_point, sample_length)
    samples.append(img)
    labels.append(label)

samples = np.array(samples)
labels = np.array(labels)

# plt.imshow(samples[10,:,:,0])
# plt.show()
# plt.imshow(labels[10,:,:,0])
# plt.show()

with open('samples/16384_s5000/data.pkl', 'wb') as outp:
    pickle.dump(samples, outp)
outp.close()

with open('samples/16384_s5000/label.pkl', 'wb') as outp:
    pickle.dump(labels, outp)
outp.close()


