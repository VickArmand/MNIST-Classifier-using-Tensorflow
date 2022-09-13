import matplotlib.pyplot as plt
import numpy as np
def displaysampleimages(examples,labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        index=np.random.randint(0,examples.shape[0]-1)
        img=examples[index]
        outcome=labels[index]
        plt.subplot(5,5,i+1)
        plt.title(str(outcome))
        # add space between each image and label
        plt.tight_layout()
        # show images in grayscale
        plt.imshow(img,cmap='gray')
    plt.show()