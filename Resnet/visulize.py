import pickle
import matplotlib.pyplot as plt

losses = pickle.load(open('losses.p', 'rb'))

epochs = [e[0] for e in losses]
training_loss = [e[1] for e in losses]
val_loss = [e[2] for e in losses]

plt.plot(epochs, training_loss,  label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()
