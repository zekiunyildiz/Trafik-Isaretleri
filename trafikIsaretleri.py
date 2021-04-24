import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

################## Parametreler ##################

path = "myData" #dosyamın adını yazıyorum.
labelFile = 'labels.csv'
batch_size_value = 50
steps_per_epoch_value = 200
epochs_value = 15
imageDimesions = (32,32,3)
testRatio = 0.2 # 1000 resim varsa 200 tanesini seçicez
validationRatio = 0.2  #800 resim vardı 160 resim yapıyor.

################## Parametreler ##################

################## Resimleri yükleyelim ##################

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes...")
for x in range(0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end = " ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

################## Resimleri yükleyelim ##################

################## Split Data ##################

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = validationRatio)

# x_train = Dizideki resimlerin eğitimi
# y_train = Karşılık gelen sınıfın kimliği

################## Split Data ##################

################## Görüntü sayısının her veri kümesi için label sayısıyla eşleşip eşleşmediğini kontrol etmek için ##################

print("Data Shapes")
print("Train", end=" ");print(x_train.shape, y_train.shape)
print("Validation", end=" ");print(x_validation.shape, y_validation.shape)
print("Test", end=" ");print(x_test.shape, y_test.shape)
assert(x_train.shape[0]==y_train.shape[0]), "Eğitim setindeki etiket sayısına eşit olmayan görüntü sayısı"
assert(x_validation.shape[0]==y_validation.shape[0]), "Doğrulama kümesindeki etiketlerin sayısına eşit olmayan görüntü sayısı."
assert(x_test.shape[0]==y_test.shape[0]), "Test setindeki etiket sayısına eşit olmayan görüntü sayısı."
assert(x_train.shape[1:]==(imageDimesions)), "Train görüntülerinin boyutu yanlış"
assert(x_validation.shape[1:]==(imageDimesions)), "Validation görüntülerinin boyutu yanlış"
assert(x_test.shape[1:]==(imageDimesions)), "Test görüntülerinin boyutu yanlış"

################## Görüntü sayısının her veri kümesi için label sayısıyla eşleşip eşleşmediğini kontrol etmek için ##################

################## CSV dosyasını okuyun ##################

data = pd.read_csv(labelFile)
print("Data Shape ",data.shape,type(data))

################## tüm sınıfların bazı örnek resimlerini göster ##################

num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = x_train[y_train==j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1),:,:], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i==2:
            axs[j][i].set_title(str(j)+ "-"+row["Name"])
            num_of_samples.append(len(x_selected))

################## tüm sınıfların bazı örnek resimlerini göster ##################

################## ------- ##################

print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes), num_of_samples)
plt.title("Eğitim veri setinin dağılımı")
plt.xlabel("Class Number")
plt.ylabel("Number of images")
plt.show()

################## görüntüleri ön işleme ##################

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)    #GrayScale çevirme
    img = equalize(img)     #ışığı standardize etmek
    img = img/255       #0 ile 255 yerine 0 ile 1 arasındaki değerleri normalleştirmek için
    return img


x_train = np.array(list(map(preprocessing, x_train))) #tüm görüntüleri tahriş etmek ve önceden işlemek için
x_validation = np.array(list(map(preprocessing,x_validation)))
x_test = np.array(list(map(preprocessing,x_test)))
cv2.imshow("GrayScale Images",x_train[random.randint(0,len(x_train)-1)]) #eğitimin doğru yapılıp yapılmadığını kontrol etmek için

##################### 1 derinlik ekleyelim #####################

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

##################### görüntülerin artırılması: daha genel hale getirmek için #####################

dataGen = ImageDataGenerator(width_shift_range=0.1, # 0.1 = 10% >> eğer 1 den fazl
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(x_train)
batches = dataGen.flow(x_train,y_train,batch_size=20) #veri oluşturucunun görüntü oluşturmasını istemek parti boyutu = her çağrıldığında oluşturulan görüntü sayısı
x_batch, y_batch = next(batches)

# Artırılmış görüntü örneklerini göstermek için

fig,axs = plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test,noOfClasses)

############ Evrişimsel Sinir Ağları (Convolutional neural network) modeli

def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5,5) #bu, özellikleri almak için görüntünün etrafında hareket eden çekirdektir.
                           #bu, 32 32 resim kullanılırken her kenarlıktan 2 pikseli kaldırır
    size_of_Filter2=(3,3)
    size_of_pool = (2,2) #Daha fazla genelleştirmek, aşırı uyumu azaltmak için tüm özellik haritasının ölçeğini küçültün
    no_Of_Nodes = 500

    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # DAHA FAZLA DÖNÜŞÜM KATMANLARI EKLEMEK = DAHA AZ ÖZELLİKLER ANCAK DOĞRULUĞUN ARTMASINA NEDEN OLABİLİR
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # FİLTRE DERİNLİĞİNİ / SAYISINI ETKİLEMEZ

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


############################### TRAIN
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size= batch_size_value),
                              steps_per_epoch=steps_per_epoch_value, epochs=epochs_value,
                              validation_data=(x_validation, y_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# MODELİ BİR PICKLE NESNE OLARAK SAKLAYIN 
pickle_out = open("model_trained.p", "wb")  # wb = WRITE BYTE
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)


