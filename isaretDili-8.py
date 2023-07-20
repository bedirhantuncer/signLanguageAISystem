import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import math
import os

#etiketleri labels.txt dosyasından alma
labels_path = "isaret_dili_modeli/labels.txt"
labelsfile = open(labels_path, 'r')

# initialize classes and read in lines until there are no more
classes = []
line = labelsfile.readline()
while line:
    # sadece sınıf adını al ve classes listesine ekle
    classes.append(line.split(' ', 1)[1].rstrip())
    line = labelsfile.readline()
# etiket dosyasını kapat
labelsfile.close()
print(classes)

# teachable machine modelini yükle
model_path = "isaret_dili_modeli/keras_model.h5"
model = tf.models.load_model(model_path, compile=False)

cap = cv2.VideoCapture(0)

# webcam video genişliği ve yüksekliği piksel cinsinden -> kendi boyutunuza göre ayarlayın
# eğer yakalama penceresinin yanlarında siyah çubuklar görüyorsanız değerleri ayarlayın
frameWidth = 1280
frameHeight = 720

# genişlik ve yüksekliği piksel cinsinden ayarla
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
# otomatik kazançı etkinleştir
cap.set(cv2.CAP_PROP_GAIN, 0)

while True:

    # daha iyi anlaşılabilmesi için bilimsel gösterimi devre dışı bırak
    np.set_printoptions(suppress=True)

    # Keras modeline beslemek için doğru şekle sahip bir dizi oluştur.
    # 1x 224x224 piksel RGB görüntüsünü giriş olarak kullanıyoruz.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # görüntüyü yakala
    check, frame = cap.read()
    frame = cv2.flip(frame, 1)  ### flip komutu ile sağ ve sol yöndeki tersliği düzelt

    # TM modeliyle kullanmak için kareye kırp
    margin = int(((frameWidth - frameHeight) / 2))
    square_frame = frame[0:frameHeight, margin:margin + frameHeight]
    # TM modeliyle kullanmak için 224x224 boyutuna yeniden boyutlandır
    resized_img = cv2.resize(square_frame, (224, 224))
    # görüntü rengini modele uygun hale getir
    model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # görüntüyü bir numpy dizisine dönüştür
    image_array = np.asarray(model_img)
    # görüntüyü normalize et
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # diziyi yükle
    data[0] = normalized_image_array

    # tahmini çalıştır
    predictions = model.predict(data)

    # güven eşiği %90 olarak ayarlanmıştır.
    conf_threshold = 90
    confidence = []
    conf_label = ""
    threshold_class = ""
    # etiketler için alt kenarda siyah bir çerçeve oluştur
    per_line = 2  # her satırdaki sınıf sayısı
    bordered_frame = cv2.copyMakeBorder(
        square_frame,
        top=0,
        bottom=30 + 15 * math.ceil(len(classes) / per_line),
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    # her bir sınıf için
    for i in range(0, len(classes)):
        # tahmin güvenini yüzdeye dönüştür ve 1B listesine ekle
        confidence.append(int(predictions[0][i] * 100))
        # her satır için metin ekle
        if (i != 0 and not i % per_line):
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(int(0), int(frameHeight + 25 + 15 * math.ceil(i / per_line))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255)
            )
            conf_label = ""
        # sınıfları ve güvenleri etiket metnine ekle
        conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
        # son satırı yazdır
        if (i == (len(classes) - 1)):
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(int(0), int(frameHeight + 25 + 15 * math.ceil((i + 1) / per_line))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255)
            )
            conf_label = ""
        # güven eşiğinin üzerinde ise sıraya gönder
        if confidence[i] > conf_threshold:
            #speakQ.put(classes[i])
            threshold_class = classes[i]
    # güven eşiğinin üzerindeki sınıfı etiket olarak ekle
    cv2.putText(
        img=bordered_frame,
        text=threshold_class,
        org=(int(0), int(frameHeight + 20)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=(255, 255, 255)
    )

    # göster ve 1 ms beklet
    cv2.imshow('webcam goruntusu', bordered_frame)
    cv2.waitKey(1)
