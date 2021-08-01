# Tensorflow ile Transfer Learning - Ölçeklendirme (Scaling)

Önceki iki not defterinde (özellik çıkarma ve ince ayar) transfer learningin gücünü gördük.

Artık daha küçük modelleme deneylerimizin işe yaradığını biliyoruz, daha fazla veriyle işleri bir adım öteye taşımanın zamanı geldi.

Bu, makine öğrenimi ve derin öğrenmede yaygın bir uygulamadır: Daha büyük miktarda veriye ölçeklendirmeden önce az miktarda veri üzerinde çalışan bir model edinin.

> 🔑 Not: Makine öğrenimi uygulayıcılarının sloganını unutmadınız mı? "Deney, deney, deney."

Food Vision projemizin hayata geçmesine biraz daha yaklaşmanın zamanı geldi. Bu not defterinde, Food101 verilerinin 10 sınıfını kullanmaktan, Food101 veri kümesindeki tüm sınıfları kullanmaya doğru ölçeklendireceğiz.

Hedefimiz, orijinal [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf) belgesinin sonuçlarını %10 veri ile geçmektir.

- Food101 verilerinin %10'unun indirilmesi ve hazırlanması (eğitim verilerinin %10'u)
- Food101 eğitim verilerinin %10'unda bir özellik çıkarma aktarımı öğrenme modeli eğitimi
- Özellik çıkarma modelimizin ince ayarını yapma
- Eğitilmiş modelimizi kaydetme ve yükleme
- Eğitim verilerinin %10'u üzerinden eğitilen Food Vision modelimizin performansının değerlendirilmesi
- Modelimizin en yanlış tahminlerini bulma
- Food Vision modelimiz ile özel gıda görselleri üzerinde tahminlerde bulunmak

## Yardımcı Fonksiyonları Oluşturalım


```python
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"{log_dir}")
  return tensorboard_callback
```


```python
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
```


```python
import zipfile

def unzip_data(filename):
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
```


```python
def compare_historys(original_history, new_history, initial_epochs=5):
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') 
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
```


```python
import os

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"'{dirpath}' klasöründe {len(filenames)} veri var.")
```


```python
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), 
         yticks=np.arange(n_classes), 
         xticklabels=labels,
         yticklabels=labels)
  
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  if savefig:
    fig.savefig("confusion_matrix.png")
```

## 101 Yemek Sınıfı: Daha Az Veriyle Çalışmak

Şimdiye kadar kullandığımız transfer öğrenme modelinin 10 Yemek Sınıfı veri seti ile oldukça iyi çalıştığını onayladık. Şimdi, tam 101 Yemek Sınıfı ile nasıl gittiklerini görme zamanı.

Orijinal Food101 veri setinde sınıf başına 1000 görüntü (eğitim setindeki her sınıftan 750 ve test setindeki her sınıftan 250), toplam 101.000 görüntü vardır.



```python
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip 

unzip_data("101_food_classes_10_percent.zip")

train_dir = "101_food_classes_10_percent/train/"
test_dir = "101_food_classes_10_percent/test/"
```

    --2021-07-20 10:29:35--  https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.199.128, 108.177.98.128, 74.125.197.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.199.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1625420029 (1.5G) [application/zip]
    Saving to: ‘101_food_classes_10_percent.zip’
    
    101_food_classes_10 100%[===================>]   1.51G   147MB/s    in 7.6s    
    
    2021-07-20 10:29:42 (204 MB/s) - ‘101_food_classes_10_percent.zip’ saved [1625420029/1625420029]
    



```python
walk_through_dir("101_food_classes_10_percent")
```

    '101_food_classes_10_percent' klasöründe 0 veri var.
    '101_food_classes_10_percent/train' klasöründe 0 veri var.
    '101_food_classes_10_percent/train/croque_madame' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/frozen_yogurt' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/bruschetta' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/paella' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/dumplings' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/chicken_wings' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/french_onion_soup' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/baklava' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/risotto' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/gyoza' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/tiramisu' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/eggs_benedict' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/churros' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/baby_back_ribs' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/hot_dog' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/sushi' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/macaroni_and_cheese' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/chocolate_mousse' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/steak' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/edamame' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/falafel' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/huevos_rancheros' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/prime_rib' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/clam_chowder' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/lobster_bisque' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/crab_cakes' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/omelette' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/hummus' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pork_chop' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pizza' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/ravioli' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/panna_cotta' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/grilled_cheese_sandwich' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/ice_cream' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/tacos' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/ramen' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/cup_cakes' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/escargots' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/filet_mignon' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/chocolate_cake' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/red_velvet_cake' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/macarons' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/bibimbap' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/poutine' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/greek_salad' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/fried_rice' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/beef_carpaccio' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/bread_pudding' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/tuna_tartare' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/seaweed_salad' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pho' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/foie_gras' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/french_toast' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/fish_and_chips' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/club_sandwich' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/cheese_plate' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/chicken_quesadilla' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/takoyaki' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/mussels' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/beef_tartare' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/lobster_roll_sandwich' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/sashimi' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/strawberry_shortcake' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/chicken_curry' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/nachos' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/onion_rings' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/cheesecake' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/caesar_salad' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/beignets' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/french_fries' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/carrot_cake' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/fried_calamari' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/hot_and_sour_soup' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/garlic_bread' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/shrimp_and_grits' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/caprese_salad' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/donuts' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/spaghetti_carbonara' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pad_thai' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/samosa' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/gnocchi' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/deviled_eggs' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/cannoli' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/waffles' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/miso_soup' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/lasagna' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/apple_pie' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pancakes' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/peking_duck' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/spring_rolls' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/scallops' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/ceviche' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/beet_salad' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/creme_brulee' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/grilled_salmon' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/oysters' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/pulled_pork_sandwich' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/spaghetti_bolognese' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/breakfast_burrito' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/hamburger' klasöründe 75 veri var.
    '101_food_classes_10_percent/train/guacamole' klasöründe 75 veri var.
    '101_food_classes_10_percent/test' klasöründe 0 veri var.
    '101_food_classes_10_percent/test/croque_madame' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/frozen_yogurt' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/bruschetta' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/paella' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/dumplings' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/chicken_wings' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/french_onion_soup' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/baklava' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/risotto' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/gyoza' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/tiramisu' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/eggs_benedict' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/churros' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/baby_back_ribs' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/hot_dog' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/sushi' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/macaroni_and_cheese' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/chocolate_mousse' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/steak' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/edamame' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/falafel' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/huevos_rancheros' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/prime_rib' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/clam_chowder' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/lobster_bisque' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/crab_cakes' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/omelette' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/hummus' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pork_chop' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pizza' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/ravioli' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/panna_cotta' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/grilled_cheese_sandwich' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/ice_cream' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/tacos' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/ramen' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/cup_cakes' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/escargots' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/filet_mignon' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/chocolate_cake' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/red_velvet_cake' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/macarons' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/bibimbap' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/poutine' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/greek_salad' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/fried_rice' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/beef_carpaccio' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/bread_pudding' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/tuna_tartare' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/seaweed_salad' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pho' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/foie_gras' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/french_toast' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/fish_and_chips' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/club_sandwich' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/cheese_plate' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/chicken_quesadilla' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/takoyaki' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/mussels' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/beef_tartare' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/lobster_roll_sandwich' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/sashimi' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/strawberry_shortcake' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/chicken_curry' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/nachos' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/onion_rings' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/cheesecake' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/caesar_salad' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/beignets' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/french_fries' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/carrot_cake' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/fried_calamari' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/hot_and_sour_soup' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/garlic_bread' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/shrimp_and_grits' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/caprese_salad' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/donuts' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/spaghetti_carbonara' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pad_thai' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/samosa' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/gnocchi' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/deviled_eggs' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/cannoli' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/waffles' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/miso_soup' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/lasagna' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/apple_pie' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pancakes' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/peking_duck' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/spring_rolls' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/scallops' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/ceviche' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/beet_salad' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/creme_brulee' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/grilled_salmon' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/oysters' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/pulled_pork_sandwich' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/spaghetti_bolognese' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/breakfast_burrito' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/hamburger' klasöründe 250 veri var.
    '101_food_classes_10_percent/test/guacamole' klasöründe 250 veri var.


Görsellerimizi ve etiketlerimizi, dizini modelimize geçirmemizi sağlayan bir TensorFlow veri türü olan tf.data.Dataset'e dönüştürmek için `image_dataset_from_directory()` işlevini kullanalım.

Test veri kümesi için, daha sonra üzerinde tekrarlanabilir değerlendirme ve görselleştirme yapabilmek için `shuffle=False` ayarını yapacağız.



```python
import tensorflow as tf

IMG_SIZE = (224, 224)

train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SIZE)
                                                                                
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False)
```

    Found 7575 files belonging to 101 classes.
    Found 25250 files belonging to 101 classes.


Harika! Eğitim setinde sınıf başına 75 görüntü (75 görüntü * 101 sınıf = 7575 görüntü) ve test setinde 25250 görüntü (250 görüntü * 101 sınıf = 25250 görüntü) ile verilerimiz beklendiği gibi içe aktarılmış gibi görünüyor.

## 101 Yemek Sınıfının %10'unda Transfer Öğrenimi ile Büyük Bir Modeli Eğitin

Gıda görüntü verilerimiz, modelleme zamanı olarak TensorFlow'a aktarıldı.

Deneylerimizi hızlı tutmak için, birkaç dönem için önceden eğitilmiş bir modelle özellik çıkarma transferi öğrenimini kullanarak başlayacağız ve ardından birkaç dönem için daha ince ayar yapacağız.

Daha spesifik olarak, hedefimiz, eğitim verilerinin %10'u ve aşağıdaki modelleme kurulumu ile orijinal Food101 belgesindeki (101 sınıfta %50,76 doğruluk) taban çizgisini geçip geçemeyeceğimizi görmek olacaktır:

- Eğitim sırasında ilerlememizi kaydetmek için bir ModelCheckpoint geri çağrısı, bu, her seferinde sıfırdan eğitim almak zorunda kalmadan daha sonra daha fazla eğitim deneyebileceğimiz anlamına gelir.
- Doğrudan modele entegre edilmiş veri büyütme
Temel modelimiz olarak `tf.keras.applications`'dan üst katman olmayan EfficientNetB0 mimarisi
- 101 gizli nöronlu (gıda sınıfı sayısıyla aynı) ve çıktı katmanı olarak softmax aktivasyonlu bir Yoğun katman
İkiden fazla sınıfla uğraştığımız için kayıp fonksiyonu olarak kategorik çapraz entropi
- Varsayılan ayarlarla Adam optimize edici
- Test verilerinin %15'ini değerlendirirken eğitim verilerine 5 tam geçiş fit etmek

ModelCheckpoint callback'i oluşturarak başlayalım.

Modelimizin görünmeyen veriler üzerinde iyi performans göstermesini istediğimizden, onu doğrulama doğruluğu metriğini izleyecek ve bu konuda en iyi puanı alan model ağırlıklarını kaydedecek şekilde ayarlayacağız.


```python
checkpoint_path = "101_classes_10_percent_data_model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         monitor="val_accuracy",
                                                         save_best_only=True) 
```

Kontrol noktası hazır. Şimdi Sequential API ile küçük bir veri büyütme modeli oluşturalım. Küçük boyutlu bir eğitim seti ile çalıştığımız için bu, modelimizin eğitim verilerine gereğinden fazla uymasını önlemeye yardımcı olacaktır.


```python
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

data_augmentation = Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  preprocessing.RandomZoom(0.2),
], name="data_augmentation")
```

Güzel! Data_augmentation Sequential modelini Functional API modelimize bir katman olarak ekleyebileceğiz. Bu şekilde, modelimizi daha sonra eğitmeye devam etmek istersek, veri büyütme zaten yerleşiktir.

İşlevsel API modellerinden bahsetmişken, temel modelimiz olarak `tf.keras.applications.EfficientNetB0` kullanarak bir özellik çıkarma aktarımı öğrenme modeli oluşturmanın zamanı geldi.

Temel modeli `include_top=False` parametresini kullanarak içe aktaracağız, böylece kendi çıktı katmanlarımızı, özellikle `GlobalAveragePooling2D()` (temel modelin çıktılarını çıktı katmanı tarafından kullanılabilir bir şekle yoğunlaştırır), ardından bir Yoğun katman ekleyebiliriz.


```python
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3), name="input_layer") 
x = data_augmentation(inputs) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x) 
outputs = layers.Dense(len(train_data_all_10_percent.class_names), activation="softmax", name="output_layer")(x) 
model = tf.keras.Model(inputs, outputs)
```

    Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    16711680/16705208 [==============================] - 0s 0us/step



```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_layer (InputLayer)     [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    data_augmentation (Sequentia (None, None, None, 3)     0         
    _________________________________________________________________
    efficientnetb0 (Functional)  (None, None, None, 1280)  4049571   
    _________________________________________________________________
    global_average_pooling (Glob (None, 1280)              0         
    _________________________________________________________________
    output_layer (Dense)         (None, 101)               129381    
    =================================================================
    Total params: 4,178,952
    Trainable params: 129,381
    Non-trainable params: 4,049,571
    _________________________________________________________________


İyi görünüyor! İşlevsel modelimizin 5 katmanı vardır, ancak bu katmanların her birinin içinde değişen miktarlarda katmanlar vardır.

Trainable ve non-trainable parametrelerin sayısına dikkat edin. Görünen o ki, eğitilebilir tek parametre output_layer içinde, ki bu özellik çıkarmanın bu ilk çalıştırmasında tam olarak peşinde olduğumuz şey; modelin çıktılarını özel verilerimize göre ayarlamasına izin verirken temel modeldeki (EfficientNetb0) tüm öğrenilen kalıpları dondurma.


```python
# modeli derleme
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# modeli fit etme
history_all_classes_10_percent = model.fit(train_data_all_10_percent,
                                           epochs=5, 
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), 
                                           callbacks=[checkpoint_callback]) 
```

    Epoch 1/5
    237/237 [==============================] - 124s 376ms/step - loss: 3.4567 - accuracy: 0.2446 - val_loss: 2.5810 - val_accuracy: 0.4404
    Epoch 2/5
    237/237 [==============================] - 71s 300ms/step - loss: 2.3444 - accuracy: 0.4585 - val_loss: 2.1951 - val_accuracy: 0.4703
    Epoch 3/5
    237/237 [==============================] - 65s 271ms/step - loss: 1.9753 - accuracy: 0.5337 - val_loss: 2.0319 - val_accuracy: 0.4934
    Epoch 4/5
    237/237 [==============================] - 61s 257ms/step - loss: 1.7592 - accuracy: 0.5782 - val_loss: 1.9716 - val_accuracy: 0.4942
    Epoch 5/5
    237/237 [==============================] - 59s 247ms/step - loss: 1.6027 - accuracy: 0.6046 - val_loss: 1.8762 - val_accuracy: 0.5199


Görünüşe göre verinin %10'u ile taban çizgimizi (orijinal Food101 makalesinden elde edilen sonuçlar) geçtik! 5 dakikadan kısa bir sürede... bu derin öğrenmenin gücüdür ve daha doğrusu, transfer öğreniminin.

Kayıp eğrileri nasıl görünüyor?


```python
plot_loss_curves(history_all_classes_10_percent)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_24_0.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_24_1.png)
    


🤔 Soru: Bu eğriler ne öneriyor? İpucu: ideal olarak, iki eğri birbirine çok benzer olmalıdır.

## İnce Ayar (Fine Tuning)

Özellik çıkarma transferi öğrenme modelimiz iyi performans gösteriyor. Neden temel modelde birkaç katmanda ince ayar yapmaya çalışmıyoruz ve herhangi bir iyileştirme elde edip edemeyeceğimizi görmüyoruz?

İyi haber şu ki, `ModelCheckpoint` geri araması sayesinde zaten iyi performans gösteren modelimizin kaydedilmiş ağırlıklarına sahibiz, böylece ince ayar herhangi bir fayda sağlamıyorsa geri dönebiliriz.

Temel modelde ince ayar yapmak için önce `trainable` özniteliğini `True` olarak ayarlayacağız, tüm donmuşları çözeceğiz.

Daha sonra, nispeten küçük bir eğitim veri setimiz olduğu için, son 5 hariç her katmanı yeniden donduracağız ve onları eğitilebilir hale getireceğiz.


```python
# Temel modeldeki tüm katmanları çöz
base_model.trainable = True

# Son 5 hariç her katmanı yeniden dondur
for layer in base_model.layers[:-5]:
  layer.trainable = False
```

Modelimizdeki katmanlarda yeni bir değişiklik yaptık ve modelimizde her değişiklik yaptığımızda ne yapmamız gerekiyor? (Yeniden derlemek :) )

İnce ayar yaptığımız için, önceki eğitilmiş ağırlıklardaki güncellemelerin çok büyük olmamasını sağlamak için 10 kat daha düşük bir öğrenme oranı kullanacağız.


```python
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4), # Varsayılandan 10 kat daha düşük öğrenme oranı
              metrics=['accuracy'])
```

Model yeniden derlendi, istediğimiz katmanların eğitilebilir olduğundan emin olmaya ne dersiniz?


```python
for layer in model.layers:
  print(layer.name, layer.trainable)
```

    input_layer True
    data_augmentation True
    efficientnetb0 True
    global_average_pooling True
    output_layer True



```python
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name, layer.trainable)
```

    0 input_1 False
    1 rescaling False
    2 normalization False
    3 stem_conv_pad False
    4 stem_conv False
    5 stem_bn False
    6 stem_activation False
    7 block1a_dwconv False
    8 block1a_bn False
    9 block1a_activation False
    10 block1a_se_squeeze False
    11 block1a_se_reshape False
    12 block1a_se_reduce False
    13 block1a_se_expand False
    14 block1a_se_excite False
    15 block1a_project_conv False
    16 block1a_project_bn False
    17 block2a_expand_conv False
    18 block2a_expand_bn False
    19 block2a_expand_activation False
    20 block2a_dwconv_pad False
    21 block2a_dwconv False
    22 block2a_bn False
    23 block2a_activation False
    24 block2a_se_squeeze False
    25 block2a_se_reshape False
    26 block2a_se_reduce False
    27 block2a_se_expand False
    28 block2a_se_excite False
    29 block2a_project_conv False
    30 block2a_project_bn False
    31 block2b_expand_conv False
    32 block2b_expand_bn False
    33 block2b_expand_activation False
    34 block2b_dwconv False
    35 block2b_bn False
    36 block2b_activation False
    37 block2b_se_squeeze False
    38 block2b_se_reshape False
    39 block2b_se_reduce False
    40 block2b_se_expand False
    41 block2b_se_excite False
    42 block2b_project_conv False
    43 block2b_project_bn False
    44 block2b_drop False
    45 block2b_add False
    46 block3a_expand_conv False
    47 block3a_expand_bn False
    48 block3a_expand_activation False
    49 block3a_dwconv_pad False
    50 block3a_dwconv False
    51 block3a_bn False
    52 block3a_activation False
    53 block3a_se_squeeze False
    54 block3a_se_reshape False
    55 block3a_se_reduce False
    56 block3a_se_expand False
    57 block3a_se_excite False
    58 block3a_project_conv False
    59 block3a_project_bn False
    60 block3b_expand_conv False
    61 block3b_expand_bn False
    62 block3b_expand_activation False
    63 block3b_dwconv False
    64 block3b_bn False
    65 block3b_activation False
    66 block3b_se_squeeze False
    67 block3b_se_reshape False
    68 block3b_se_reduce False
    69 block3b_se_expand False
    70 block3b_se_excite False
    71 block3b_project_conv False
    72 block3b_project_bn False
    73 block3b_drop False
    74 block3b_add False
    75 block4a_expand_conv False
    76 block4a_expand_bn False
    77 block4a_expand_activation False
    78 block4a_dwconv_pad False
    79 block4a_dwconv False
    80 block4a_bn False
    81 block4a_activation False
    82 block4a_se_squeeze False
    83 block4a_se_reshape False
    84 block4a_se_reduce False
    85 block4a_se_expand False
    86 block4a_se_excite False
    87 block4a_project_conv False
    88 block4a_project_bn False
    89 block4b_expand_conv False
    90 block4b_expand_bn False
    91 block4b_expand_activation False
    92 block4b_dwconv False
    93 block4b_bn False
    94 block4b_activation False
    95 block4b_se_squeeze False
    96 block4b_se_reshape False
    97 block4b_se_reduce False
    98 block4b_se_expand False
    99 block4b_se_excite False
    100 block4b_project_conv False
    101 block4b_project_bn False
    102 block4b_drop False
    103 block4b_add False
    104 block4c_expand_conv False
    105 block4c_expand_bn False
    106 block4c_expand_activation False
    107 block4c_dwconv False
    108 block4c_bn False
    109 block4c_activation False
    110 block4c_se_squeeze False
    111 block4c_se_reshape False
    112 block4c_se_reduce False
    113 block4c_se_expand False
    114 block4c_se_excite False
    115 block4c_project_conv False
    116 block4c_project_bn False
    117 block4c_drop False
    118 block4c_add False
    119 block5a_expand_conv False
    120 block5a_expand_bn False
    121 block5a_expand_activation False
    122 block5a_dwconv False
    123 block5a_bn False
    124 block5a_activation False
    125 block5a_se_squeeze False
    126 block5a_se_reshape False
    127 block5a_se_reduce False
    128 block5a_se_expand False
    129 block5a_se_excite False
    130 block5a_project_conv False
    131 block5a_project_bn False
    132 block5b_expand_conv False
    133 block5b_expand_bn False
    134 block5b_expand_activation False
    135 block5b_dwconv False
    136 block5b_bn False
    137 block5b_activation False
    138 block5b_se_squeeze False
    139 block5b_se_reshape False
    140 block5b_se_reduce False
    141 block5b_se_expand False
    142 block5b_se_excite False
    143 block5b_project_conv False
    144 block5b_project_bn False
    145 block5b_drop False
    146 block5b_add False
    147 block5c_expand_conv False
    148 block5c_expand_bn False
    149 block5c_expand_activation False
    150 block5c_dwconv False
    151 block5c_bn False
    152 block5c_activation False
    153 block5c_se_squeeze False
    154 block5c_se_reshape False
    155 block5c_se_reduce False
    156 block5c_se_expand False
    157 block5c_se_excite False
    158 block5c_project_conv False
    159 block5c_project_bn False
    160 block5c_drop False
    161 block5c_add False
    162 block6a_expand_conv False
    163 block6a_expand_bn False
    164 block6a_expand_activation False
    165 block6a_dwconv_pad False
    166 block6a_dwconv False
    167 block6a_bn False
    168 block6a_activation False
    169 block6a_se_squeeze False
    170 block6a_se_reshape False
    171 block6a_se_reduce False
    172 block6a_se_expand False
    173 block6a_se_excite False
    174 block6a_project_conv False
    175 block6a_project_bn False
    176 block6b_expand_conv False
    177 block6b_expand_bn False
    178 block6b_expand_activation False
    179 block6b_dwconv False
    180 block6b_bn False
    181 block6b_activation False
    182 block6b_se_squeeze False
    183 block6b_se_reshape False
    184 block6b_se_reduce False
    185 block6b_se_expand False
    186 block6b_se_excite False
    187 block6b_project_conv False
    188 block6b_project_bn False
    189 block6b_drop False
    190 block6b_add False
    191 block6c_expand_conv False
    192 block6c_expand_bn False
    193 block6c_expand_activation False
    194 block6c_dwconv False
    195 block6c_bn False
    196 block6c_activation False
    197 block6c_se_squeeze False
    198 block6c_se_reshape False
    199 block6c_se_reduce False
    200 block6c_se_expand False
    201 block6c_se_excite False
    202 block6c_project_conv False
    203 block6c_project_bn False
    204 block6c_drop False
    205 block6c_add False
    206 block6d_expand_conv False
    207 block6d_expand_bn False
    208 block6d_expand_activation False
    209 block6d_dwconv False
    210 block6d_bn False
    211 block6d_activation False
    212 block6d_se_squeeze False
    213 block6d_se_reshape False
    214 block6d_se_reduce False
    215 block6d_se_expand False
    216 block6d_se_excite False
    217 block6d_project_conv False
    218 block6d_project_bn False
    219 block6d_drop False
    220 block6d_add False
    221 block7a_expand_conv False
    222 block7a_expand_bn False
    223 block7a_expand_activation False
    224 block7a_dwconv False
    225 block7a_bn False
    226 block7a_activation False
    227 block7a_se_squeeze False
    228 block7a_se_reshape False
    229 block7a_se_reduce False
    230 block7a_se_expand False
    231 block7a_se_excite False
    232 block7a_project_conv True
    233 block7a_project_bn True
    234 top_conv True
    235 top_bn True
    236 top_activation True


Mükemmel! Modelimize ince ayar yapma zamanı.

Herhangi bir yararın olup olmadığını görmek için 5 epoch daha yeterli olmalıdır (ancak her zaman daha fazlasını deneyebiliriz).

`Fit()` işlevindeki `initial_epoch` parametresini kullanarak özellik çıkarma modelinin kaldığı yerden eğitime başlayacağız.


```python
fine_tune_epochs = 10

history_all_classes_10_percent_fine_tune = model.fit(train_data_all_10_percent,
                                                     epochs=fine_tune_epochs,
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)), 
                                                     initial_epoch=history_all_classes_10_percent.epoch[-1]) 
```

    Epoch 5/10
    237/237 [==============================] - 65s 246ms/step - loss: 1.3694 - accuracy: 0.6510 - val_loss: 1.9126 - val_accuracy: 0.5024
    Epoch 6/10
    237/237 [==============================] - 56s 233ms/step - loss: 1.2323 - accuracy: 0.6706 - val_loss: 1.9015 - val_accuracy: 0.5106
    Epoch 7/10
    237/237 [==============================] - 54s 228ms/step - loss: 1.1418 - accuracy: 0.6969 - val_loss: 1.9327 - val_accuracy: 0.5021
    Epoch 8/10
    237/237 [==============================] - 54s 226ms/step - loss: 1.0874 - accuracy: 0.7119 - val_loss: 1.9087 - val_accuracy: 0.5026
    Epoch 9/10
    237/237 [==============================] - 51s 214ms/step - loss: 1.0345 - accuracy: 0.7236 - val_loss: 1.8899 - val_accuracy: 0.5037
    Epoch 10/10
    237/237 [==============================] - 51s 215ms/step - loss: 0.9691 - accuracy: 0.7391 - val_loss: 1.8552 - val_accuracy: 0.5154


Bir kez daha, eğitim sırasında test verilerinin sadece küçük bir kısmını değerlendiriyorduk, modelimizin tüm test verileri üzerinde nasıl gittiğini öğrenelim.


```python
results_all_classes_10_percent_fine_tune = model.evaluate(test_data)
results_all_classes_10_percent_fine_tune
```

    790/790 [==============================] - 90s 113ms/step - loss: 1.6013 - accuracy: 0.5789





    [1.6013119220733643, 0.578930675983429]



Hmm... Görünüşe göre modelimiz ince ayardan biraz güç almış.

`Compare_historys()` fonksiyonumuzu kullanarak ve eğitim eğrilerinin ne söylediğini görerek daha iyi bir resim elde edebiliriz.


```python
compare_historys(original_history=history_all_classes_10_percent,
                 new_history=history_all_classes_10_percent_fine_tune,
                 initial_epochs=5)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_38_0.png)
    


İnce ayardan sonra modelimizin eğitim metrikleri önemli ölçüde iyileşti, ancak doğrulama çok fazla değil. Görünüşe göre modelimiz overfitting oldu.

Yine de sorun değil, önceden eğitilmiş bir modelin eğitildiği veriler özel verilerinize benzer olduğunda, ince ayarın fazla uydurmaya yol açması çok sık görülür.

Bizim durumumuzda, önceden eğitilmiş modelimiz EfficientNetB0, tıpkı gıda veri setimiz gibi birçok gerçek gıda resmini içeren ImageNet üzerinde eğitilmiştir.

Özellik çıkarma zaten iyi çalışıyorsa, ince ayardan gördüğünüz iyileştirmeler, veri kümeniz, temel modelinizin önceden eğitilmiş olduğu verilerden önemli ölçüde farklıymış gibi büyük olmayabilir.

## Eğitilmiş Modelimizi Kaydetme

Modelimizi sıfırdan yeniden eğitmek zorunda kalmamak için `save()` yöntemini kullanarak dosyaya kaydedelim.


```python
# model.save("drive/My Drive/")
```

## Tüm Farklı Sınıflarda Büyük Modelin Performansının Değerlendirilmesi

Kullandığımız değerlendirme ölçütlerine göre oldukça iyi performans gösteren, eğitilmiş ve kaydedilmiş bir modelimiz var.

Ama metrik şemaları, hadi modelimizin performansını biraz daha derinlemesine inceleyelim ve bazı görselleştirmeler yapalım.

Bunu yapmak için, kaydedilen modeli yükleyeceğiz ve bunu test veri setinde bazı tahminler yapmak için kullanacağız.

> 🔑 Not: Bir makine öğrenimi modelini değerlendirmek, eğitmek kadar önemlidir. Metrikler aldatıcı olabilir. İyi görünen eğitim numaralarına aldanmadığınızdan emin olmak için modelinizin performansını her zaman görünmeyen veriler üzerinde görselleştirmelisiniz.


```python
import tensorflow as tf

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/06_101_food_class_10_percent_saved_big_dog_model.zip
saved_model_path = "06_101_food_class_10_percent_saved_big_dog_model.zip"
unzip_data(saved_model_path)

model = tf.keras.models.load_model(saved_model_path.split(".")[0]) 
```

Yüklenen modelimizin gerçekten eğitilmiş bir model olduğundan emin olmak için, performansını test veri setinde değerlendirelim.


```python
loaded_loss, loaded_accuracy = model.evaluate(test_data)
loaded_loss, loaded_accuracy
```

    790/790 [==============================] - 92s 115ms/step - loss: 1.8027 - accuracy: 0.6078





    (1.8027206659317017, 0.6077623963356018)



Olağanüstü! Yüklenen modelimiz, kaydetmeden önceki kadar iyi performans gösteriyor gibi görünüyor. Bazı tahminlerde bulunalım.

## Eğitimli Modelimiz ile Tahminler Yapmak

Eğitilmiş modelimizi değerlendirmek için, onunla bazı tahminler yapmamız ve ardından bu tahminleri test veri seti ile karşılaştırmamız gerekiyor.

Model, test veri setini hiç görmediği için, bu bize modelin gerçek dünyada eğitildiği şeye benzer veriler üzerinde nasıl performans göstereceğine dair bir gösterge vermelidir.

Eğitilmiş modelimiz ile tahminler yapmak için, test verilerini geçerek `predict()` yöntemini kullanabiliriz.

Verilerimiz çok sınıflı olduğundan, bunu yapmak muhtemelen her örnek için bir tensör tahmini döndürür.

Başka bir deyişle, eğitilen model bir görüntüyü her gördüğünde, onu eğitim sırasında öğrendiği tüm kalıplarla karşılaştıracak ve görüntünün o sınıf olma olasılığının her sınıf (101'inin tümü) için bir çıktı döndürecektir.


```python
pred_probs = model.predict(test_data, verbose=1)
```

    790/790 [==============================] - 64s 79ms/step


Tüm test görüntülerini modelimize ilettik ve her birinde hangi gıda olduğunu düşündüğü hakkında bir tahminde bulunmasını istedik.

Yani test veri setinde 25250 görselimiz olsaydı, sizce kaç tahminimiz olmalı?



```python
len(pred_probs)
```




    25250



Ve her görüntü 101 sınıftan biri olabilseydi, her görüntü için kaç tahminimiz olacağını düşünüyorsunuz?


```python
pred_probs.shape
```




    (25250, 101)



Sahip olduğumuz şeye genellikle bir tahmin olasılık tensörü (veya dizi) denir.

Bakalım ilk 10 nasıl görünüyor.


```python
pred_probs[:10]
```




    array([[5.95420077e-02, 3.57419503e-06, 4.13768589e-02, ...,
            1.41386813e-09, 8.35307583e-05, 3.08974274e-03],
           [9.64016676e-01, 1.37532707e-09, 8.47805641e-04, ...,
            5.42872003e-05, 7.83623513e-12, 9.84663906e-10],
           [9.59258676e-01, 3.25335823e-05, 1.48669467e-03, ...,
            7.18913384e-07, 5.43973158e-07, 4.02759651e-05],
           ...,
           [4.73132670e-01, 1.29312355e-07, 1.48055656e-03, ...,
            5.97501639e-04, 6.69690999e-05, 2.34693434e-05],
           [4.45719399e-02, 4.72655188e-07, 1.22585356e-01, ...,
            6.34984963e-06, 7.53185031e-06, 3.67787597e-03],
           [7.24390090e-01, 1.92497107e-09, 5.23109738e-05, ...,
            1.22913450e-03, 1.57926350e-09, 9.63957209e-05]], dtype=float32)



Pekala, elimizde gerçekten çok küçük sayılar olan bir grup tensör var gibi görünüyor, bunlardan birini yakınlaştırmaya ne dersiniz?


```python
print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like:\n {pred_probs[0]}")
print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")
```

    Number of prediction probabilities for sample 0: 101
    What prediction probability sample 0 looks like:
     [5.9542008e-02 3.5741950e-06 4.1376859e-02 1.0660556e-09 8.1613978e-09
     8.6639664e-09 8.0926822e-07 8.5652499e-07 1.9859017e-05 8.0977776e-07
     3.1727747e-09 9.8673661e-07 2.8532164e-04 7.8049051e-10 7.4230169e-04
     3.8916416e-05 6.4740193e-06 2.4977280e-06 3.7891099e-05 2.0678388e-07
     1.5538422e-05 8.1506943e-07 2.6230446e-06 2.0010630e-07 8.3827456e-07
     5.4215989e-06 3.7390860e-06 1.3150533e-08 2.7761406e-03 2.8051838e-05
     6.8562162e-10 2.5574835e-05 1.6688865e-04 7.6407297e-10 4.0452729e-04
     1.3150634e-08 1.7957379e-06 1.4448218e-06 2.3062859e-02 8.2466784e-07
     8.5365781e-07 1.7138614e-06 7.0525107e-06 1.8402169e-08 2.8553407e-07
     7.9483234e-06 2.0681514e-06 1.8525066e-07 3.3619774e-08 3.1522498e-04
     1.0410913e-05 8.5448539e-07 8.4741873e-01 1.0555415e-05 4.4094671e-07
     3.7404148e-05 3.5306231e-05 3.2489133e-05 6.7314817e-05 1.2852616e-08
     2.6219660e-10 1.0318080e-05 8.5744046e-05 1.0569896e-06 2.1293374e-06
     3.7637557e-05 7.5973162e-08 2.5340563e-04 9.2905600e-07 1.2598126e-04
     6.2621725e-06 1.2458752e-08 4.0519579e-05 6.8727985e-08 1.2546318e-06
     5.2887291e-08 7.5425071e-08 7.5398362e-05 7.7540375e-05 6.4025829e-07
     9.9033400e-07 2.2225820e-05 1.5013893e-05 1.4038504e-07 1.2232545e-05
     1.9044733e-02 4.9999417e-05 4.6226096e-06 1.5388227e-07 3.3824102e-07
     3.9228336e-09 1.6563691e-07 8.1320686e-05 4.8965021e-06 2.4068285e-07
     2.3124028e-05 3.1040650e-04 3.1379946e-05 1.4138681e-09 8.3530758e-05
     3.0897427e-03]
    The class with the highest predicted probability by the model for sample 0: 52


Daha önce tartıştığımız gibi, modelimize geçtiğimiz her görüntü tensörü için, çıktı nöronlarının sayısı ve son katmandaki aktivasyon fonksiyonu nedeniyle (`layers.Dense(len(train_data_all_10_percent.class_names), activation="softmax"`) çıktı verir. 101 sınıfın her biri için 0 ile 1 arasında bir tahmin olasılığı vardır.

Ve en yüksek tahmin olasılığının endeksi, modelin en olası etiket olduğunu düşündüğü şey olarak düşünülebilir. Benzer şekilde, tahmin olasılık değeri ne kadar düşükse, model hedef görüntünün o belirli sınıf olduğunu o kadar az düşünür.

🔑 Not: Softmax aktivasyon fonksiyonunun doğası gereği, tek bir örnek için tahmin olasılıklarının her birinin toplamı 1 (veya en azından 1'e çok yakın) olacaktır. Örneğin. pred_probs[0].sum() = 1.

`argmax()` yöntemini kullanarak her tahmin olasılık tensöründeki maksimum değerin indeksini bulabiliriz.


```python
pred_classes = pred_probs.argmax(axis=1)
pred_classes[:10]
```




    array([52,  0,  0, 80, 79, 61, 29,  0, 85,  0])



Güzel! Artık test veri kümemizdeki örneklerin her biri için tahmin edilen sınıf indeksine sahibiz.

Modelimizi daha fazla değerlendirmek için bunları test veri kümesi etiketleriyle karşılaştırabileceğiz.

Test veri kümesi etiketlerini almak için `unbatch()` yöntemini kullanarak `test_data` nesnemizi (bir `tf.data.Dataset` biçimindedir) çözebiliriz.

Bunu yapmak, test veri setindeki resimlere ve etiketlere erişmemizi sağlayacaktır. Etiketler tek sıcak (hot-encoding) kodlanmış biçimde olduğundan, etiketin dizinini döndürmek için `argmax()` yöntemini kullanacağız.

> 🔑 Not: Bu çözülme, test verisi nesnesini oluştururken `shuffle=False` yapmamızın nedenidir. Aksi takdirde, test veri setini her yüklediğimizde (tahmin yaparken olduğu gibi), her seferinde karıştırılacaktı, yani tahminlerimizi etiketlerle karşılaştırmaya çalışsaydık, farklı sırada olacaklardı.


```python
y_labels = []
for images, labels in test_data.unbatch():
  y_labels.append(labels.numpy().argmax())
y_labels[:10]
```




    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



Güzel! `test_data` karıştırılmadığından, `y_labels` dizisi, `pred_classes` dizisiyle aynı sırada geri gelir.

Son kontrol, kaç etiketimiz olduğunu görmek.


```python
len(y_labels)
```




    25250



Beklendiği gibi, etiket sayısı elimizdeki görüntü sayısıyla eşleşiyor. Modelimizin tahminlerini temel gerçek etiketleriyle karşılaştırma zamanı.

## Model Tahminlerimizi Değerlendirme

Çok basit bir değerlendirme, doğruluk etiketlerini tahmin edilen etiketlerle karşılaştıran ve bir doğruluk puanı döndüren Scikit-Learn'ün `accuracy_score()` işlevini kullanmaktır.

`y_labels` ve `pred_classes` dizilerimizi doğru bir şekilde oluşturduysak, bu, daha önce kullandığımız `evaluate()` yöntemiyle aynı doğruluk değerini (veya en azından çok yakın) döndürmelidir.


```python
from sklearn.metrics import accuracy_score
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
sklearn_accuracy
```




    0.6077623762376237




```python
import numpy as np
print(f"Close? {np.isclose(loaded_accuracy, sklearn_accuracy)} | Difference: {loaded_accuracy - sklearn_accuracy}")
```

    Close? True | Difference: 2.0097978059574473e-08


Tamam, `pred_classes` dizimiz ve `y_labels` dizilerimiz doğru sırada görünüyor.

Karışıklık matrisi ile biraz daha görsellik katmaya ne dersiniz?

Bunu yapmak için, önceki bir not defterinde oluşturduğumuz `make_confusion_matrix` işlevimizi kullanacağız.

Şu anda tahminlerimiz ve doğruluk etiketlerimiz tamsayılar biçimindedir, ancak gerçek adlarını alırsak anlamak çok daha kolay olacaktır. Bunu, `test_data` nesnemizde `class_names` niteliğini kullanarak yapabiliriz.



```python
class_names = test_data.class_names
class_names
```




    ['apple_pie',
     'baby_back_ribs',
     'baklava',
     'beef_carpaccio',
     'beef_tartare',
     'beet_salad',
     'beignets',
     'bibimbap',
     'bread_pudding',
     'breakfast_burrito',
     'bruschetta',
     'caesar_salad',
     'cannoli',
     'caprese_salad',
     'carrot_cake',
     'ceviche',
     'cheese_plate',
     'cheesecake',
     'chicken_curry',
     'chicken_quesadilla',
     'chicken_wings',
     'chocolate_cake',
     'chocolate_mousse',
     'churros',
     'clam_chowder',
     'club_sandwich',
     'crab_cakes',
     'creme_brulee',
     'croque_madame',
     'cup_cakes',
     'deviled_eggs',
     'donuts',
     'dumplings',
     'edamame',
     'eggs_benedict',
     'escargots',
     'falafel',
     'filet_mignon',
     'fish_and_chips',
     'foie_gras',
     'french_fries',
     'french_onion_soup',
     'french_toast',
     'fried_calamari',
     'fried_rice',
     'frozen_yogurt',
     'garlic_bread',
     'gnocchi',
     'greek_salad',
     'grilled_cheese_sandwich',
     'grilled_salmon',
     'guacamole',
     'gyoza',
     'hamburger',
     'hot_and_sour_soup',
     'hot_dog',
     'huevos_rancheros',
     'hummus',
     'ice_cream',
     'lasagna',
     'lobster_bisque',
     'lobster_roll_sandwich',
     'macaroni_and_cheese',
     'macarons',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'onion_rings',
     'oysters',
     'pad_thai',
     'paella',
     'pancakes',
     'panna_cotta',
     'peking_duck',
     'pho',
     'pizza',
     'pork_chop',
     'poutine',
     'prime_rib',
     'pulled_pork_sandwich',
     'ramen',
     'ravioli',
     'red_velvet_cake',
     'risotto',
     'samosa',
     'sashimi',
     'scallops',
     'seaweed_salad',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring_rolls',
     'steak',
     'strawberry_shortcake',
     'sushi',
     'tacos',
     'takoyaki',
     'tiramisu',
     'tuna_tartare',
     'waffles']




```python
make_confusion_matrix(y_true=y_labels,
                      y_pred=pred_classes,
                      classes=class_names,
                      figsize=(100, 100),
                      text_size=20,
                      norm=False,
                      savefig=True)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_69_0.png)
    


Vay! Şimdi bu büyük bir karışıklık matrisi. İlk başta biraz ürkütücü görünebilir ama biraz yakınlaştırdıktan sonra, hangi sınıfların "kafasının karıştığı" konusunda bize nasıl fikir verdiğini görebiliriz.

İyi haber şu ki, tahminlerin çoğu sol üstten sağ alt köşeye doğru, yani doğrular.

Modelin kafası en çok, pig_chop örnekleri için filet_mignon ve tiramisu örnekleri için Chocolate_cake gibi görsel olarak benzer görünen sınıflarda karışıyor gibi görünüyor.

Bir sınıflandırma problemi üzerinde çalıştığımız için, Scikit-Learn'in `classification_report()` işlevini kullanarak modelimizin tahminlerini daha da değerlendirebiliriz.



```python
from sklearn.metrics import classification_report
print(classification_report(y_labels, pred_classes))
```

                  precision    recall  f1-score   support
    
               0       0.29      0.20      0.24       250
               1       0.51      0.69      0.59       250
               2       0.56      0.65      0.60       250
               3       0.74      0.53      0.62       250
               4       0.73      0.43      0.54       250
               5       0.34      0.54      0.42       250
               6       0.67      0.79      0.72       250
               7       0.82      0.76      0.79       250
               8       0.40      0.37      0.39       250
               9       0.62      0.44      0.51       250
              10       0.62      0.42      0.50       250
              11       0.84      0.49      0.62       250
              12       0.52      0.74      0.61       250
              13       0.56      0.60      0.58       250
              14       0.56      0.59      0.57       250
              15       0.44      0.32      0.37       250
              16       0.45      0.75      0.57       250
              17       0.37      0.51      0.43       250
              18       0.43      0.60      0.50       250
              19       0.68      0.60      0.64       250
              20       0.68      0.75      0.71       250
              21       0.35      0.64      0.45       250
              22       0.30      0.37      0.33       250
              23       0.66      0.77      0.71       250
              24       0.83      0.72      0.77       250
              25       0.76      0.71      0.73       250
              26       0.51      0.42      0.46       250
              27       0.78      0.72      0.75       250
              28       0.70      0.69      0.69       250
              29       0.70      0.68      0.69       250
              30       0.92      0.63      0.75       250
              31       0.78      0.70      0.74       250
              32       0.75      0.83      0.79       250
              33       0.89      0.98      0.94       250
              34       0.68      0.78      0.72       250
              35       0.78      0.66      0.72       250
              36       0.53      0.56      0.55       250
              37       0.30      0.55      0.39       250
              38       0.78      0.63      0.69       250
              39       0.27      0.33      0.30       250
              40       0.72      0.81      0.76       250
              41       0.81      0.62      0.70       250
              42       0.50      0.58      0.54       250
              43       0.75      0.60      0.67       250
              44       0.74      0.45      0.56       250
              45       0.77      0.85      0.81       250
              46       0.81      0.46      0.58       250
              47       0.44      0.49      0.46       250
              48       0.45      0.81      0.58       250
              49       0.50      0.44      0.47       250
              50       0.54      0.39      0.46       250
              51       0.71      0.86      0.78       250
              52       0.51      0.77      0.61       250
              53       0.67      0.68      0.68       250
              54       0.88      0.75      0.81       250
              55       0.86      0.69      0.76       250
              56       0.56      0.24      0.34       250
              57       0.62      0.45      0.52       250
              58       0.68      0.58      0.62       250
              59       0.70      0.37      0.49       250
              60       0.83      0.59      0.69       250
              61       0.54      0.81      0.65       250
              62       0.72      0.49      0.58       250
              63       0.94      0.86      0.90       250
              64       0.78      0.85      0.81       250
              65       0.82      0.82      0.82       250
              66       0.69      0.32      0.44       250
              67       0.41      0.58      0.48       250
              68       0.90      0.78      0.83       250
              69       0.84      0.82      0.83       250
              70       0.62      0.83      0.71       250
              71       0.81      0.46      0.59       250
              72       0.64      0.65      0.65       250
              73       0.51      0.44      0.47       250
              74       0.72      0.61      0.66       250
              75       0.84      0.90      0.87       250
              76       0.78      0.78      0.78       250
              77       0.36      0.27      0.31       250
              78       0.79      0.74      0.76       250
              79       0.44      0.81      0.57       250
              80       0.57      0.60      0.59       250
              81       0.65      0.70      0.68       250
              82       0.38      0.31      0.34       250
              83       0.58      0.80      0.67       250
              84       0.61      0.38      0.47       250
              85       0.44      0.74      0.55       250
              86       0.71      0.86      0.78       250
              87       0.41      0.39      0.40       250
              88       0.83      0.80      0.81       250
              89       0.71      0.31      0.43       250
              90       0.92      0.69      0.79       250
              91       0.83      0.87      0.85       250
              92       0.68      0.65      0.67       250
              93       0.31      0.38      0.34       250
              94       0.61      0.54      0.57       250
              95       0.74      0.61      0.67       250
              96       0.56      0.29      0.38       250
              97       0.45      0.74      0.56       250
              98       0.47      0.33      0.39       250
              99       0.52      0.27      0.35       250
             100       0.59      0.70      0.64       250
    
        accuracy                           0.61     25250
       macro avg       0.63      0.61      0.61     25250
    weighted avg       0.63      0.61      0.61     25250
    


`classification_report()`, sınıf başına kesinlik, geri çağırma ve f1-skorunun çıktısını verir.

Bir hatırlatıcı:

- **Precision** - Gerçek pozitiflerin toplam numune sayısına oranı. Daha yüksek hassasiyet, daha az yanlış pozitife yol açar (model 0 olması gerektiğinde 1'i tahmin eder).
- **Recall** - Gerçek pozitiflerin toplam gerçek pozitif ve yanlış negatif sayısına oranı (model, 1 olması gerektiğinde 0'ı tahmin eder). Daha yüksek hatırlama, daha az yanlış negatife yol açar.
- **F1 Score **- Kesinlik ve hatırlamayı tek bir metrikte birleştirir. 1 en iyisidir, 0 en kötüsüdür.

Yukarıdaki çıktı yardımcı olur, ancak bu kadar çok sınıfla anlaşılması biraz zor.

Bir görselleştirme yardımıyla bunu kolaylaştırıp kolaylaştırmayacağımıza bakalım.

İlk olarak, `output_dict=True` ayarını yaparak `classification_report()` çıktısını sözlük olarak alacağız.


```python
classification_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
classification_report_dict
```




    {'0': {'f1-score': 0.24056603773584903,
      'precision': 0.29310344827586204,
      'recall': 0.204,
      'support': 250},
     '1': {'f1-score': 0.5864406779661017,
      'precision': 0.5088235294117647,
      'recall': 0.692,
      'support': 250},
     '10': {'f1-score': 0.5047619047619047,
      'precision': 0.6235294117647059,
      'recall': 0.424,
      'support': 250},
     '100': {'f1-score': 0.641025641025641,
      'precision': 0.5912162162162162,
      'recall': 0.7,
      'support': 250},
     '11': {'f1-score': 0.6161616161616161,
      'precision': 0.8356164383561644,
      'recall': 0.488,
      'support': 250},
     '12': {'f1-score': 0.6105610561056106,
      'precision': 0.5196629213483146,
      'recall': 0.74,
      'support': 250},
     '13': {'f1-score': 0.5775193798449612,
      'precision': 0.5601503759398496,
      'recall': 0.596,
      'support': 250},
     '14': {'f1-score': 0.574757281553398,
      'precision': 0.5584905660377358,
      'recall': 0.592,
      'support': 250},
     '15': {'f1-score': 0.36744186046511623,
      'precision': 0.4388888888888889,
      'recall': 0.316,
      'support': 250},
     '16': {'f1-score': 0.5654135338345864,
      'precision': 0.4530120481927711,
      'recall': 0.752,
      'support': 250},
     '17': {'f1-score': 0.42546063651591287,
      'precision': 0.3659942363112392,
      'recall': 0.508,
      'support': 250},
     '18': {'f1-score': 0.5008403361344538,
      'precision': 0.4318840579710145,
      'recall': 0.596,
      'support': 250},
     '19': {'f1-score': 0.6411889596602972,
      'precision': 0.6832579185520362,
      'recall': 0.604,
      'support': 250},
     '2': {'f1-score': 0.6022304832713754,
      'precision': 0.5625,
      'recall': 0.648,
      'support': 250},
     '20': {'f1-score': 0.7123809523809523,
      'precision': 0.68,
      'recall': 0.748,
      'support': 250},
     '21': {'f1-score': 0.45261669024045265,
      'precision': 0.350109409190372,
      'recall': 0.64,
      'support': 250},
     '22': {'f1-score': 0.3291592128801431,
      'precision': 0.2977346278317152,
      'recall': 0.368,
      'support': 250},
     '23': {'f1-score': 0.7134935304990757,
      'precision': 0.6632302405498282,
      'recall': 0.772,
      'support': 250},
     '24': {'f1-score': 0.7708779443254817,
      'precision': 0.8294930875576036,
      'recall': 0.72,
      'support': 250},
     '25': {'f1-score': 0.734020618556701,
      'precision': 0.7574468085106383,
      'recall': 0.712,
      'support': 250},
     '26': {'f1-score': 0.4625550660792952,
      'precision': 0.5147058823529411,
      'recall': 0.42,
      'support': 250},
     '27': {'f1-score': 0.7494824016563146,
      'precision': 0.776824034334764,
      'recall': 0.724,
      'support': 250},
     '28': {'f1-score': 0.6935483870967742,
      'precision': 0.6991869918699187,
      'recall': 0.688,
      'support': 250},
     '29': {'f1-score': 0.6910569105691057,
      'precision': 0.7024793388429752,
      'recall': 0.68,
      'support': 250},
     '3': {'f1-score': 0.616822429906542,
      'precision': 0.7415730337078652,
      'recall': 0.528,
      'support': 250},
     '30': {'f1-score': 0.7476190476190476,
      'precision': 0.9235294117647059,
      'recall': 0.628,
      'support': 250},
     '31': {'f1-score': 0.7357293868921776,
      'precision': 0.7802690582959642,
      'recall': 0.696,
      'support': 250},
     '32': {'f1-score': 0.7855787476280836,
      'precision': 0.7472924187725631,
      'recall': 0.828,
      'support': 250},
     '33': {'f1-score': 0.9371428571428572,
      'precision': 0.8945454545454545,
      'recall': 0.984,
      'support': 250},
     '34': {'f1-score': 0.7238805970149255,
      'precision': 0.6783216783216783,
      'recall': 0.776,
      'support': 250},
     '35': {'f1-score': 0.715835140997831,
      'precision': 0.7819905213270142,
      'recall': 0.66,
      'support': 250},
     '36': {'f1-score': 0.5475728155339805,
      'precision': 0.5320754716981132,
      'recall': 0.564,
      'support': 250},
     '37': {'f1-score': 0.3870056497175141,
      'precision': 0.29912663755458513,
      'recall': 0.548,
      'support': 250},
     '38': {'f1-score': 0.6946902654867257,
      'precision': 0.7772277227722773,
      'recall': 0.628,
      'support': 250},
     '39': {'f1-score': 0.29749103942652333,
      'precision': 0.2694805194805195,
      'recall': 0.332,
      'support': 250},
     '4': {'f1-score': 0.544080604534005,
      'precision': 0.7346938775510204,
      'recall': 0.432,
      'support': 250},
     '40': {'f1-score': 0.7622641509433963,
      'precision': 0.7214285714285714,
      'recall': 0.808,
      'support': 250},
     '41': {'f1-score': 0.7029478458049886,
      'precision': 0.8115183246073299,
      'recall': 0.62,
      'support': 250},
     '42': {'f1-score': 0.537037037037037,
      'precision': 0.5,
      'recall': 0.58,
      'support': 250},
     '43': {'f1-score': 0.6651884700665188,
      'precision': 0.746268656716418,
      'recall': 0.6,
      'support': 250},
     '44': {'f1-score': 0.5586034912718205,
      'precision': 0.7417218543046358,
      'recall': 0.448,
      'support': 250},
     '45': {'f1-score': 0.8114285714285714,
      'precision': 0.7745454545454545,
      'recall': 0.852,
      'support': 250},
     '46': {'f1-score': 0.5831202046035805,
      'precision': 0.8085106382978723,
      'recall': 0.456,
      'support': 250},
     '47': {'f1-score': 0.4641509433962264,
      'precision': 0.4392857142857143,
      'recall': 0.492,
      'support': 250},
     '48': {'f1-score': 0.577524893314367,
      'precision': 0.4481236203090508,
      'recall': 0.812,
      'support': 250},
     '49': {'f1-score': 0.47234042553191485,
      'precision': 0.5045454545454545,
      'recall': 0.444,
      'support': 250},
     '5': {'f1-score': 0.41860465116279066,
      'precision': 0.34177215189873417,
      'recall': 0.54,
      'support': 250},
     '50': {'f1-score': 0.45581395348837206,
      'precision': 0.5444444444444444,
      'recall': 0.392,
      'support': 250},
     '51': {'f1-score': 0.7783783783783783,
      'precision': 0.7081967213114754,
      'recall': 0.864,
      'support': 250},
     '52': {'f1-score': 0.6124401913875598,
      'precision': 0.5092838196286472,
      'recall': 0.768,
      'support': 250},
     '53': {'f1-score': 0.6759443339960238,
      'precision': 0.6719367588932806,
      'recall': 0.68,
      'support': 250},
     '54': {'f1-score': 0.8103448275862069,
      'precision': 0.8785046728971962,
      'recall': 0.752,
      'support': 250},
     '55': {'f1-score': 0.7644444444444444,
      'precision': 0.86,
      'recall': 0.688,
      'support': 250},
     '56': {'f1-score': 0.3398328690807799,
      'precision': 0.5596330275229358,
      'recall': 0.244,
      'support': 250},
     '57': {'f1-score': 0.5209302325581396,
      'precision': 0.6222222222222222,
      'recall': 0.448,
      'support': 250},
     '58': {'f1-score': 0.6233766233766233,
      'precision': 0.6792452830188679,
      'recall': 0.576,
      'support': 250},
     '59': {'f1-score': 0.486910994764398,
      'precision': 0.7045454545454546,
      'recall': 0.372,
      'support': 250},
     '6': {'f1-score': 0.7229357798165138,
      'precision': 0.6677966101694915,
      'recall': 0.788,
      'support': 250},
     '60': {'f1-score': 0.6885245901639344,
      'precision': 0.8305084745762712,
      'recall': 0.588,
      'support': 250},
     '61': {'f1-score': 0.6495176848874598,
      'precision': 0.543010752688172,
      'recall': 0.808,
      'support': 250},
     '62': {'f1-score': 0.5823389021479712,
      'precision': 0.7218934911242604,
      'recall': 0.488,
      'support': 250},
     '63': {'f1-score': 0.895397489539749,
      'precision': 0.9385964912280702,
      'recall': 0.856,
      'support': 250},
     '64': {'f1-score': 0.8129770992366412,
      'precision': 0.7773722627737226,
      'recall': 0.852,
      'support': 250},
     '65': {'f1-score': 0.82, 'precision': 0.82, 'recall': 0.82, 'support': 250},
     '66': {'f1-score': 0.44141689373297005,
      'precision': 0.6923076923076923,
      'recall': 0.324,
      'support': 250},
     '67': {'f1-score': 0.47840531561461797,
      'precision': 0.4090909090909091,
      'recall': 0.576,
      'support': 250},
     '68': {'f1-score': 0.832618025751073,
      'precision': 0.8981481481481481,
      'recall': 0.776,
      'support': 250},
     '69': {'f1-score': 0.8340080971659919,
      'precision': 0.8442622950819673,
      'recall': 0.824,
      'support': 250},
     '7': {'f1-score': 0.7908902691511386,
      'precision': 0.8197424892703863,
      'recall': 0.764,
      'support': 250},
     '70': {'f1-score': 0.7101200686106347,
      'precision': 0.6216216216216216,
      'recall': 0.828,
      'support': 250},
     '71': {'f1-score': 0.5903307888040712,
      'precision': 0.8111888111888111,
      'recall': 0.464,
      'support': 250},
     '72': {'f1-score': 0.6468253968253969,
      'precision': 0.6417322834645669,
      'recall': 0.652,
      'support': 250},
     '73': {'f1-score': 0.4743589743589744,
      'precision': 0.5091743119266054,
      'recall': 0.444,
      'support': 250},
     '74': {'f1-score': 0.658008658008658,
      'precision': 0.7169811320754716,
      'recall': 0.608,
      'support': 250},
     '75': {'f1-score': 0.8665377176015473,
      'precision': 0.8389513108614233,
      'recall': 0.896,
      'support': 250},
     '76': {'f1-score': 0.7808764940239045,
      'precision': 0.7777777777777778,
      'recall': 0.784,
      'support': 250},
     '77': {'f1-score': 0.30875576036866365,
      'precision': 0.3641304347826087,
      'recall': 0.268,
      'support': 250},
     '78': {'f1-score': 0.7603305785123966,
      'precision': 0.7863247863247863,
      'recall': 0.736,
      'support': 250},
     '79': {'f1-score': 0.571830985915493,
      'precision': 0.44130434782608696,
      'recall': 0.812,
      'support': 250},
     '8': {'f1-score': 0.3866943866943867,
      'precision': 0.4025974025974026,
      'recall': 0.372,
      'support': 250},
     '80': {'f1-score': 0.5870841487279843,
      'precision': 0.5747126436781609,
      'recall': 0.6,
      'support': 250},
     '81': {'f1-score': 0.6756756756756757,
      'precision': 0.6529850746268657,
      'recall': 0.7,
      'support': 250},
     '82': {'f1-score': 0.34285714285714286,
      'precision': 0.3804878048780488,
      'recall': 0.312,
      'support': 250},
     '83': {'f1-score': 0.6711409395973154,
      'precision': 0.5780346820809249,
      'recall': 0.8,
      'support': 250},
     '84': {'f1-score': 0.4653465346534653,
      'precision': 0.6103896103896104,
      'recall': 0.376,
      'support': 250},
     '85': {'f1-score': 0.5525525525525525,
      'precision': 0.4423076923076923,
      'recall': 0.736,
      'support': 250},
     '86': {'f1-score': 0.7783783783783783,
      'precision': 0.7081967213114754,
      'recall': 0.864,
      'support': 250},
     '87': {'f1-score': 0.3975409836065574,
      'precision': 0.40756302521008403,
      'recall': 0.388,
      'support': 250},
     '88': {'f1-score': 0.8130081300813008,
      'precision': 0.8264462809917356,
      'recall': 0.8,
      'support': 250},
     '89': {'f1-score': 0.4301675977653631,
      'precision': 0.7129629629629629,
      'recall': 0.308,
      'support': 250},
     '9': {'f1-score': 0.5117370892018779,
      'precision': 0.6193181818181818,
      'recall': 0.436,
      'support': 250},
     '90': {'f1-score': 0.7881548974943051,
      'precision': 0.9153439153439153,
      'recall': 0.692,
      'support': 250},
     '91': {'f1-score': 0.84765625,
      'precision': 0.8282442748091603,
      'recall': 0.868,
      'support': 250},
     '92': {'f1-score': 0.6652977412731006,
      'precision': 0.6835443037974683,
      'recall': 0.648,
      'support': 250},
     '93': {'f1-score': 0.34234234234234234,
      'precision': 0.3114754098360656,
      'recall': 0.38,
      'support': 250},
     '94': {'f1-score': 0.5714285714285714,
      'precision': 0.6118721461187214,
      'recall': 0.536,
      'support': 250},
     '95': {'f1-score': 0.6710526315789473,
      'precision': 0.7427184466019418,
      'recall': 0.612,
      'support': 250},
     '96': {'f1-score': 0.3809523809523809,
      'precision': 0.5625,
      'recall': 0.288,
      'support': 250},
     '97': {'f1-score': 0.5644916540212443,
      'precision': 0.4547677261613692,
      'recall': 0.744,
      'support': 250},
     '98': {'f1-score': 0.3858823529411765,
      'precision': 0.4685714285714286,
      'recall': 0.328,
      'support': 250},
     '99': {'f1-score': 0.35356200527704484,
      'precision': 0.5193798449612403,
      'recall': 0.268,
      'support': 250},
     'accuracy': 0.6077623762376237,
     'macro avg': {'f1-score': 0.6061252197245781,
      'precision': 0.6328666845830312,
      'recall': 0.6077623762376237,
      'support': 25250},
     'weighted avg': {'f1-score': 0.606125219724578,
      'precision': 0.6328666845830311,
      'recall': 0.6077623762376237,
      'support': 25250}}



Pekala, burada hala birkaç değer var, biraz daraltmaya ne dersiniz?

f1-skoru, recall ve precission tek bir metrikte birleştirdiğinden, buna odaklanalım.

Bunu ayıklamak için `class_f1_scores` adında boş bir sözlük oluşturacağız ve ardından sınıf adını ve `f1-score`'u anahtar olarak, class_f1_scores içindeki değer çiftlerini ekleyerek `classification_report_dict` içindeki her öğe arasında dolaşacağız.



```python
class_f1_scores = {}
for k, v in classification_report_dict.items():
  if k == "accuracy":
    break
  else:
    class_f1_scores[class_names[int(k)]] = v["f1-score"]
class_f1_scores
```




    {'apple_pie': 0.24056603773584903,
     'baby_back_ribs': 0.5864406779661017,
     'baklava': 0.6022304832713754,
     'beef_carpaccio': 0.616822429906542,
     'beef_tartare': 0.544080604534005,
     'beet_salad': 0.41860465116279066,
     'beignets': 0.7229357798165138,
     'bibimbap': 0.7908902691511386,
     'bread_pudding': 0.3866943866943867,
     'breakfast_burrito': 0.5117370892018779,
     'bruschetta': 0.5047619047619047,
     'caesar_salad': 0.6161616161616161,
     'cannoli': 0.6105610561056106,
     'caprese_salad': 0.5775193798449612,
     'carrot_cake': 0.574757281553398,
     'ceviche': 0.36744186046511623,
     'cheese_plate': 0.5654135338345864,
     'cheesecake': 0.42546063651591287,
     'chicken_curry': 0.5008403361344538,
     'chicken_quesadilla': 0.6411889596602972,
     'chicken_wings': 0.7123809523809523,
     'chocolate_cake': 0.45261669024045265,
     'chocolate_mousse': 0.3291592128801431,
     'churros': 0.7134935304990757,
     'clam_chowder': 0.7708779443254817,
     'club_sandwich': 0.734020618556701,
     'crab_cakes': 0.4625550660792952,
     'creme_brulee': 0.7494824016563146,
     'croque_madame': 0.6935483870967742,
     'cup_cakes': 0.6910569105691057,
     'deviled_eggs': 0.7476190476190476,
     'donuts': 0.7357293868921776,
     'dumplings': 0.7855787476280836,
     'edamame': 0.9371428571428572,
     'eggs_benedict': 0.7238805970149255,
     'escargots': 0.715835140997831,
     'falafel': 0.5475728155339805,
     'filet_mignon': 0.3870056497175141,
     'fish_and_chips': 0.6946902654867257,
     'foie_gras': 0.29749103942652333,
     'french_fries': 0.7622641509433963,
     'french_onion_soup': 0.7029478458049886,
     'french_toast': 0.537037037037037,
     'fried_calamari': 0.6651884700665188,
     'fried_rice': 0.5586034912718205,
     'frozen_yogurt': 0.8114285714285714,
     'garlic_bread': 0.5831202046035805,
     'gnocchi': 0.4641509433962264,
     'greek_salad': 0.577524893314367,
     'grilled_cheese_sandwich': 0.47234042553191485,
     'grilled_salmon': 0.45581395348837206,
     'guacamole': 0.7783783783783783,
     'gyoza': 0.6124401913875598,
     'hamburger': 0.6759443339960238,
     'hot_and_sour_soup': 0.8103448275862069,
     'hot_dog': 0.7644444444444444,
     'huevos_rancheros': 0.3398328690807799,
     'hummus': 0.5209302325581396,
     'ice_cream': 0.6233766233766233,
     'lasagna': 0.486910994764398,
     'lobster_bisque': 0.6885245901639344,
     'lobster_roll_sandwich': 0.6495176848874598,
     'macaroni_and_cheese': 0.5823389021479712,
     'macarons': 0.895397489539749,
     'miso_soup': 0.8129770992366412,
     'mussels': 0.82,
     'nachos': 0.44141689373297005,
     'omelette': 0.47840531561461797,
     'onion_rings': 0.832618025751073,
     'oysters': 0.8340080971659919,
     'pad_thai': 0.7101200686106347,
     'paella': 0.5903307888040712,
     'pancakes': 0.6468253968253969,
     'panna_cotta': 0.4743589743589744,
     'peking_duck': 0.658008658008658,
     'pho': 0.8665377176015473,
     'pizza': 0.7808764940239045,
     'pork_chop': 0.30875576036866365,
     'poutine': 0.7603305785123966,
     'prime_rib': 0.571830985915493,
     'pulled_pork_sandwich': 0.5870841487279843,
     'ramen': 0.6756756756756757,
     'ravioli': 0.34285714285714286,
     'red_velvet_cake': 0.6711409395973154,
     'risotto': 0.4653465346534653,
     'samosa': 0.5525525525525525,
     'sashimi': 0.7783783783783783,
     'scallops': 0.3975409836065574,
     'seaweed_salad': 0.8130081300813008,
     'shrimp_and_grits': 0.4301675977653631,
     'spaghetti_bolognese': 0.7881548974943051,
     'spaghetti_carbonara': 0.84765625,
     'spring_rolls': 0.6652977412731006,
     'steak': 0.34234234234234234,
     'strawberry_shortcake': 0.5714285714285714,
     'sushi': 0.6710526315789473,
     'tacos': 0.3809523809523809,
     'takoyaki': 0.5644916540212443,
     'tiramisu': 0.3858823529411765,
     'tuna_tartare': 0.35356200527704484,
     'waffles': 0.641025641025641}



İyi görünüyor!

Görünüşe göre sözlüğümüz sınıf isimlerine göre sıralanmış. Ancak, farklı puanları görselleştirmeye çalışıyorsak, bir tür düzende olmaları daha hoş görünebilir.

class_f1_scores sözlüğümüzü bir panda DataFrame'e dönüştürelim ve artan biçimde sıralayalım mı?



```python
import pandas as pd
f1_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)
f1_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_name</th>
      <th>f1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>edamame</td>
      <td>0.937143</td>
    </tr>
    <tr>
      <th>63</th>
      <td>macarons</td>
      <td>0.895397</td>
    </tr>
    <tr>
      <th>75</th>
      <td>pho</td>
      <td>0.866538</td>
    </tr>
    <tr>
      <th>91</th>
      <td>spaghetti_carbonara</td>
      <td>0.847656</td>
    </tr>
    <tr>
      <th>69</th>
      <td>oysters</td>
      <td>0.834008</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>huevos_rancheros</td>
      <td>0.339833</td>
    </tr>
    <tr>
      <th>22</th>
      <td>chocolate_mousse</td>
      <td>0.329159</td>
    </tr>
    <tr>
      <th>77</th>
      <td>pork_chop</td>
      <td>0.308756</td>
    </tr>
    <tr>
      <th>39</th>
      <td>foie_gras</td>
      <td>0.297491</td>
    </tr>
    <tr>
      <th>0</th>
      <td>apple_pie</td>
      <td>0.240566</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>




```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(list(f1_scores["class_name"]))
ax.set_xlabel("f1-score")
ax.set_title("F1-Scores for 10 Different Classes")
ax.invert_yaxis();

def autolabel(rects): 
  for rect in rects:
    width = rect.get_width()
    ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
            f"{width:.2f}",
            ha='center', va='bottom')

autolabel(scores)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_78_0.png)
    


Şimdi bu iyi görünen bir grafik! Yani, metin konumlandırma biraz geliştirilebilir ama şimdilik idare eder.

Modelimizin tahminlerini görselleştirmenin bize performansına dair tamamen yeni bir fikir verdiğini görebiliyor musunuz?

Birkaç dakika önce yalnızca bir doğruluk puanımız vardı, ancak şimdi modelimizin sınıf bazında ne kadar iyi performans gösterdiğinin bir göstergesine sahibiz.

Görünüşe göre modelimiz apple_pie ve ravioli gibi sınıflarda oldukça düşük performans gösterirken, edamame ve pho gibi sınıflar için performans oldukça yüksek.

Bunun gibi bulgular bize deneylerimizde nereye gidebileceğimize dair ipuçları veriyor. Belki de kötü performans gösteren sınıflar hakkında daha fazla veri toplamamız gerekebilir veya belki de en kötü performans gösteren sınıflar hakkında tahminde bulunmak zordur.

## Test Görüntülerinde Tahminleri Görselleştirme

Gerçek test zamanı. Tahminleri gerçek görüntüler üzerinde görselleştirme. İstediğiniz tüm metriklere bakabilirsiniz, ancak bazı tahminleri görselleştirene kadar modelinizin nasıl performans gösterdiğini gerçekten bilemezsiniz.

Halihazırda, modelimiz seçtiğimiz herhangi bir görüntü üzerinde tahminde bulunamaz. Görüntü önce bir tensöre yüklenmelidir.

Bu nedenle, herhangi bir görüntü üzerinde tahminde bulunmaya başlamak için, bir görüntüyü bir tensöre yüklemek için bir fonksiyon oluşturacağız.

Spesifik olarak:

- `tf.io.read_file(`) kullanarak bir hedef görüntü dosya yolunu okuyun.
- `tf.io.decode_image()` kullanarak görüntüyü bir Tensöre dönüştürün.
- Görüntüyü, `tf.image.resize()` kullanarak modelimizin üzerinde çalıştığı (224 x 224) görüntülerle aynı boyutta olacak şekilde yeniden boyutlandırın.
- Gerekirse 0 ve 1 arasındaki tüm piksel değerlerini elde etmek için görüntüyü ölçeklendirin.


```python
def load_and_prep_image(filename, img_shape=224, scale=True):
  img = tf.io.read_file(filename)
  img = tf.io.decode_image(img)
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    return img/255.
  else:
    return img
```

Görüntü yükleme ve ön işleme işlevi hazır.

Şimdi bir kod yazalım:

Test veri kümesinden birkaç rastgele görüntü yükleyin.
Onlarla ilgili tahminlerde bulunun.
Modelin tahmin edilen etiketi, tahmin olasılığı ve kesin doğruluk etiketi ile birlikte orijinal görüntüleri çizin.


```python
import os
import random

plt.figure(figsize=(17, 10))
for i in range(3):
  class_name = random.choice(class_names)
  filename = random.choice(os.listdir(test_dir + "/" + class_name))
  filepath = test_dir + class_name + "/" + filename

  img = load_and_prep_image(filepath, scale=False) 
  pred_prob = model.predict(tf.expand_dims(img, axis=0)) 
  pred_class = class_names[pred_prob.argmax()] 

  plt.subplot(1, 3, i+1)
  plt.imshow(img/255.)
  if class_name == pred_class:
    title_color = "g"
  else:
    title_color = "r"
  plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
  plt.axis(False);
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_83_0.png)
    


Yeterince rastgele örneklemden geçtikten sonra, modelin görsel olarak benzer sınıflarda, baby_back_ribs'in biftekle karıştırılması ve bunun tersi gibi çok daha kötü tahminler yapma eğiliminde olduğu netlik kazanmaya başlar.

## En Yanlış Tahminleri Bulma

Nasıl çalıştığına dair iyi bir fikir edinmek için modelinizin tahminlerinin en az 100'den fazla rastgele örneğini gözden geçirmek iyi bir fikirdir.

Bir süre sonra, modelin bazı görüntülerde çok yüksek tahmin olasılığı ile tahmin yaptığını fark edebilirsiniz, yani tahmininden çok emin ama yine de etiketi yanlış anlıyor.

Bu en yanlış tahminler, modelinizin performansı hakkında daha fazla bilgi edinmenize yardımcı olabilir.

Öyleyse, modelin bir görüntü için yüksek bir tahmin olasılığı verdiği (örneğin 0.95+) ancak tahmini yanlış yaptığı tüm tahminleri toplamak için bir kod yazalım.

Aşağıdaki adımlardan geçeceğiz:

1. `list_files()` yöntemini kullanarak test veri kümesindeki tüm görüntü dosyası yollarını alın.
2. Görüntü dosya yollarının, kesin doğruluk etiketlerinin, tahmin sınıflarının, maksimum tahmin olasılıklarının, kesin gerçek sınıf adlarının ve tahmin edilen sınıf adlarının pandas DataFrame'ini oluşturun.
  - **Not:** Mutlaka böyle bir DataFrame oluşturmamız gerekmez, ancak bu, ilerledikçe işleri görselleştirmemize yardımcı olur.
3. Tüm yanlış tahminleri bulmak için DataFrame'imizi kullanın (temel gerçeğin tahminle eşleşmediği durumlarda).
4. DataFrame'i yanlış tahminlere ve en yüksek maksimum tahmin olasılıklarına göre sıralayın.
5. Görüntüleri en yüksek tahmin olasılıkları ile görselleştirin, ancak yanlış tahmine sahip olun.


```python
# 1.adım
filepaths = []
for filepath in test_data.list_files("101_food_classes_10_percent/test/*/*.jpg", 
                                     shuffle=False):
  filepaths.append(filepath.numpy())
filepaths[:10]
```




    [b'101_food_classes_10_percent/test/apple_pie/1011328.jpg',
     b'101_food_classes_10_percent/test/apple_pie/101251.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1034399.jpg',
     b'101_food_classes_10_percent/test/apple_pie/103801.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1038694.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1047447.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1068632.jpg',
     b'101_food_classes_10_percent/test/apple_pie/110043.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1106961.jpg',
     b'101_food_classes_10_percent/test/apple_pie/1113017.jpg']



Şimdi tüm test görüntü dosya yollarına sahibiz, bunları aşağıdakilerle birlikte bir DataFrame'de birleştirelim:

- Temel doğruluk etiketleri (y_labels).
- Modelin öngördüğü sınıf (pred_classes).
- Maksimum tahmin olasılık değeri (pred_probs.max(axis=1)).
- Temel doğruluk sınıfı adları.
- Öngörülen sınıf adları.




```python
# 2.adım
import pandas as pd
pred_df = pd.DataFrame({"img_path": filepaths,
                        "y_true": y_labels,
                        "y_pred": pred_classes,
                        "pred_conf": pred_probs.max(axis=1),
                        "y_true_classname": [class_names[i] for i in y_labels],
                        "y_pred_classname": [class_names[i] for i in pred_classes]}) 
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>img_path</th>
      <th>y_true</th>
      <th>y_pred</th>
      <th>pred_conf</th>
      <th>y_true_classname</th>
      <th>y_pred_classname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>52</td>
      <td>0.847419</td>
      <td>apple_pie</td>
      <td>gyoza</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>0</td>
      <td>0.964017</td>
      <td>apple_pie</td>
      <td>apple_pie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>0</td>
      <td>0.959259</td>
      <td>apple_pie</td>
      <td>apple_pie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>80</td>
      <td>0.658606</td>
      <td>apple_pie</td>
      <td>pulled_pork_sandwich</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>79</td>
      <td>0.367900</td>
      <td>apple_pie</td>
      <td>prime_rib</td>
    </tr>
  </tbody>
</table>
</div>



Güzel! Tahminin doğru mu yanlış mı olduğunu bize anlatan basit bir sütun yapmaya ne dersiniz?


```python
# 3.adım
pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>img_path</th>
      <th>y_true</th>
      <th>y_pred</th>
      <th>pred_conf</th>
      <th>y_true_classname</th>
      <th>y_pred_classname</th>
      <th>pred_correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>52</td>
      <td>0.847419</td>
      <td>apple_pie</td>
      <td>gyoza</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>0</td>
      <td>0.964017</td>
      <td>apple_pie</td>
      <td>apple_pie</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>0</td>
      <td>0.959259</td>
      <td>apple_pie</td>
      <td>apple_pie</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>80</td>
      <td>0.658606</td>
      <td>apple_pie</td>
      <td>pulled_pork_sandwich</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/1...</td>
      <td>0</td>
      <td>79</td>
      <td>0.367900</td>
      <td>apple_pie</td>
      <td>prime_rib</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Ve şimdi hangi tahminlerin doğru veya yanlış olduğunu ve tahmin olasılıklarıyla birlikte bildiğimize göre, yanlış tahminleri sıralayarak ve tahmin olasılıklarını azaltarak "en yanlış" 100 tahmini elde etmeye ne dersiniz?


```python
# 4.adım
top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:100]
top_100_wrong.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>img_path</th>
      <th>y_true</th>
      <th>y_pred</th>
      <th>pred_conf</th>
      <th>y_true_classname</th>
      <th>y_pred_classname</th>
      <th>pred_correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21810</th>
      <td>b'101_food_classes_10_percent/test/scallops/17...</td>
      <td>87</td>
      <td>29</td>
      <td>0.999997</td>
      <td>scallops</td>
      <td>cup_cakes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>231</th>
      <td>b'101_food_classes_10_percent/test/apple_pie/8...</td>
      <td>0</td>
      <td>100</td>
      <td>0.999995</td>
      <td>apple_pie</td>
      <td>waffles</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15359</th>
      <td>b'101_food_classes_10_percent/test/lobster_rol...</td>
      <td>61</td>
      <td>53</td>
      <td>0.999988</td>
      <td>lobster_roll_sandwich</td>
      <td>hamburger</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23539</th>
      <td>b'101_food_classes_10_percent/test/strawberry_...</td>
      <td>94</td>
      <td>83</td>
      <td>0.999987</td>
      <td>strawberry_shortcake</td>
      <td>red_velvet_cake</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21400</th>
      <td>b'101_food_classes_10_percent/test/samosa/3140...</td>
      <td>85</td>
      <td>92</td>
      <td>0.999981</td>
      <td>samosa</td>
      <td>spring_rolls</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24540</th>
      <td>b'101_food_classes_10_percent/test/tiramisu/16...</td>
      <td>98</td>
      <td>83</td>
      <td>0.999947</td>
      <td>tiramisu</td>
      <td>red_velvet_cake</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2511</th>
      <td>b'101_food_classes_10_percent/test/bruschetta/...</td>
      <td>10</td>
      <td>61</td>
      <td>0.999945</td>
      <td>bruschetta</td>
      <td>lobster_roll_sandwich</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5574</th>
      <td>b'101_food_classes_10_percent/test/chocolate_m...</td>
      <td>22</td>
      <td>21</td>
      <td>0.999939</td>
      <td>chocolate_mousse</td>
      <td>chocolate_cake</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17855</th>
      <td>b'101_food_classes_10_percent/test/paella/2314...</td>
      <td>71</td>
      <td>65</td>
      <td>0.999931</td>
      <td>paella</td>
      <td>mussels</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23797</th>
      <td>b'101_food_classes_10_percent/test/sushi/16593...</td>
      <td>95</td>
      <td>86</td>
      <td>0.999904</td>
      <td>sushi</td>
      <td>sashimi</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18001</th>
      <td>b'101_food_classes_10_percent/test/pancakes/10...</td>
      <td>72</td>
      <td>67</td>
      <td>0.999904</td>
      <td>pancakes</td>
      <td>omelette</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11642</th>
      <td>b'101_food_classes_10_percent/test/garlic_brea...</td>
      <td>46</td>
      <td>10</td>
      <td>0.999877</td>
      <td>garlic_bread</td>
      <td>bruschetta</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10847</th>
      <td>b'101_food_classes_10_percent/test/fried_calam...</td>
      <td>43</td>
      <td>68</td>
      <td>0.999872</td>
      <td>fried_calamari</td>
      <td>onion_rings</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23631</th>
      <td>b'101_food_classes_10_percent/test/strawberry_...</td>
      <td>94</td>
      <td>83</td>
      <td>0.999858</td>
      <td>strawberry_shortcake</td>
      <td>red_velvet_cake</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>b'101_food_classes_10_percent/test/beef_tartar...</td>
      <td>4</td>
      <td>5</td>
      <td>0.999858</td>
      <td>beef_tartare</td>
      <td>beet_salad</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10854</th>
      <td>b'101_food_classes_10_percent/test/fried_calam...</td>
      <td>43</td>
      <td>68</td>
      <td>0.999854</td>
      <td>fried_calamari</td>
      <td>onion_rings</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23904</th>
      <td>b'101_food_classes_10_percent/test/sushi/33652...</td>
      <td>95</td>
      <td>86</td>
      <td>0.999823</td>
      <td>sushi</td>
      <td>sashimi</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7316</th>
      <td>b'101_food_classes_10_percent/test/cup_cakes/1...</td>
      <td>29</td>
      <td>83</td>
      <td>0.999816</td>
      <td>cup_cakes</td>
      <td>red_velvet_cake</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13144</th>
      <td>b'101_food_classes_10_percent/test/gyoza/31214...</td>
      <td>52</td>
      <td>92</td>
      <td>0.999799</td>
      <td>gyoza</td>
      <td>spring_rolls</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10880</th>
      <td>b'101_food_classes_10_percent/test/fried_calam...</td>
      <td>43</td>
      <td>68</td>
      <td>0.999778</td>
      <td>fried_calamari</td>
      <td>onion_rings</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Çok ilginç... sadece temel doğruluk sınıf adı (y_true_classname) ve tahmin sınıf adı sütununu (y_pred_classname) karşılaştırarak herhangi bir eğilim fark ettiniz mi?

Onları görselleştirirsek daha kolay olabilir.


```python
images_to_view = 9
start_index = 10 
plt.figure(figsize=(15, 10))
for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()): 
  plt.subplot(3, 3, i+1)
  img = load_and_prep_image(row[1], scale=True)
  _, _, _, _, pred_prob, y_true, y_pred, _ = row 
  plt.imshow(img)
  plt.title(f"actual: {y_true}, pred: {y_pred} \nprob: {pred_prob:.2f}")
  plt.axis(False)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_94_0.png)
    


Modelin en yanlış tahminlerini incelemek genellikle birkaç şeyi anlamaya yardımcı olabilir:

- Bazı etiketler yanlış olabilir - Modelimiz yeterince iyi olursa, aslında belirli sınıflarda çok iyi tahmin yapmayı öğrenebilir. Bu, modelin doğru etiketi öngördüğü bazı görüntülerin, temel doğruluk etiketinin yanlış olması durumunda yanlış olarak gösterilebileceği anlamına gelir. Durum buysa, modelimizi veri kümelerimizdeki etiketleri iyileştirmemize ve dolayısıyla gelecekteki modelleri potansiyel olarak daha iyi hale getirmemize yardımcı olması için sıklıkla kullanabiliriz. Etiketleri geliştirmeye yardımcı olmak için modeli kullanma sürecine genellikle [aktif öğrenme](https://blog.scaleway.com/active-learning-some-datapoints-are-more-equal-than-others/) denir.
- Daha fazla örnek toplanabilir mi? - Belirli bir sınıf için kötü tahmin edilen yinelenen bir model varsa, daha fazla modelleri geliştirmek için o belirli sınıftan farklı senaryolarda daha fazla örnek toplamak iyi bir fikir olabilir.

## Büyük Modeli Test Görüntülerinde ve Ayrıca Özel Gıda Görüntülerinde Test Edin

Şimdiye kadar test veri setinden modelimizin bazı tahminlerini görselleştirdik, ancak gerçek testin zamanı geldi: kendi özel yiyecek görüntülerimiz üzerinde tahminler yapmak için modelimizi kullanacağız
Bunun için kendi resimlerinizi Google Colab'a yüklemek veya bir klasöre koyarak not defterine yüklemek isteyebilirsiniz.

Benim durumumda, çeşitli yiyeceklerin altı ya da daha fazla görüntüsünden oluşan kendi küçük veri setimi hazırladım.

Bunları indirelim ve unzip edelim.


```python
!gdown --id 12hVGYlCfhagSGjPb80uR0BOjZOHN1SvA
unzip_data("custom_food_images.zip")
```

    Downloading...
    From: https://drive.google.com/uc?id=12hVGYlCfhagSGjPb80uR0BOjZOHN1SvA
    To: /content/custom_food_images.zip
    13.2MB [00:00, 207MB/s]


Harika, bunları yükleyebilir ve `load_and_prep_image()` işlevimizi kullanarak tensörlere dönüştürebiliriz ama önce bir görüntü dosyayolları listesine ihtiyacımız var.


```python
custom_food_images = ["custom_food_images/" + img_path for img_path in os.listdir("custom_food_images")]
custom_food_images
```




    ['custom_food_images/pizza-dad.jpeg',
     'custom_food_images/hamburger.jpeg',
     'custom_food_images/chicken_wings.jpeg',
     'custom_food_images/steak.jpeg',
     'custom_food_images/sushi.jpeg',
     'custom_food_images/ramen.jpeg']



Artık daha önce resimlerimize yüklemek için kullandığımıza benzer bir kod kullanabilir, eğitimli modelimizi kullanarak her biri için bir tahminde bulunabilir ve ardından resmi tahmin edilen sınıfla birlikte çizebiliriz.


```python
for img in custom_food_images:
  img = load_and_prep_image(img, scale=False) 
  pred_prob = model.predict(tf.expand_dims(img, axis=0))
  pred_class = class_names[pred_prob.argmax()] 
  plt.figure()
  plt.imshow(img/255.)
  plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
  plt.axis(False)
```


    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_0.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_1.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_2.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_3.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_4.png)
    



    
![png](TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_files/TensorFlow_ile_Transfer_Learning_%C3%96l%C3%A7eklendirme_%28Scaling_Up%29_101_5.png)
    


Bir makine öğrenimi modelinin önceden hazırlanmış bir test veri setinde çalıştığını görmek harika ama kendi verileriniz üzerinde çalıştığını görmek akıllara durgunluk veriyor.

Ve tahmin edin ne oldu... modelimiz eğitim görüntülerinin yalnızca %10'u ile bu inanılmaz sonuçları (temel değerden %10+ daha iyi) elde etti.
