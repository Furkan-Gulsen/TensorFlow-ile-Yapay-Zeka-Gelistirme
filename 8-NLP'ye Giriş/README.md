# NLP'ye Giriş

Doğal dil işlemenin (NLP) temel amacı, doğal dilden bilgi elde etmektir. Doğal dil geniş bir terimdir ancak aşağıdakilerden herhangi birini kapsadığını düşünebilirsiniz:

- Metin (bir e-postada, blog gönderisinde, kitapta, Tweette bulunanlar gibi)
- Konuşma (bir doktorla yaptığınız konuşma, telefonuna verdiğiniz sesli komutlar)

Metin ve konuşma şemsiyesi altında yapmak isteyebileceğiniz birçok farklı şey var. Bir e-posta uygulaması oluşturuyorsanız, spam olup olmadıklarını (sınıflandırma) görmek için gelen e-postaları taramak isteyebilirsiniz.

Müşteri geri bildirim şikayetlerini analiz etmeye çalışıyorsanız, bunların işletmenizin hangi bölümü için olduğunu keşfetmek isteyebilirsiniz.

> 🔑 Not: Bu tür verilerin her ikisine de genellikle diziler denir (bir cümle, bir sözcük dizisidir). Bu nedenle, NLP problemlerinde karşılaşacağınız yaygın bir terime **seq2seq** denir, başka bir deyişle, bir dizideki bilgiyi başka bir dizi oluşturmak için bulmaktır (örneğin, bir konuşma komutunu metin tabanlı adımlar dizisine dönüştürmek).

TensorFlow'da NLP ile pratik yapmak için daha önce kullandığımız adımları bu sefer metin verileriyle uygulayacağız:

```
Metin -> sayılara dönüştürün -> bir model oluşturun -> modeli kalıpları bulmak için eğitin -> kalıpları kullanın (tahminlerde bulunun)
```

## İçerik: 

- Bir metin veri kümesini indirme
- Metin verilerini görselleştirme
- Tokenization kullanarak metni sayılara dönüştürme
- Belirtilmiş metnimizi bir gömmeye dönüştürmek
- Bir metin veri kümesini modelleme
  - Temel ile başlama (TF-IDF)
  - Birkaç derin öğrenme metin modeli oluşturma
    - Yoğun, LSTM, GRU, Conv1D, Aktarım öğrenimi
- Her bir modelimizin performansını karşılaştırma
- Modellerimizi bir toplulukta birleştirmek
- Eğitilmiş bir modeli kaydetme ve yükleme
- En yanlış tahminleri bulunma

---

Eğitime başlamadan önce gerekli fonksiyonları oluşturalım.


```python
import zipfile
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
```


```python
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
```


```python
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
def unzip_data(filename):
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
```


```python
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
```

## Veri Kümesini İndirme

Bir metin veri kümesi indirerek başlayalım. [Real or Not](https://www.kaggle.com/c/nlp-getting-started/data)'u kullanacağız. Doğal afetler hakkında metin tabanlı Tweetler içeren Kaggle sitesinde bulunan veri seti.

**Gerçek Tweetler** aslında felaketlerle ilgilidir, örneğin:

```
Jetstar and Virgin forced to cancel Bali flights again because of ash from Mount Raung volcano
```

**Gerçek Olmayan Tweetler**, felaketlerle ilgili olmayan Tweetlerdir (her konuda olabilir), örneğin:

```
'Education is the most powerful weapon which you can use to change the world.' Nelson #Mandela #quote
```


```python
# Download data (same as from Kaggle)
!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
unzip_data("nlp_getting_started.zip")
```

    --2021-08-22 07:09:20--  https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.204.128, 64.233.188.128, 64.233.189.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.204.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 607343 (593K) [application/zip]
    Saving to: ‘nlp_getting_started.zip’
    
    nlp_getting_started 100%[===================>] 593.11K  --.-KB/s    in 0.007s  
    
    2021-08-22 07:09:20 (87.6 MB/s) - ‘nlp_getting_started.zip’ saved [607343/607343]
    
    

`nlp_getting_started.zip` dosyasında 3 farklı csv belgesi vardır: Bunlar: 

- **sample_submission.csv** 
Modelinizin tahminlerini içeren Kaggle yarışmasına göndereceğiniz dosyanın bir örneği.
- **train.csv**
Gerçek ve gerçek olmayan felaket Tweetlerinin eğitim örnekleri.
- **test.csv**
Gerçek ve gerçek olmayan felaket Tweet örneklerinin test edilmesi için örnekler.

<img src="https://boostlabs.com/wp-content/uploads/2019/09/10-types-of-data-visualization-1.jpg" />

## Bir Metin Veri Kümesini Görselleştirme

Çalışmak için yeni bir veri kümesi edindikten sonra, önce ne yapmalısınız? Keşfetmek mi? Kontrol etmek mi? Doğrulak mı? Hepsi doğru :)

Sloganı hatırlayın: görselleştirin, görselleştirin, görselleştirin.

Şu anda metin veri örneklerimiz `.csv` dosyaları biçimindedir. Onları görsel hale getirmenin kolay bir yolu için onları pandas DataFrame'e çevirelim.

> 📖 Okuma: Birçok farklı formatta metin veri setleriyle karşılaşabilirsiniz. CSV dosyalarının (üzerinde çalıştığımız şey) yanı sıra, muhtemelen `.txt` dosyaları ve `.json` dosyalarıyla da karşılaşacaksınız. Bu tür dosyalarla çalışmak için RealPython'un aşağıdaki iki makalesini okumanızı tavsiye ederim:

- [Python'da Dosyalar Nasıl Okunur ve Yazılır](https://realpython.com/read-write-files-python/)
- [Python'da JSON Verileriyle Çalışmak](https://realpython.com/python-json/)


```python
# .csv dosyalarını pandas DataFrame'lerine dönüştürün
import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



İndirdiğimiz eğitim verileri muhtemelen zaten karıştırılmıştır. Ama emin olmak için tekrar karıştıralım.


```python
train_df_shuffled = train_df.sample(frac=1, random_state=42) 
train_df_shuffled.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2644</th>
      <td>3796</td>
      <td>destruction</td>
      <td>NaN</td>
      <td>So you have a new weapon that can cause un-ima...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2227</th>
      <td>3185</td>
      <td>deluge</td>
      <td>NaN</td>
      <td>The f$&amp;amp;@ing things I do for #GISHWHES Just...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5448</th>
      <td>7769</td>
      <td>police</td>
      <td>UK</td>
      <td>DT @georgegalloway: RT @Galloway4Mayor: ÛÏThe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>191</td>
      <td>aftershock</td>
      <td>NaN</td>
      <td>Aftershock back to school kick off was great. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6845</th>
      <td>9810</td>
      <td>trauma</td>
      <td>Montgomery County, MD</td>
      <td>in response to trauma Children of Addicts deve...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Eğitim verilerinin nasıl bir `"target"` sütunu olduğuna dikkat edin.

`"target"` sütununun değerini tahmin etmek için eğitim veri kümesinin `"text"` sütununda kalıpları (örneğin farklı kelime kombinasyonları) bulmak için kod yazacağız. Test veri kümesinin bir `"target"` sütunu yok.

```
Inputs (text column) -> Machine Learning Algorithm -> Outputs (target column)
```


```python
# Test verilerinin bir hedefi yok (tahmin etmeye çalıştığımız şey bu)
test_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



Her hedeften kaç tane örneğimiz olduğunu kontrol edelim.


```python
# Her sınıftan kaç örnek var?
train_df.target.value_counts()
```




    0    4342
    1    3271
    Name: target, dtype: int64



İki sınıf değeri olduğundan, `binary_classification` problemiyle uğraşacağız gibi duruyor. Veri setimizi incelediğimizde dengeli bir dağılım görüyoruz. %60 olumsuz, %40 olumlu sınıf içeriyor.

- 1: gerçek bir felaket twet'i
- 0: gerçek olmayan bir felaket twet'i

Peki elimizde ki toplam örnek sayısı kaç?


```python
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")
```

    Total training samples: 7613
    Total test samples: 3263
    Total samples: 10876
    

Pekala, yeterli miktarda eğitim ve test verisine sahibiz gibi görünüyor. Görselleştirme zamanı, hadi rastgele metin örneklerini görselleştirmek için bazı kodlar yazalım.

> **🤔 Soru:** Rastgele örnekleri neden görselleştirelim? Örnekleri sırayla görselleştirebilirsiniz, ancak bu yalnızca belirli bir veri alt kümesini görmenize neden olabilir. Üzerinde çalıştığınız farklı veri türleri hakkında bir fikir edinmek için önemli miktarda (100+) rastgele örneği görselleştirmek daha iyidir. Makine öğreniminde rastgeleliğin gücünü asla hafife almayın.


```python
import random
random_index = random.randint(0, len(train_df)-5) 
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")
```

    Target: 0 (not real disaster)
    Text:
    @QuotesTTG Save the panicking for when you get to Helios. ;)
    
    ---
    
    Target: 0 (not real disaster)
    Text:
    Patience Jonathan On The Move To Hijack APC In BayelsaåÊState http://t.co/Vh8QtbyPZt
    
    ---
    
    Target: 0 (not real disaster)
    Text:
    Ali you flew planes and ran into burning buildings why are you making soup for that man child?! #BooRadleyVanCullen
    
    ---
    
    Target: 0 (not real disaster)
    Text:
    @ACOUSTICMALOLEY no he was blazing it
    
    ---
    
    Target: 1 (real disaster)
    Text:
    National Briefing | West: California: Spring Oil Spill Estimate Grows: Documents released on Wednesday d... http://t.co/hTxAi05y7B (NYT)
    
    ---
    
    

## Verileri Eğitim ve Doğrulama Kümelerine Ayırın

Test setinde etiket olmadığından ve eğitilmiş modellerimizi değerlendirmek için bir yola ihtiyacımız olduğundan, eğitim verilerinden bazılarını ayıracağız ve bir doğrulama seti oluşturacağız.

Modelimiz eğitildiğinde (Tweet örneklerindeki kalıpları denediğinde), yalnızca eğitim kümesindeki verileri görür ve doğrulama kümesini kullanarak görünmeyen veriler üzerinde nasıl performans gösterdiğini görebiliriz.

Pandas Series veri türlerinden bölmelerimizi daha sonra kullanım kolaylığı için string listelerine (metin için) ve ints listelerine (etiketler için) dönüştüreceğiz.

Eğitim veri setimizi bölmek ve bir doğrulama veri seti oluşturmak için Scikit-Learn'in `train_test_split()` yöntemini kullanacağız ve eğitim örneklerinin %10'unu doğrulama setine ayıracağız.


```python
from sklearn.model_selection import train_test_split

# Eğitim verilerini eğitim ve doğrulama kümelerine bölmek için train_test_split kullan
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, # örneklerin %10'unu doğrulama setine ayır
                                                                            random_state=42) # tekrarlanabilirlik için rastgele durum
```


```python
# Uzunlukları kontrol edin
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)
```




    (6851, 6851, 762, 762)




```python
# İlk 10 eğitim cümlesini ve etiketlerini görüntüleyin
train_sentences[:10], train_labels[:10]
```




    (array(['@mogacola @zamtriossu i screamed after hitting tweet',
            'Imagine getting flattened by Kurt Zouma',
            '@Gurmeetramrahim #MSGDoing111WelfareWorks Green S welfare force ke appx 65000 members har time disaster victim ki help ke liye tyar hai....',
            "@shakjn @C7 @Magnums im shaking in fear he's gonna hack the planet",
            'Somehow find you and I collide http://t.co/Ee8RpOahPk',
            '@EvaHanderek @MarleyKnysh great times until the bus driver held us hostage in the mall parking lot lmfao',
            'destroy the free fandom honestly',
            'Weapons stolen from National Guard Armory in New Albany still missing #Gunsense http://t.co/lKNU8902JE',
            '@wfaaweather Pete when will the heat wave pass? Is it really going to be mid month? Frisco Boy Scouts have a canoe trip in Okla.',
            'Patient-reported outcomes in long-term survivors of metastatic colorectal cancer - British Journal of Surgery http://t.co/5Yl4DC1Tqt'],
           dtype=object), array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1]))



## Metni Sayılara dönüştürme

Tweetler ve etiketler içeren bir eğitim setimiz ve bir doğrulama setimiz var. Etiketlerimiz sayısal (0 ve 1) biçimindedir, ancak Tweetlerimiz dize biçimindedir.

> 🤔 Soru: Metin verilerimizle bir makine öğrenmesi algoritması kullanabilmemiz için sizce ne yapmamız gerekiyor?

"Sayıya çevir" gibi bir cevap verdiyseniz, haklısınız. Bir makine öğrenimi algoritması, girdilerinin sayısal biçimde olmasını gerektirir.

NLP'de metni sayılara dönüştürmek için iki ana kavram vardır:

- **Tokenization**<br>
Kelimeden veya karakterden veya alt kelimeden sayısal bir değere düz bir eşleme. Üç ana tokenizasyon seviyesi vardır:
  1. Kelime düzeyinde simgeleştirmeyi "I love TensorFlow" cümlesiyle kullanmak, "I"nin 0, "love"  1 ve "TensorFlow"un 2 olmasına neden olabilir. Bu durumda, bir dizideki her sözcük tek bir simge olarak kabul edilir.
  2. A-Z harflerini 1-26 değerlerine dönüştürmek gibi karakter düzeyinde simgeleştirme. Bu durumda, bir dizideki her karakter tek bir simge olarak kabul edilir.
  3. Alt sözcük belirleme, sözcük düzeyinde ve karakter düzeyinde simgeleştirme arasındadır. Tek tek kelimeleri daha küçük parçalara ayırmayı ve ardından bu daha küçük parçaları sayılara dönüştürmeyi içerir. Örneğin, "my favorite food is pineapple pizza", "my, favor, rite, fo, oo, od, is, pin, ine, app, le, piz, za" olabilir. Bunu yaptıktan sonra, bu alt kelimeler daha sonra sayısal bir değere eşlenir. Bu durumda, her kelime birden fazla belirteç olarak kabul edilebilir.

- **Embedding**<br>
Embed, öğrenilebilen doğal dilin bir temsilidir. Temsil, bir özellik vektörü şeklinde gelir. Örneğin, "dance" kelimesi 5 boyutlu vektör [-0.8547, 0.4559, -0.3332, 0.9877, 0.1112] ile temsil edilebilir. Burada not etmek önemlidir, özellik vektörünün boyutu ayarlanabilir. Embed kullanmanın iki yolu vardır:
  1. Kendi embed işleminizi oluşturun - Metniniz sayılara dönüştürüldüğünde (embed için gereklidir), onları bir embed  katmanına (tf.keras.layers.Embedding gibi) koyabilirsiniz ve model eğitimi sırasında bir embed gösterimi öğrenilecektir.
  2. Önceden öğrenilmiş bir yerleştirmeyi yeniden kullanın - Çevrimiçi olarak önceden eğitilmiş birçok yerleştirme mevcuttur. Bu önceden eğitilmiş yerleştirmeler genellikle büyük metin kütlelerinde (tüm Wikipedia'da olduğu gibi) öğrenilmiştir ve bu nedenle doğal dilin iyi bir temel temsiline sahiptir. Modelinizi başlatmak ve kendi özel görevinize göre ince ayar yapmak için önceden eğitilmiş bir yerleştirme kullanabilirsiniz.

> Soru: Hangi düzeyde belirteç kullanmalıyım? Hangi embedi  seçmeliyim?

Sorununuza bağlı. Karakter düzeyinde tokenization/embed ve sözcük düzeyinde word-level-tokenization/embed deneyebilir ve hangisinin en iyi performansı gösterdiğini görebilirsiniz. Bunları istiflemeyi bile deneyebilirsiniz (örneğin, embed katmanlarınızın çıktılarını tf.keras.layers.concatenate kullanarak birleştirmek).

Önceden eğitilmiş sözcük yerleştirmeleri arıyorsanız, Word2vec yerleştirmeleri, GloVe yerleştirmeleri ve TensorFlow Hub'da bulunan seçeneklerin çoğu, başlamak için harika yerlerdir.

> 🔑 Not: Önceden eğitilmiş bir bilgisayarlı görü modelini aramaya benzer şekilde, probleminiz için kullanmak üzere önceden eğitilmiş sözcük yerleştirmelerini arayabilirsiniz. "TensorFlow'da önceden eğitilmiş kelime yerleştirmelerini kullan" gibi bir şey aramayı deneyin.

### Metin Vektörleştirme

İlk önce tokenizasyon (kelimelerimizi sayılarla eşleştirme) alıştırması yapacağız. Sözlerimizi simgeleştirmek için, yararlı önişleme katmanı `tf.keras.layers.experimental.preprocessing.TextVectorization` kullanacağız.

TextVectorization katmanı aşağıdaki parametreleri alır:
- **max_tokens**<br>
Kelime dağarcığınızdaki maksimum kelime sayısı (örneğin, metninizdeki 20000 veya benzersiz kelime sayısı), OOV (kelime dışı) belirteçleri için bir değer içerir.
- **standardize**<br>
Metni standartlaştırma yöntemi. Varsayılan, metni alçaltan ve tüm noktalama işaretlerini kaldıran "`lower_and_strip_punctuation`"dır.
- **split**<br>
Metin nasıl bölünür, varsayılan olarak boşluklara bölünen "split"tir.
- **ngrams**<br>
Belirteç başına kaç sözcük içerecek, örneğin, ngrams=2 belirteçleri 2'lik sürekli dizilere böler.
- **output_mode**<br>
Belirteçler nasıl çıkarılır, "int" (tamsayı eşleme), "binary" (tek-sıcak kodlama), "count" veya "tf-idf" olabilir. 
- **output_sequence_length**<br>
Çıktı için belirtilmiş dizinin uzunluğu. Örneğin, çıktı_dizi_uzunluk=150 ise, tüm belirteçli diziler 150 belirteç uzunluğunda olacaktır.
- **pad_to_max_tokens**<br>
True (varsayılan) ise, sözlükteki benzersiz jeton sayısı max_tokens'den az olsa bile çıktı özelliği ekseni max_tokens olarak doldurulur.


```python
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

text_vectorizer = TextVectorization(max_tokens=None,
                                    standardize="lower_and_strip_punctuation",
                                    split="whitespace", 
                                    ngrams=None, 
                                    output_mode="int",
                                    output_sequence_length=None,
                                    pad_to_max_tokens=False)
```

Bir TextVectorization nesnesini varsayılan ayarlarla başlattık, ancak bunu kendi kullanım durumumuz için biraz özelleştirelim. Özellikle `max_tokens` ve `output_sequence_length` için değerler belirleyelim.

`max_tokens` (kelimelerdeki kelime sayısı) için 10.000'in katları (10.000, 20.000, 30.000) veya metninizdeki tam benzersiz kelime sayısı (ör. 32.179) ortak değerlerdir. Kullanım durumumuz için 10.000 kullanacağız.

Ve `output_sequence_length` için, eğitim setindeki Tweet başına ortalama jeton sayısını kullanacağız. Ama önce onu bulmamız gerekecek.


```python
# Eğitim Tweetlerinde ortalama jeton (kelime) sayısını bulma
round(sum([len(i.split()) for i in train_sentences])/len(train_sentences))
```




    15



Şimdi özel parametrelerimizi kullanarak başka bir TextVectorization nesnesi oluşturalım.


```python
# Metin vektörleştirme değişkenlerini ayarlayın
max_vocab_length = 10000 # kelime dağarcığımızda bulunması gereken maksimum kelime sayısı
max_length = 15 # dizilerimiz maksimum uzunluk olacaktır (ör. modelimiz bir Tweetten kaç kelime görüyor?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
```

Güzel! `TextVectorization` örneğimizi text_vectorizer verilerimizle eşleştirmek için, eğitim metnimizi iletirken `adapt()` yöntemini çağırabiliriz.


```python
# Metin vektörleştiriciyi eğitim metnine fit etme
text_vectorizer.adapt(train_sentences)
```

Eğitim verileri eşlendi! Text_vectorizer'ımızı özel bir cümle üzerinde deneyelim (eğitim verilerinde görebileceğinize benzer bir cümle).


```python
# Örnek cümle oluştur ve onu belirt
sample_sentence = "There's a flood in my street!"
text_vectorizer([sample_sentence])
```




    <tf.Tensor: shape=(1, 15), dtype=int64, numpy=
    array([[264,   3, 232,   4,  13, 698,   0,   0,   0,   0,   0,   0,   0,
              0,   0]])>



Harika, görünüşe göre metnimizi sayılara dönüştürmenin bir yolu var (bu durumda, kelime düzeyinde simgeleştirme). Döndürülen tensörün sonundaki 0'lara dikkat edin, bunun nedeni output_sequence_length=15 olarak ayarlamış olmamızdır, yani text_vectorizer'a ilettiğimiz dizinin boyutu ne olursa olsun, her zaman 15 uzunluğunda bir dizi döndürür.

Birkaç rastgele cümle üzerinde text_vectorizer'ımızı denemeye ne dersiniz?


```python
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nVectorized version:")
text_vectorizer([random_sentence])
```

    Original text:
    Quirk Injury Law's News is out! http://t.co/HxVIhDuShP Stories via @dantmatrafajlo      
    
    Vectorized version:
    




    <tf.Tensor: shape=(1, 15), dtype=int64, numpy=
    array([[9416,  345, 2068,   58,    9,   36,    1, 1172,   49,    1,    0,
               0,    0,    0,    0]])>



İyi görünüyor! Son olarak, `get_vocabulary()` yöntemini kullanarak sözlüğümüzdeki benzersiz belirteçleri kontrol edebiliriz.


```python
# Kelime dağarcığındaki benzersiz kelimeleri alın
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5]
bottom_5_words = words_in_vocab[-5:] 
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}") 
print(f"Bottom 5 least common words: {bottom_5_words}")
```

    Number of words in vocab: 10000
    Top 5 most common words: ['', '[UNK]', 'the', 'a', 'in']
    Bottom 5 least common words: ['pages', 'paeds', 'pads', 'padres', 'paddytomlinson1']
    

Gömme Katmanı Kullanarak Gömme Oluşturma
Metnimizi sayılarla eşleştirmenin bir yolu var. Bir adım daha ileri gidip bu sayıları bir gömme haline getirmeye ne dersiniz?

Bir gömmenin güçlü yanı, eğitim sırasında öğrenilebilmesidir. Bu, yalnızca statik olmaktan ziyade (örneğin 1 = I, 2 = love, 3 = TensorFlow), bir kelimenin sayısal gösteriminin, bir model veri örneklerinden geçerken geliştirilebileceği anlamına gelir.

`tf.keras.layers.Embedding` katmanını kullanarak bir kelimenin gömülmesinin nasıl göründüğünü görebiliriz.

Burada ilgilendiğimiz ana parametreler şunlardır:

- **input_dim** <br>
Sözlüğün boyutu 
- **output_dim**<br>
Çıktı gömme vektörünün boyutu, örneğin 100 değeri, her kelime için 100 boyutunda bir özellik vektörü verir.
- **embeddings_initializer**<br>
Gömme matrisi nasıl başlatılır, varsayılan değer, tek tip dağılımla gömme matrisini rastgele başlatan "tek biçimli"dir. Bu, önceden öğrenilmiş yerleştirmeleri kullanmak için değiştirilebilir.
- **input_length**<br> Gömme katmanına geçirilen dizilerin uzunluğu.

Bunları bilerek bir gömme katmanı yapalım.


```python
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128, 
                             embeddings_initializer="uniform",
                             input_length=max_length) 

embedding
```




    <keras.layers.embeddings.Embedding at 0x7f5bf062d050>



Mükemmel, TensoFlow katmanının nasıl gömüldüğünü fark ettiniz mi? Bu önemlidir çünkü onu bir modelin parçası olarak kullanabiliriz, yani parametreleri (kelime temsilleri) model öğrendikçe güncellenebilir ve geliştirilebilir.

Örnek bir cümle üzerinde deneyelim mi?


```python
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed
```

    Original text:
    Anyone wanna come over and watch Twister with me? #toosoon :-)      
    
    Embedded version:
    




    <tf.Tensor: shape=(1, 15, 128), dtype=float32, numpy=
    array([[[-2.67661568e-02,  3.09980996e-02,  9.19399410e-03, ...,
             -1.61751285e-02,  2.21572071e-03, -7.00148195e-03],
            [-4.41038385e-02, -4.84013557e-03,  2.72500776e-02, ...,
             -1.73950568e-02, -2.18516830e-02, -9.85272974e-03],
            [-1.61503777e-02, -3.82886529e-02,  2.60415785e-02, ...,
             -3.55404019e-02,  8.02986324e-05, -7.18279928e-03],
            ...,
            [-3.36676128e-02, -6.04265928e-03, -5.23805618e-04, ...,
             -4.41053882e-02,  1.10260025e-02,  1.55389644e-02],
            [-3.36676128e-02, -6.04265928e-03, -5.23805618e-04, ...,
             -4.41053882e-02,  1.10260025e-02,  1.55389644e-02],
            [-3.36676128e-02, -6.04265928e-03, -5.23805618e-04, ...,
             -4.41053882e-02,  1.10260025e-02,  1.55389644e-02]]],
          dtype=float32)>



Cümledeki her belirteç, 128 uzunlukta bir özellik vektörüne dönüştürülür.


```python
sample_embed[0][0]
```




    <tf.Tensor: shape=(128,), dtype=float32, numpy=
    array([-2.6766157e-02,  3.0998100e-02,  9.1939941e-03,  1.1298049e-02,
            1.9281853e-02,  1.2828600e-02,  2.7620319e-02, -4.6756484e-02,
           -4.7940601e-02, -9.4258562e-03,  8.8882931e-03,  9.9281296e-03,
           -1.1045527e-02,  2.3107305e-03,  1.1442937e-02, -2.6214374e-02,
            1.9464921e-02,  9.2443340e-03, -1.1554696e-02,  1.6496528e-02,
            1.1290133e-02,  1.7679345e-02,  4.1412916e-02,  8.3859339e-03,
           -2.3221433e-02, -1.5267409e-02,  6.1252825e-03, -2.3301685e-02,
            8.3223693e-03,  3.4590412e-02, -1.8654786e-02,  2.2479955e-02,
            2.3416769e-02, -1.4507163e-02, -3.3194818e-02,  1.5210751e-02,
            1.2733910e-02, -4.2662036e-02, -3.1142402e-02,  2.9466037e-02,
           -1.3123263e-02,  2.6843850e-02,  2.1498416e-02,  4.5063887e-02,
           -3.6920715e-02,  1.0851800e-02,  8.6697713e-03,  4.7077391e-02,
            3.5928216e-02, -4.7313895e-02,  1.0041595e-03,  1.1565756e-02,
           -6.1505325e-03,  3.5577092e-02,  3.0785192e-02, -1.0508526e-02,
            2.8408121e-02,  2.3087058e-02,  4.5386180e-03,  5.6251884e-03,
            1.3862085e-02,  2.7922798e-02,  3.3377770e-02,  4.2726230e-02,
           -9.0640187e-03,  5.7292096e-03, -1.4854670e-02,  1.6860548e-02,
            2.0201530e-02, -4.1249715e-02, -2.0793175e-02,  3.2375220e-02,
           -3.9839089e-02, -4.9911141e-03, -7.8412518e-03,  2.9475693e-02,
           -3.2879878e-02,  2.2757497e-02,  9.1821551e-03, -3.9630160e-03,
            1.7598383e-03,  6.4185746e-03,  3.9488077e-07,  3.5669398e-02,
            2.0191576e-02,  4.4047508e-02,  2.1852169e-02, -4.9949538e-02,
            2.9918279e-02,  1.1461068e-02, -4.4214189e-02, -4.5909584e-02,
           -1.4579408e-03, -3.6948562e-02, -3.2227956e-02,  2.8313946e-02,
           -9.3356520e-04,  7.1534142e-03,  1.9067053e-02, -2.5455356e-02,
           -1.3002217e-02,  3.1422488e-03, -4.7948360e-02, -2.8451920e-02,
           -4.4162154e-02, -6.4193085e-04,  2.4962079e-02, -3.2402515e-02,
            6.5453053e-03, -6.9365017e-03,  1.5323464e-02,  1.9762184e-02,
            3.9757859e-02,  4.8979964e-02,  4.1511167e-02, -2.3521185e-03,
           -4.2080723e-02, -1.8117238e-02,  4.2290125e-02, -3.8343213e-02,
            2.6690889e-02,  1.0297049e-02,  3.5892416e-02, -2.7472233e-02,
           -2.4868632e-02, -1.6175129e-02,  2.2157207e-03, -7.0014820e-03],
          dtype=float32)>



🔑 Not: Önceki iki kavram (tokenization ve embedding) birçok NLP görevinin temelidir. Bu nedenle, herhangi bir şeyden emin değilseniz, anlayışınıza daha fazla yardımcı olmak için kendi deneylerinizi araştırdığınızdan ve yürüttüğünüzden emin olun.

## Bir Metin Veri Kümesini Modelleme

Girdilerinizi ve çıktılarınızı hazırladıktan sonra, aradaki boşluğu kapatmak için hangi makine öğrenimi modelinin oluşturulacağını bulmak meselesidir.

Artık metin verilerimizi sayılara dönüştürmenin bir yolu olduğuna göre, onu modellemek için makine öğrenimi modelleri oluşturmaya başlayabiliriz.

Bol bol pratik yapmak için, her biri kendi deneyi olan bir dizi farklı model oluşturacağız. Daha sonra her modelin sonuçlarını karşılaştıracağız ve hangisinin en iyi performansı gösterdiğini göreceğiz.

Daha spesifik olarak, aşağıdakileri inşa edeceğiz:

- Model 0: Naive Bayes (temel)
- Model 1: İleri beslemeli sinir ağı (yoğun model)
- Model 2: LSTM modeli
- Model 3: GRU modeli
- Model 4: Çift Yönlü-LSTM modeli
- Model 5: 1B Evrişimli Sinir Ağı
- Model 6: TensorFlow Hub Önceden Eğitilmiş Özellik Çıkarıcı (feature extraction)
- Model 7: Eğitim verilerinin %10 ile model 6'nın aynısı 

Model 0, diğer daha derin modellerin birbirini yenmesini bekleyeceğimiz bir temel elde etmek için en basit olanıdır.

Her deney aşağıdaki adımlardan geçecektir:

- Modeli oluşturun
- Modeli eğit
- Modelle tahminler yapın
- Daha sonra karşılaştırma için tahmin değerlendirme metriklerini takip edin

### Model 0: Temel oluşturma

Tüm makine öğrenimi modelleme deneylerinde olduğu gibi, bir temel model oluşturmak önemlidir, böylece gelecekteki deneyler için üzerine inşa edebileceğiniz bir kıyaslama elde edersiniz.

Temel çizgimizi oluşturmak için, kelimelerimizi sayılara dönüştürmek için TF-IDF (terim frekansı-ters belge frekansı) formülünü kullanarak bir Scikit-Learn Pipeline oluşturacağız ve ardından bunları Multinomial Naive Bayes algoritması ile modelleyeceğiz. Bu, Scikit-Learn makine öğrenimi haritasına başvurularak seçildi.

📖 TD-IDF algoritması hakkında okunması gereken bir makale: https://www.onely.com/blog/what-is-tf-idf/


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model_0 = Pipeline([
                    ("tfidf", TfidfVectorizer()),# tfidf kullanarak kelimeleri sayılara dönüştürün
                    ("clf", MultinomialNB()) # metni modelle
])

model_0.fit(train_sentences, train_labels)
```




    Pipeline(memory=None,
             steps=[('tfidf',
                     TfidfVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.float64'>,
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=None,
                                     min_df=1, ngram_range=(1, 1), norm='l2',
                                     preprocessor=None, smooth_idf=True,
                                     stop_words=None, strip_accents=None,
                                     sublinear_tf=False,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, use_idf=True,
                                     vocabulary=None)),
                    ('clf',
                     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],
             verbose=False)



`Multinomial Naive Bayes` gibi sığ bir model kullanmanın yararı, eğitimin çok hızlı olmasıdır. Modelimizi değerlendirelim ve temel metriğimizi bulalım.


```python
baseline_score = model_0.score(val_sentences, val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")
```

    Our baseline model achieves an accuracy of: 79.27%
    

Temel modelimiz ile bazı tahminler yapmaya ne dersiniz?


```python
baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]
```




    array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])



### Model Deneylerimiz İçin Bir Değerlendirme Fonksiyonu Oluşturma

Bunları olduğu gibi değerlendirebiliriz, ancak ileride birkaç modeli aynı şekilde değerlendireceğimiz için, bir dizi tahmin ve kesinlik etiketi alan ve aşağıdakileri hesaplayan bir yardımcı fonksiyon oluşturalım:

- Accuracy
- Precision
- Recall
- F1-score

🔑 Not: Bir sınıflandırma sorunuyla uğraştığımız için yukarıdaki metrikler en uygun olanlardır. Bir regresyon problemi ile çalışıyor olsaydık, MAE (ortalama mutlak hata) gibi diğer metrikler daha iyi bir seçim olurdu.


```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100

  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
baseline_results
```




    {'accuracy': 79.26509186351706,
     'f1': 0.7862189758049549,
     'precision': 0.8111390004213173,
     'recall': 0.7926509186351706}



### Model 1: Basit Bir Yoğun (Dense) Model

İnşa edeceğimiz ilk derin model, tek katmanlı yoğun bir modeldir. Aslında, zar zor tek bir katmana sahip olacak.

Metnimizi ve etiketlerimizi girdi olarak alacak, metni simgeleştirecek, bir gömme oluşturacak, gömmenin ortalamasını bulacak (Küresel Ortalama Havuzlamayı kullanarak) ve ardından ortalamayı bir çıktı birimi ve bir sigmoid etkinleştirme işleviyle tam bağlantılı bir katmandan geçirecektir.

Önceki cümle kulağa ağız dolusu gibi geliyorsa, onu kodladığımızda mantıklı olacaktır (şüpheniz varsa, kodlayın).


```python
# TensorBoard günlüklerini kaydetmek için dizin oluşturun
SAVE_DIR = "model_logs"
```

Şimdi kullanıma hazır bir TensorBoard geri çağırma işlevimiz var, hadi ilk derin modelimizi oluşturalım.


```python
from tensorflow.keras import layers

# girdiler 1 boyutlu dizelerdir
inputs = layers.Input(shape=(1,), dtype="string")

# giriş metnini sayılara çevirin
x = text_vectorizer(inputs) 

# numaralandırılmış sayıların bir gömülmesini oluşturun
x = embedding(x) 

# gömmenin boyutunu azaltın (modeli bu katman olmadan çalıştırmayı deneyin ve ne olduğunu görün)
x = layers.GlobalAveragePooling1D()(x)

# çıktı katmanını oluşturun, ikili çıktılar isteyin, bu nedenle sigmoid aktivasyonunu kullanın
outputs = layers.Dense(1, activation="sigmoid")(x) 

# modeli oluşturun
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense") 
```

İyi görünüyor. Modelimiz girdi olarak 1 boyutlu bir dize alır (bizim durumumuzda bir Tweet), ardından text_vectorizer kullanarak dizeyi belirtir ve gömmeyi kullanarak bir gömme oluşturur.

Daha sonra (isteğe bağlı olarak) çıktı katmanına ilettiğimiz tensörün boyutsallığını azaltmak için gömme katmanının çıktılarını havuzlarız.

Son olarak, havuzlama katmanının çıktısını sigmoid aktivasyonu ile yoğun bir katmana geçiriyoruz (sorunumuz ikili sınıflandırma olduğu için sigmoid kullanıyoruz).

Modelimizi verilerle fit etmeden önce onu derlememiz gerekiyor. İkili sınıflandırma ile çalıştığımız için, kayıp fonksiyonumuz ve Adam optimize edici olarak "`binary_crossentropy`" kullanacağız.


```python
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# model derlendi. Özetine göz atalım
model_1.summary()
```

    Model: "model_1_dense"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 15)                0         
    _________________________________________________________________
    embedding (Embedding)        (None, 15, 128)           1280000   
    _________________________________________________________________
    global_average_pooling1d (Gl (None, 128)               0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 129       
    =================================================================
    Total params: 1,280,129
    Trainable params: 1,280,129
    Non-trainable params: 0
    _________________________________________________________________
    

Eğitilebilir parametrelerin çoğu, gömme katmanında bulunur. 10.000 (input_dim=10000) boyutunda bir sözcük dağarcığı için 128 boyutunda (output_dim=128) bir yerleştirme oluşturduğumuzu, dolayısıyla 1.280.000 eğitilebilir parametre oluşturduğumuzu hatırlayın.

Pekala, modelimiz derlendi, 5 epoch kullanarak eğitim verilerimize fit edelim. Modelimizin eğitim ölçümlerinin log'lara kaydedildiğinden emin olmak için TensorBoard geri çağırma işlevimizi de ileteceğiz.


```python
model_1_history = model_1.fit(
    train_sentences, 
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                           experiment_name="simple_dense_model")])
```

    Saving TensorBoard log files to: model_logs/simple_dense_model/20210822-070929
    Epoch 1/5
    215/215 [==============================] - 5s 9ms/step - loss: 0.6113 - accuracy: 0.6904 - val_loss: 0.5373 - val_accuracy: 0.7559
    Epoch 2/5
    215/215 [==============================] - 1s 6ms/step - loss: 0.4417 - accuracy: 0.8187 - val_loss: 0.4700 - val_accuracy: 0.7887
    Epoch 3/5
    215/215 [==============================] - 1s 6ms/step - loss: 0.3467 - accuracy: 0.8619 - val_loss: 0.4552 - val_accuracy: 0.7979
    Epoch 4/5
    215/215 [==============================] - 1s 6ms/step - loss: 0.2842 - accuracy: 0.8914 - val_loss: 0.4623 - val_accuracy: 0.7913
    Epoch 5/5
    215/215 [==============================] - 1s 6ms/step - loss: 0.2373 - accuracy: 0.9113 - val_loss: 0.4754 - val_accuracy: 0.7848
    

Güzel! Bu kadar basit bir model kullandığımız için her epoch çok hızlı işliyor. Modelimizin doğrulama setindeki performansını kontrol edelim.


```python
model_1.evaluate(val_sentences, val_labels)
```

    24/24 [==============================] - 0s 4ms/step - loss: 0.4754 - accuracy: 0.7848
    




    [0.4753652513027191, 0.7847769260406494]



Ve modelimizin eğitim loglarını TensorBoard ile takip ettiğimize göre, onları görselleştirmeye ne dersiniz? Bunu, TensorBoard log dosyalarımızı (model_logs dizininde bulunur) TensorBoard.dev'e yükleyerek yapabiliriz.

🔑 Not: TensorBoard.dev'e yüklediğiniz her şeyin herkese açık hale geleceğini unutmayın. Paylaşmak istemediğiniz antrenman kayıtları varsa yüklemeyin.


```python
# !tensorboard dev upload --logdir ./model_logs \
#   --name "First deep model on text data" \
#   --description "Trying a dense model with an embedding layer" \
#   --one_shot # exits the uploader when upload has finished
```

<img src="https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/08-tensorboard-dense-model-training-curves.png" />

Güzel! Bunlar bazı renkli eğitim eğrileri. Modelin fazla mı yoksa yetersiz mi olduğunu söyler misiniz? İlk derin modelimizi oluşturduk ve eğittik, bir sonraki adım onunla bazı tahminler yapmak.


```python
# tahminler yapma
model_1_pred_probs = model_1.predict(val_sentences)
model_1_pred_probs[:10]
```




    array([[0.39653158],
           [0.7810238 ],
           [0.9976156 ],
           [0.10114352],
           [0.10857761],
           [0.9328917 ],
           [0.9154491 ],
           [0.9949896 ],
           [0.9680576 ],
           [0.20620456]], dtype=float32)



Son katmanımız bir sigmoid aktivasyon fonksiyonu kullandığından, tahminlerimizi olasılıklar şeklinde geri alıyoruz.

Bunları tahmin sınıflarına dönüştürmek için `tf.round()` kullanacağız, yani 0,5'in altındaki tahmin olasılıkları 0'a ve 0,5'in üzerindekiler 1'e yuvarlanacaktır.

🔑 Not: Pratikte, bir sigmoid tahmin olasılığının çıktı eşiğinin mutlaka 0,5 olması gerekmez. Örneğin, test yoluyla, seçtiğiniz değerlendirme metrikleri için 0,25'lik bir kesmenin daha iyi olduğunu görebilirsiniz.


```python
# Tahmin olasılıklarını tek boyutlu float tensöre dönüştürün
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:20]
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1.], dtype=float32)>



Şimdi modelimizin tahminlerini sınıflar şeklinde elde ettik, onları temel doğruluk doğrulama etiketleriyle karşılaştırmak için `calculate_results()` işlevimizi kullanabiliriz.


```python
model_1_results = calculate_results(y_true=val_labels, y_pred=model_1_preds)
model_1_results
```




    {'accuracy': 78.4776902887139,
     'f1': 0.7818959205825942,
     'precision': 0.789165199286798,
     'recall': 0.7847769028871391}



İlk derin modelimizi temel modelimizle karşılaştırmaya ne dersiniz?


```python
import numpy as np
np.array(list(model_1_results.values())) > np.array(list(baseline_results.values()))
```




    array([False, False, False, False])



Bu tür bir karşılaştırmayı (yeni modele kıyasla temel) birkaç kez yapacağımız için, bize yardımcı olacak bir fonksiyon oluşturalım.


```python
def compare_baseline_to_new_results(baseline_results, new_model_results):
  for key, value in baseline_results.items():
    print(f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, Difference: {new_model_results[key]-value:.2f}")

compare_baseline_to_new_results(baseline_results=baseline_results, 
                                new_model_results=model_1_results)
```

    Baseline accuracy: 79.27, New accuracy: 78.48, Difference: -0.79
    Baseline precision: 0.81, New precision: 0.79, Difference: -0.02
    Baseline recall: 0.79, New recall: 0.78, Difference: -0.01
    Baseline f1: 0.79, New f1: 0.78, Difference: -0.00
    

# Tekrarlayan Sinir Ağları (RNN'ler)

Bir sonraki modelleme deneylerimiz için, Tekrarlayan Sinir Ağı (RNN) adı verilen özel bir tür sinir ağı kullanacağız.

Bir RNN'nin önermesi basittir: gelecekte size yardımcı olması için geçmişten gelen bilgileri kullanın (tekrarlayan terimi buradan gelir). Başka bir deyişle, bir girdi (X) alın ve önceki tüm girdilere dayanarak bir çıktı (y) hesaplayın.

Bu kavram, özellikle doğal dil metinlerinin (Tweet'lerimiz gibi) pasajları gibi dizilerle uğraşırken yararlıdır.

Örneğin, bu cümleyi okuduğunuzda, mevcut köpek kelimesinin anlamını deşifre ederken önceki kelimeleri bağlam içine alırsınız. Geçerli bir kelime olan "köpek" kelimesini sonuna koydum ama cümlenin geri kalanı bağlamında bir anlam ifade etmiyor.

Bir RNN bir metin dizisine (zaten sayısal biçimde) baktığında, öğrendiği modeller dizinin sırasına göre sürekli olarak güncellenir.

Basit bir örnek için iki cümle alın:

- Geçen hafta büyük deprem oldu, değil mi?
- Geçen hafta büyük bir deprem olmadı.

Her ikisi de tamamen aynı kelimeleri içerir, ancak farklı anlamlara sahiptir. Sözcüklerin sırası anlamı belirler (noktalama işaretlerinin de anlamı dikte ettiği tartışılabilir, ancak basitlik adına, kelimelere odaklanalım).

Tekrarlayan sinir ağları, bir dizi dizi tabanlı problem için kullanılabilir:

- **Bire bir:**<br> 
bir girdi, bir çıktı, görüntü sınıflandırması gibi.
- **Birden çoğa:**<br> 
bir giriş, resim yazısı gibi birçok çıkış (resim girişi, resim yazısı çıkışı olarak bir metin dizisi).
- **Çoktan bire:**<br>
birçok girdi, metin sınıflandırması gibi bir çıktı (bir Tweet'i gerçek hata veya gerçek hata değil olarak sınıflandırma).
- **Çoktan çoğa:**<br>
birçok girdi, makine çevirisi (İngilizceden İspanyolcaya çevirme) veya konuşmayı metne (giriş olarak ses dalgası, çıktı olarak metin) gibi birçok çıktı.

Vahşi doğada RNN'lerle karşılaştığınızda, büyük olasılıkla aşağıdakilerin varyantlarıyla karşılaşacaksınız:

- Uzun kısa süreli hafıza hücreleri (LSTM'ler).
- Kapılı yinelenen birimler (GRU'lar).
- Çift yönlü RNN'ler (bir dizi boyunca ileri ve geri, soldan sağa ve sağdan sola geçer).

Bunların her birinin ayrıntılarına girmek bu defterin kapsamı dışındadır (bunun yerine onları kullanmaya odaklanacağız), şimdilik bilmeniz gereken en önemli şey, dizileri modellemede çok etkili olduklarını kanıtladıklarıdır.

Yazmak üzere olduğumuz kodun perde arkasında neler olduğunu daha iyi anlamak için aşağıdaki kaynakları tavsiye ederim:
> * [MIT Deep Learning Lecture on Recurrent Neural Networks](https://youtu.be/SEnXr6v2ifU) 
> * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 
> * [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) 


## Model 2: LSTM

RNN'lerin ne olduğu ve ne işe yaradığıyla ilgili tüm bu konuşmalardan sonra, eminim siz de bir tane oluşturmaya heveslisinizdir. LSTM destekli bir RNN ile başlayacağız.

TensorFlow'da LSTM hücresinin (LSTM hücresi ve LSTM katmanı genellikle birbirinin yerine kullanılır) gücünden yararlanmak için [`tensorflow.keras.layers.LSTM()`](https://www.tensorflow.org/) kullanacağız. api_docs/python/tf/keras/layers/LSTM).

Modelimiz `model_1` ile çok benzer bir yapı alacak:

```
Input (metin) -> Tokenization -> Embedding -> Layers -> Output (etiket olasılığı)
```

Temel fark, gömme ve çıktımız arasına bir LSTM katmanı ekleyeceğimiz olacaktır.


```python
# LSTM oluşturma
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
print(x.shape)

x = layers.LSTM(64)(x)
print(x.shape)

outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_LSTM")
```

    (None, 15, 128)
    (None, 64)
    

> 🔑 **Not:** [TensorFlow LSTM katmanı](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) için belgeleri okurken, çok sayıda parametre bulacaksınız . Bunların çoğu, mümkün olduğunca hızlı hesaplanmalarını sağlamak için ayarlanmıştır. Ayarlamak isteyeceğiniz başlıca olanlar "units" (gizli birimlerin sayısı) ve "return_sequences"dir (LSTM veya diğer tekrarlayan katmanları istiflerken bunu "True" olarak ayarlayın).

Şimdi LSTM modelimizi oluşturduk, hadi onu `"binary_crossentropy"` kaybı ve Adam optimizer kullanarak derleyelim.


```python
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Modelimizi fit etmeden önce bir özet geçelim:
model_2.summary()
```

    Model: "model_2_LSTM"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 15)                0         
    _________________________________________________________________
    embedding (Embedding)        (None, 15, 128)           1280000   
    _________________________________________________________________
    lstm (LSTM)                  (None, 64)                49408     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 1,329,473
    Trainable params: 1,329,473
    Non-trainable params: 0
    _________________________________________________________________
    

İyi görünüyor! LSTM katmanımızda "model_1"den çok daha fazla eğitilebilir parametre fark edeceksiniz.

Bu sayının nereden geldiğini bilmek istiyorsanız, bir LSTM hücresindeki parametre sayısını hesaplamak için yukarıdaki kaynakları ve aşağıdakileri incelemenizi öneririm:
* [LSTM hücresindeki parametre sayısını hesaplamak için Stack Overflow yanıtı](https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network ) yazan Marcin Możejko
* [LSTM birimindeki ve katmanındaki parametre sayısı hesaplanıyor](https://medium.com/@priyadarshi.cse/calcizing-number-of-parameters-in-a-lstm-unit-layer-7e491978e1e4) Shridhar Priyadarshi

Şimdi ilk RNN modelimiz derlendi, onu eğitim verilerimizle fit edelim, doğrulama verileri üzerinde doğrulayalım ve TensorBoard geri çağrımızı kullanarak eğitim parametrelerini takip edelim.


```python
model_2_history = model_2.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(SAVE_DIR, "LSTM")])
```

    Saving TensorBoard log files to: model_logs/LSTM/20210822-070941
    Epoch 1/5
    215/215 [==============================] - 7s 14ms/step - loss: 0.2219 - accuracy: 0.9237 - val_loss: 0.5674 - val_accuracy: 0.7795
    Epoch 2/5
    215/215 [==============================] - 2s 10ms/step - loss: 0.1562 - accuracy: 0.9429 - val_loss: 0.5965 - val_accuracy: 0.7730
    Epoch 3/5
    215/215 [==============================] - 2s 10ms/step - loss: 0.1307 - accuracy: 0.9510 - val_loss: 0.7095 - val_accuracy: 0.7769
    Epoch 4/5
    215/215 [==============================] - 2s 10ms/step - loss: 0.1053 - accuracy: 0.9606 - val_loss: 0.7949 - val_accuracy: 0.7874
    Epoch 5/5
    215/215 [==============================] - 2s 10ms/step - loss: 0.0886 - accuracy: 0.9673 - val_loss: 0.8062 - val_accuracy: 0.7808
    

Güzel! LSTM hücrelerini kullanan ilk eğitimli RNN modelimize sahibiz. Onunla bazı tahminler yapalım. Son katmandaki sigmoid aktivasyon fonksiyonu nedeniyle daha önce olduğu gibi aynı şey olacak, modelimizde `predict()` yöntemini çağırdığımızda sınıflardan ziyade tahmin olasılıklarını döndürecek.


```python
model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs.shape, model_2_pred_probs[:10]
```




    ((762, 1), array([[0.1147354 ],
            [0.94750935],
            [0.9996575 ],
            [0.11396935],
            [0.00261294],
            [0.99850935],
            [0.95742464],
            [0.9997907 ],
            [0.9996567 ],
            [0.39139414]], dtype=float32))



Bu tahmin olasılıklarını en yakın tam sayıya yuvarlayarak tahmin sınıflarına dönüştürebiliriz (varsayılan olarak 0,5'in altındaki tahmin olasılıkları 0'a, 0,5'in üzerindekiler ise 1'e gidecektir).


```python
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 1., 1., 0., 0., 1., 1., 1., 1., 0.], dtype=float32)>



Güzel, şimdi LSTM modelimizi değerlendirmek için `caculate_results()` işlevimizi ve bunu temel modelimizle karşılaştırmak için `Compare_baseline_to_new_results()` işlevimizi kullanalım.


```python
model_2_results = calculate_results(y_true=val_labels,
                                    y_pred=model_2_preds)
model_2_results
```




    {'accuracy': 78.08398950131233,
     'f1': 0.7794817733933848,
     'precision': 0.781486692298758,
     'recall': 0.7808398950131233}




```python
compare_baseline_to_new_results(baseline_results, model_2_results)
```

    Baseline accuracy: 79.27, New accuracy: 78.08, Difference: -1.18
    Baseline precision: 0.81, New precision: 0.78, Difference: -0.03
    Baseline recall: 0.79, New recall: 0.78, Difference: -0.01
    Baseline f1: 0.79, New f1: 0.78, Difference: -0.01
    

## Model 3: GRU

Bir başka popüler ve etkili RNN bileşeni, GRU veya kapılı tekrarlayan birimdir. GRU hücresi, bir LSTM hücresine benzer özelliklere sahiptir ancak daha az parametreye sahiptir.

GRU hücresini TensorFlow'da kullanmak için [`tensorflow.keras.layers.GRU()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU) sınıfını çağırabiliriz.

GRU destekli modelin mimarisi, kullandığımız yapıyla aynı olacak:

```
Input (metin) -> Tokenization -> Embedding -> Layers -> Output (etiket olasılığı)
```
Yine, tek fark, gömme ve çıktı arasında kullandığımız katman(lar) olacaktır.


```python
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GRU(64)(x) 
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")
```

TensorFlow, modellerimizde GRU hücresi gibi güçlü bileşenleri kullanmayı kolaylaştırır. Ve şimdi üçüncü modelimiz yapıldı, eskisi gibi derleyelim.


```python
model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.summary()
```

    Model: "model_3_GRU"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 15)                0         
    _________________________________________________________________
    embedding (Embedding)        (None, 15, 128)           1280000   
    _________________________________________________________________
    gru (GRU)                    (None, 64)                37248     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 1,317,313
    Trainable params: 1,317,313
    Non-trainable params: 0
    _________________________________________________________________
    

`model_2` (LSTM) ve `model_3` (GRU) arasındaki eğitilebilir parametre sayısındaki farka dikkat edin. Fark, GRU hücresinden daha fazla eğitilebilir parametreye sahip LSTM hücresinden gelir.

Modelimize daha önce yaptığımız gibi fit edeceğiz. Ayrıca, `create_tensorboard_callback()` fonksiyonumuzu kullanarak model sonuçlarımızı takip edeceğiz.


```python
model_3_history = model_3.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(SAVE_DIR, "GRU")])
```

    Saving TensorBoard log files to: model_logs/GRU/20210822-070958
    Epoch 1/5
    215/215 [==============================] - 5s 14ms/step - loss: 0.1533 - accuracy: 0.9399 - val_loss: 0.7544 - val_accuracy: 0.7690
    Epoch 2/5
    215/215 [==============================] - 2s 9ms/step - loss: 0.0892 - accuracy: 0.9677 - val_loss: 0.6545 - val_accuracy: 0.7808
    Epoch 3/5
    215/215 [==============================] - 2s 9ms/step - loss: 0.0744 - accuracy: 0.9720 - val_loss: 0.8501 - val_accuracy: 0.7743
    Epoch 4/5
    215/215 [==============================] - 2s 9ms/step - loss: 0.0622 - accuracy: 0.9740 - val_loss: 0.9582 - val_accuracy: 0.7743
    Epoch 5/5
    215/215 [==============================] - 2s 10ms/step - loss: 0.0565 - accuracy: 0.9768 - val_loss: 1.2119 - val_accuracy: 0.7717
    

TensorFlow'daki GRU hücresinin optimize edilmiş varsayılan ayarları nedeniyle, eğitim hiç uzun sürmez. Doğrulama örnekleri üzerinde bazı tahminlerde bulunma zamanı.


```python
model_3_pred_probs = model_3.predict(val_sentences)
model_3_pred_probs.shape, model_3_pred_probs[:10]
```




    ((762, 1), array([[1.6326897e-02],
            [8.8803720e-01],
            [9.9982280e-01],
            [4.5834299e-02],
            [1.4948420e-04],
            [9.9980944e-01],
            [9.7764552e-01],
            [9.9996221e-01],
            [9.9989629e-01],
            [9.8516256e-01]], dtype=float32))



Yine, onları yuvarlayarak tahmin sınıflarına dönüştürebileceğimiz bir dizi tahmin olasılığı elde ederiz.


```python
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 1., 1., 0., 0., 1., 1., 1., 1., 1.], dtype=float32)>



Şimdi tahmini sınıflarımız var, bunları temel doğruluk etiketlerine göre değerlendirelim.


```python
model_3_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_3_preds)
model_3_results
```




    {'accuracy': 77.16535433070865,
     'f1': 0.7712815503325242,
     'precision': 0.7712950933950157,
     'recall': 0.7716535433070866}



Son olarak, GRU modelimizin sonuçlarını taban çizgimizle karşılaştırabiliriz.


```python
compare_baseline_to_new_results(baseline_results, model_3_results)
```

    Baseline accuracy: 79.27, New accuracy: 77.17, Difference: -2.10
    Baseline precision: 0.81, New precision: 0.77, Difference: -0.04
    Baseline recall: 0.79, New recall: 0.77, Difference: -0.02
    Baseline f1: 0.79, New f1: 0.77, Difference: -0.01
    

GRU ile LSTM arasında ki farkı, [StackExchange'de bulunan yorumda](https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm?newreg=64d5f02c755b43d2b855e03bb715e165) çok güzel özetlemiş:
* GRU, LSTM birimi gibi bilgi akışını kontrol eder, ancak bir bellek birimi kullanmak zorunda kalmadan. Herhangi bir kontrol olmaksızın tam gizli içeriği ortaya çıkarır.
* GRU nispeten yeni ve benim açımdan performans LSTM ile eşit, ancak hesaplama açısından daha verimli (belirtildiği gibi daha az karmaşık yapı). Bu yüzden giderek daha fazla kullanıldığını görüyoruz.
* Deneyimlerime göre, dil modelleme yapıyorsanız (diğer görevlerden emin değilsiniz) GRU'lar daha hızlı eğitim alır ve daha az eğitim verisi üzerinde LSTM'lerden daha iyi performans gösterir.
* GRU'lar daha basittir ve bu nedenle değiştirilmesi daha kolaydır.

## Model 4: Çift Yönlü RNN modeli

Halihazırda GRU ve LSTM hücreli iki RNN oluşturduk. Şimdi başka bir tür RNN'yi, çift yönlü RNN'yi inceleyeceğiz.

Standart bir RNN, bir diziyi soldan sağa işleyecektir, burada çift yönlü bir RNN, diziyi soldan sağa ve ardından tekrar sağdan sola işleyecektir.

Sezgisel olarak, bu, bir cümleyi ilk kez normal şekilde (soldan sağa) okuyormuşsunuz gibi düşünülebilir, ancak bir nedenden dolayı bu mantıklı gelmedi, bu yüzden kelimeler arasında geri dönüp tekrar üzerinden geçtiniz. (sağdan sola).

Pratikte, birçok dizi modeli, çift yönlü RNN'leri kullanırken performansta sıklıkla görülür ve gelişme gösterir.

Bununla birlikte, performanstaki bu gelişme genellikle daha uzun eğitim süreleri ve artan model parametreleri pahasına gelir (model soldan sağa ve sağdan sola gittiğinden, eğitilebilir parametrelerin sayısı iki katına çıkar).

Yeterince konuştuk, hadi çift yönlü bir RNN oluşturalım.

TensorFlow bir kez daha [`tensorflow.keras.layers.Bi Directional`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bi Direction) sınıfını sağlayarak bize yardımcı oluyor. Mevcut RNN'lerimizi sarmak için `Bi Directional` sınıfını kullanabilir ve onları anında çift yönlü hale getirebiliriz.


```python
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)

# çift yönlü, her iki yöne de gider, 
# bu nedenle normal bir LSTM katmanının parametrelerinin iki katıdır
x = layers.Bidirectional(layers.LSTM(64))(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")
```

> 🔑 **Not:** TensorFlow'daki herhangi bir RNN hücresinde "Çift Yönlü" sarmalayıcıyı kullanabilirsiniz. Örneğin, `layers.Bidirectional(layers.GRU(64))` çift yönlü bir GRU hücresi oluşturur.

Çift yönlü modelimiz oluşturuldu, derleyelim.


```python
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_4.summary()
```

    Model: "model_4_Bidirectional"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 15)                0         
    _________________________________________________________________
    embedding (Embedding)        (None, 15, 128)           1280000   
    _________________________________________________________________
    bidirectional (Bidirectional (None, 128)               98816     
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 1,378,945
    Trainable params: 1,378,945
    Non-trainable params: 0
    _________________________________________________________________
    

model_2'ye (normal LSTM) kıyasla model_4'te (çift yönlü LSTM) artan eğitilebilir parametre sayısına dikkat edin. Bunun nedeni, RNN'mize eklediğimiz çift yönlülüktür.

Çift yönlü modelimize fit etme ve performansını takip etme zamanı.


```python
model_4_history = model_4.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(SAVE_DIR, "bidirectional_RNN")])
```

    Saving TensorBoard log files to: model_logs/bidirectional_RNN/20210822-071022
    Epoch 1/5
    215/215 [==============================] - 8s 21ms/step - loss: 0.1078 - accuracy: 0.9653 - val_loss: 0.9425 - val_accuracy: 0.7664
    Epoch 2/5
    215/215 [==============================] - 3s 14ms/step - loss: 0.0544 - accuracy: 0.9774 - val_loss: 1.2819 - val_accuracy: 0.7585
    Epoch 3/5
    215/215 [==============================] - 3s 14ms/step - loss: 0.0462 - accuracy: 0.9791 - val_loss: 1.1894 - val_accuracy: 0.7664
    Epoch 4/5
    215/215 [==============================] - 3s 14ms/step - loss: 0.0469 - accuracy: 0.9784 - val_loss: 1.3589 - val_accuracy: 0.7782
    Epoch 5/5
    215/215 [==============================] - 3s 14ms/step - loss: 0.0401 - accuracy: 0.9806 - val_loss: 1.4919 - val_accuracy: 0.7612
    

Modelimizin çift yönlü olması nedeniyle eğitim süresinde hafif bir artış görüyoruz. Endişelenme, çok dramatik bir artış değil. Onunla bazı tahminler yapalım.


```python
model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]
```




    array([[1.2317987e-01],
           [7.2211808e-01],
           [9.9997759e-01],
           [9.2619851e-02],
           [5.5663345e-06],
           [9.9944931e-01],
           [9.8846996e-01],
           [9.9999142e-01],
           [9.9998176e-01],
           [9.5131034e-01]], dtype=float32)



Ve onları tahmin sınıflarına dönüştüreceğiz ve onları temel doğruluk etiketlerine ve temel modele göre değerlendireceğiz.


```python
# # Tahmin olasılıklarını etiketlere dönüştürün
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 1., 1., 0., 0., 1., 1., 1., 1., 1.], dtype=float32)>




```python
# Çift yönlü RNN model sonuçlarını hesaplayın
model_4_results = calculate_results(val_labels, model_4_preds)
model_4_results
```




    {'accuracy': 76.11548556430446,
     'f1': 0.7593763358258425,
     'precision': 0.7618883943071393,
     'recall': 0.7611548556430446}




```python
# Çift yönlü modelin taban çizgisine göre nasıl 
# performans gösterdiğini kontrol edin
compare_baseline_to_new_results(baseline_results, model_4_results)
```

    Baseline accuracy: 79.27, New accuracy: 76.12, Difference: -3.15
    Baseline precision: 0.81, New precision: 0.76, Difference: -0.05
    Baseline recall: 0.79, New recall: 0.76, Difference: -0.03
    Baseline f1: 0.79, New f1: 0.76, Difference: -0.03
    

## Metin için Evrişimli Sinir Ağları

Daha önce görüntüler için evrişimli sinir ağlarını (CNN'ler) kullanmış olabilirsiniz, ancak bunlar diziler için de kullanılabilir.

Görüntüler ve diziler için CNN'leri kullanma arasındaki temel fark, verilerin şeklidir. Görüntüler 2 boyutlu (yükseklik x genişlik) gelirken, diziler genellikle 1 boyutludur (bir metin dizisi).

CNN'leri dizilerle kullanmak için 2 boyutlu evrişim yerine 1 boyutlu evrişim kullanırız.

Diziler için tipik bir CNN mimarisi aşağıdaki gibi görünecektir:

```
Input (metin) -> Tokenization -> Embedding -> Layers -> Output (sınıf olasılıkları)
```

"Bu, diğer modeller için kullandığımız mimari düzene benziyor..." diye düşünüyor olabilirsiniz. Haklısınız da. Fark yine katmanlar bileşenindedir. Bir LSTM veya GRU hücresi kullanmak yerine, bir [`tensorflow.keras.layers.Conv1D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/) katmanı ve ardından bir [`tensorflow.keras.layers.GlobablMaxPool1D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool1D) katmanı kullanacağız.

## Model 5: Conv1D

Tam 1 boyutlu bir CNN modeli oluşturmadan önce, 1 boyutlu evrişim katmanını  çalışırken görelim. Önce bir metin örneğinin gömülmesini oluşturacağız ve onu bir `Conv1D()` katmanı ve `GlobalMaxPool1D()` katmanından geçirmeyi deneyeceğiz.


```python
embedding_test = embedding(text_vectorizer(["this is a test sentence"]))
conv_1d = layers.Conv1D(filters=32, kernel_size=5, activation="relu")
conv_1d_output = conv_1d(embedding_test)
max_pool = layers.GlobalMaxPool1D() 
max_pool_output = max_pool(conv_1d_output)
embedding_test.shape, conv_1d_output.shape, max_pool_output.shape
```




    (TensorShape([1, 15, 128]), TensorShape([1, 11, 32]), TensorShape([1, 32]))



Her katmanın çıktı şekillerine dikkat edin.

Gömme, ayarladığımız parametrelerin çıktı şekli boyutuna sahiptir (`input_length=15` ve`output_dim=128`).

1 boyutlu evrişim katmanı, parametreleriyle aynı hizada sıkıştırılmış bir çıktıya sahiptir. Aynı şey, maksimum havuzlama katmanı çıktısı için de geçerlidir.

Metnimiz bir dize olarak başlar, ancak çeşitli dönüştürme adımlarıyla 64 uzunluğunda bir özellik vektörüne dönüştürülür. Bu dönüşümlerin her birinin neye benzediğine bir bakalım.


```python
embedding_test[:1], conv_1d_output[:1], max_pool_output[:1]
```




    (<tf.Tensor: shape=(1, 15, 128), dtype=float32, numpy=
     array([[[ 0.00319095, -0.01416333, -0.03029248, ..., -0.03457287,
              -0.04297013, -0.04403625],
             [-0.02292195, -0.05479934, -0.00761494, ...,  0.04868896,
               0.05149416,  0.00035047],
             [ 0.01908881, -0.02073449, -0.03212655, ...,  0.04440343,
               0.01428682, -0.01586873],
             ...,
             [-0.00470371,  0.01156277, -0.01759247, ..., -0.00937972,
               0.01118098, -0.00774161],
             [-0.00470371,  0.01156277, -0.01759247, ..., -0.00937972,
               0.01118098, -0.00774161],
             [-0.00470371,  0.01156277, -0.01759247, ..., -0.00937972,
               0.01118098, -0.00774161]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 11, 32), dtype=float32, numpy=
     array([[[0.00000000e+00, 8.05034712e-02, 0.00000000e+00, 1.15600638e-02,
              0.00000000e+00, 5.81747759e-03, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 1.27381273e-02, 3.38597856e-02, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 4.80861589e-02, 0.00000000e+00,
              3.96751724e-02, 0.00000000e+00, 0.00000000e+00, 3.05385999e-02,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 4.12951857e-02, 0.00000000e+00, 6.95906281e-02,
              0.00000000e+00, 5.94841614e-02, 9.18674469e-02, 5.20823263e-02],
             [2.63430271e-02, 8.44356269e-02, 2.81330571e-03, 6.74604177e-02,
              0.00000000e+00, 8.42802040e-03, 0.00000000e+00, 1.10559706e-02,
              4.07118723e-02, 0.00000000e+00, 0.00000000e+00, 3.80547568e-02,
              0.00000000e+00, 8.79763439e-02, 4.59032431e-02, 3.13887000e-02,
              2.62255073e-02, 4.18251473e-03, 0.00000000e+00, 0.00000000e+00,
              1.02217998e-02, 0.00000000e+00, 9.08166170e-03, 2.56322473e-02,
              7.09645683e-03, 1.22743607e-01, 0.00000000e+00, 3.41253951e-02,
              0.00000000e+00, 0.00000000e+00, 6.04886301e-02, 4.24289927e-02],
             [0.00000000e+00, 2.03429386e-02, 0.00000000e+00, 9.06437784e-02,
              0.00000000e+00, 1.33926468e-02, 6.44886717e-02, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 5.57984598e-03, 0.00000000e+00,
              2.80500129e-02, 3.06625962e-02, 1.82882883e-02, 0.00000000e+00,
              0.00000000e+00, 1.14728551e-04, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.61776145e-02,
              2.19251886e-02, 6.81912601e-02, 0.00000000e+00, 2.30189972e-02,
              0.00000000e+00, 1.26050590e-02, 1.36954803e-02, 0.00000000e+00],
             [0.00000000e+00, 7.14183897e-02, 0.00000000e+00, 7.31983259e-02,
              0.00000000e+00, 0.00000000e+00, 1.36040617e-02, 0.00000000e+00,
              0.00000000e+00, 4.47261706e-02, 0.00000000e+00, 0.00000000e+00,
              1.97688714e-02, 0.00000000e+00, 2.63667684e-02, 1.55701367e-02,
              9.24898498e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.16558182e-02, 0.00000000e+00, 1.84091628e-02,
              0.00000000e+00, 9.06079710e-02, 0.00000000e+00, 5.93137965e-02,
              0.00000000e+00, 7.36142173e-02, 2.71241888e-02, 0.00000000e+00],
             [0.00000000e+00, 5.90131134e-02, 0.00000000e+00, 2.19131652e-02,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 2.85770036e-02, 3.44222710e-02, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 2.71796528e-02, 5.12055531e-02,
              5.00489324e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.82179581e-02, 1.01004215e-02, 0.00000000e+00,
              3.42076607e-02, 5.01884520e-02, 0.00000000e+00, 6.73409179e-02,
              1.43867647e-02, 5.38122803e-02, 5.80260493e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517823e-02, 6.49180636e-03, 2.89049260e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823136e-02, 2.80115753e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30828079e-02, 3.57510448e-02,
              1.88314561e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870250e-02, 4.26470116e-02, 1.91313103e-02,
              1.24170668e-02, 6.92164451e-02, 0.00000000e+00, 4.52357307e-02,
              1.14318049e-02, 4.45467979e-02, 2.09700465e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517749e-02, 6.49180962e-03, 2.89049223e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823210e-02, 2.80119071e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30828042e-02, 3.57510559e-02,
              1.88314579e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870324e-02, 4.26470190e-02, 1.91313159e-02,
              1.24170687e-02, 6.92164376e-02, 0.00000000e+00, 4.52357307e-02,
              1.14318095e-02, 4.45467979e-02, 2.09700502e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517749e-02, 6.49180589e-03, 2.89049186e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823173e-02, 2.80117441e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30827995e-02, 3.57510522e-02,
              1.88314561e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870324e-02, 4.26470190e-02, 1.91313140e-02,
              1.24170706e-02, 6.92164376e-02, 0.00000000e+00, 4.52357270e-02,
              1.14318077e-02, 4.45467979e-02, 2.09700502e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517786e-02, 6.49181195e-03, 2.89049149e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823210e-02, 2.80118053e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30827948e-02, 3.57510522e-02,
              1.88314617e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870250e-02, 4.26470228e-02, 1.91313084e-02,
              1.24170706e-02, 6.92164451e-02, 0.00000000e+00, 4.52357233e-02,
              1.14318114e-02, 4.45468016e-02, 2.09700502e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517711e-02, 6.49181567e-03, 2.89049074e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823210e-02, 2.80116277e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30828023e-02, 3.57510485e-02,
              1.88314617e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870324e-02, 4.26470228e-02, 1.91313103e-02,
              1.24170696e-02, 6.92164302e-02, 0.00000000e+00, 4.52357307e-02,
              1.14318039e-02, 4.45467904e-02, 2.09700465e-02, 0.00000000e+00],
             [0.00000000e+00, 4.34517749e-02, 6.49181474e-03, 2.89049149e-02,
              0.00000000e+00, 0.00000000e+00, 1.56823229e-02, 2.80122069e-04,
              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 0.00000000e+00, 1.30828032e-02, 3.57510522e-02,
              1.88314542e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
              0.00000000e+00, 3.27870250e-02, 4.26470190e-02, 1.91313140e-02,
              1.24170706e-02, 6.92164451e-02, 0.00000000e+00, 4.52357233e-02,
              1.14318067e-02, 4.45467941e-02, 2.09700484e-02, 0.00000000e+00]]],
           dtype=float32)>,
     <tf.Tensor: shape=(1, 32), dtype=float32, numpy=
     array([[0.02634303, 0.08443563, 0.00649182, 0.09064378, 0.        ,
             0.01339265, 0.06448867, 0.01105597, 0.04071187, 0.04472617,
             0.03442227, 0.03805476, 0.02805001, 0.08797634, 0.04808616,
             0.05120555, 0.05004893, 0.00418251, 0.        , 0.0305386 ,
             0.0102218 , 0.03821796, 0.04264702, 0.04617761, 0.03420766,
             0.12274361, 0.        , 0.06959063, 0.01438676, 0.07361422,
             0.09186745, 0.05208233]], dtype=float32)>)



Pekala, diziler için bir CNN'nin çeşitli bileşenlerinin çıktılarını gördük, onları bir araya getirelim ve tam bir model oluşturalım, onu derleyelim (tıpkı diğer modellerimizde yaptığımız gibi) ve bir özet alalım.


```python
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = layers.GlobalMaxPool1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")

model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_5.summary()
```

    Model: "model_5_Conv1D"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 15)                0         
    _________________________________________________________________
    embedding (Embedding)        (None, 15, 128)           1280000   
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 11, 32)            20512     
    _________________________________________________________________
    global_max_pooling1d_1 (Glob (None, 32)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 1,300,545
    Trainable params: 1,300,545
    Non-trainable params: 0
    _________________________________________________________________
    

Süperr! Harika görünüyor! 1-boyutlu evrişimli katman için eğitilebilir parametre sayısının `model_2`'deki LSTM katmanınınkine nasıl benzer olduğuna dikkat edin.

1D CNN modelimizi metin verilerimizle fit edelim. Önceki deneylere uygun olarak, `create_tensorboard_callback()` fonksiyonumuzu kullanarak sonuçlarını kaydedeceğiz.


```python
model_5_history = model_5.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, 
                                                                     "Conv1D")])
```

    Saving TensorBoard log files to: model_logs/Conv1D/20210822-071110
    Epoch 1/5
    215/215 [==============================] - 4s 10ms/step - loss: 0.1339 - accuracy: 0.9584 - val_loss: 0.8689 - val_accuracy: 0.7664
    Epoch 2/5
    215/215 [==============================] - 2s 7ms/step - loss: 0.0763 - accuracy: 0.9727 - val_loss: 0.9643 - val_accuracy: 0.7651
    Epoch 3/5
    215/215 [==============================] - 2s 7ms/step - loss: 0.0600 - accuracy: 0.9761 - val_loss: 1.0885 - val_accuracy: 0.7546
    Epoch 4/5
    215/215 [==============================] - 2s 7ms/step - loss: 0.0542 - accuracy: 0.9783 - val_loss: 1.1663 - val_accuracy: 0.7559
    Epoch 5/5
    215/215 [==============================] - 2s 7ms/step - loss: 0.0512 - accuracy: 0.9790 - val_loss: 1.2285 - val_accuracy: 0.7572
    

Güzel! GPU hızlandırma sayesinde 1D evrişimli modelimiz güzel ve hızlı bir şekilde eğitiyor. Onunla bazı tahminler yapalım ve eskisi gibi değerlendirelim.


```python
model_5_pred_probs = model_5.predict(val_sentences)
model_5_pred_probs[:10]
```




    array([[1.3275287e-01],
           [3.1561503e-01],
           [9.9978405e-01],
           [5.4979675e-02],
           [2.5409597e-07],
           [9.8810232e-01],
           [9.4828182e-01],
           [9.9995220e-01],
           [9.9999881e-01],
           [9.4418395e-01]], dtype=float32)




```python
model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
model_5_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 0., 1., 0., 0., 1., 1., 1., 1., 1.], dtype=float32)>




```python
model_5_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_5_preds)
model_5_results
```




    {'accuracy': 75.7217847769029,
     'f1': 0.7549472494674683,
     'precision': 0.7585989210360964,
     'recall': 0.7572178477690289}




```python
compare_baseline_to_new_results(baseline_results, model_5_results)
```

    Baseline accuracy: 79.27, New accuracy: 75.72, Difference: -3.54
    Baseline precision: 0.81, New precision: 0.76, Difference: -0.05
    Baseline recall: 0.79, New recall: 0.76, Difference: -0.04
    Baseline f1: 0.79, New f1: 0.75, Difference: -0.03
    

# Önceden Eğitilmiş Embedleri Kullanma (NLP için transfer öğrenimi)

Oluşturduğumuz ve eğittiğimiz önceki tüm derin öğrenme modelleri için her seferinde sıfırdan kendi yerleştirmelerimizi oluşturduk ve kullandık.

Ancak yaygın bir uygulama, **aktarım öğrenimi** aracılığıyla önceden eğitilmiş embedlerder yararlanmaktır. Bir sonraki modelimiz için, kendi gömme katmanımızı kullanmak yerine, onu önceden eğitilmiş bir gömme katmanıyla değiştireceğiz.

Daha spesifik olarak, [TensorFlow Hub](https://tfhub.dev/google) adresinden [Universal Sentence Encoder](https://www.aclweb.org/anthology/D18-2029.pdf) kullanacağız. (universal-sentence-encoder, çeşitli görevler için çok sayıda önceden eğitilmiş model kaynağı içeren harika bir model).

> 🔑 **Not:** TensorFlow Hub'da önceden eğitilmiş birçok farklı metin gömme seçeneği vardır, ancak bazıları diğerlerinden farklı seviyelerde metin ön işleme gerektirir. Birkaçını denemek ve kullanım durumunuza en uygun olanı görmek en iyisidir.

## Model 6: TensorFlow Hub Önceden Eğitilmiş Cümle Kodlayıcı

Oluşturduğumuz gömme katmanı ile Evrensel Cümle Kodlayıcı arasındaki temel fark, tahmin edebileceğiniz gibi, Evrensel Cümle Kodlayıcı'nın sözcük düzeyinde bir gömme oluşturmak yerine, tam bir cümle düzeyinde gömme oluşturmasıdır.

Gömme katmanımız ayrıca her kelime için 128 boyutlu bir vektör üretirken, Evrensel Cümle Kodlayıcı her cümle için 512 boyutlu bir vektör verir.

> 🔑 **Not:** Bir **encoder**, metin gibi ham verileri sayısal bir gösterime (özellik vektörü) dönüştüren bir modelin adıdır, bir **decoder** sayısal gösterimi istenen bir çıktıya dönüştürür .

Her zamanki gibi, bu en iyi bir örnekle gösterilir. Universal (evrensel) Cümle Kodlayıcı modelini yükleyelim ve birkaç cümle üzerinde test edelim.


```python
import tensorflow_hub as hub

# Evrensel Cümle Kodlayıcıyı yükle
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
embed_samples = embed([sample_sentence,
                       "When you call the universal sentence encoder \
                       on a sentence, it turns it into numbers."])

print(embed_samples[0][:50])
```

    tf.Tensor(
    [-0.01157024  0.0248591   0.0287805  -0.01271502  0.03971543  0.08827759
      0.02680986  0.05589837 -0.01068731 -0.0059729   0.00639324 -0.01819523
      0.00030817  0.09105891  0.05874644 -0.03180627  0.01512476 -0.05162928
      0.00991369 -0.06865346 -0.04209306  0.0267898   0.03011008  0.00321069
     -0.00337969 -0.04787359  0.02266718 -0.00985924 -0.04063614 -0.01292095
     -0.04666384  0.056303   -0.03949255  0.00517685  0.02495828 -0.07014439
      0.02871508  0.04947682 -0.00633971 -0.08960191  0.02807117 -0.00808362
     -0.01360601  0.05998649 -0.10361786 -0.05195372  0.00232955 -0.02332528
     -0.03758105  0.0332773 ], shape=(50,), dtype=float32)
    


```python
# Her cümle 512 boyutlu bir vektöre kodlanmıştır
embed_samples[0].shape
```




    TensorShape([512])



Cümlelerimizi Evrensel Cümle Kodlayıcıya (USE) geçirmek, onları dizelerden 512 boyutlu vektörlere kodlar; bu bizim için hiçbir anlam ifade etmez, ancak umarım makine öğrenimi modellerimiz için bir anlam ifade eder.

Modellerden bahsetmişken, gömme katmanımız olarak USE ile bir tane oluşturalım.

[`hub.KerasLayer`](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) sınıfını kullanarak TensorFlow Hub USE modülünü Keras katmanına dönüştürebiliriz.

> 🔑 **Not:** TensorFlow Hub'ı KULLAN modülünün boyutu nedeniyle, indirilmesi biraz zaman alabilir. Yine de indirildikten sonra önbelleğe alınacak ve kullanıma hazır olacaktır. Ve birçok TensorFlow Hub modülünde olduğu gibi, USE'nin daha az yer kaplayan ancak performanstan biraz ödün veren bir ["lite" sürümü](https://tfhub.dev/google/universal-sentence-encoder-lite/2) vardır. ve daha fazla ön işleme adımı gerektirir. Ancak, mevcut işlem gücünüze bağlı olarak, uygulama kullanım durumunuz için lite sürümü daha iyi olabilir.


```python
# Bu kodlama katmanını text_vectorizer ve gömme katmanımız yerine kullanabiliriz
sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[], # modelimize gelen girdilerin şekli
    dtype=tf.string, # USE katmanına gelen veri tipi girdiler
    trainable=False, # önceden eğitilmiş ağırlıkları koru
    name="USE") 
```

Güzel! Şimdi Keras katmanı olarak USE'ye sahibiz, onu Keras Sıralı modelinde kullanabiliriz.


```python
model_6 = tf.keras.Sequential([
  sentence_encoder_layer,
  layers.Dense(64, activation="relu"),
  layers.Dense(1, activation="sigmoid")
], name="model_6_USE")

# modeli derleme
model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_6.summary()
```

    Model: "model_6_USE"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    USE (KerasLayer)             (None, 512)               256797824 
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                32832     
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 256,830,721
    Trainable params: 32,897
    Non-trainable params: 256,797,824
    _________________________________________________________________
    

USE katmanındaki parametrelerin sayısına dikkat edin, bunlar çeşitli metin kaynaklarında (Wikipedia, web haberleri, web soru-cevap forumları vb.) öğrendiği önceden eğitilmiş ağırlıklardır.

Eğitilebilir parametreler yalnızca çıktı katmanlarımızdadır, başka bir deyişle, USE ağırlıklarını donmuş halde tutuyor ve onu bir özellik çıkarıcı olarak kullanıyoruz. "`hub.KerasLayer`" örneğini oluştururken "`trainable=True`" ayarını yaparak bu ağırlıklara ince ayar yapabiliriz.

Şimdi hazır bir özellik çıkarıcı modelimiz var, hadi onu eğitelim ve `create_tensorboard_callback()` fonksiyonumuzu kullanarak sonuçlarını TensorBoard'a izleyelim.


```python
model_6_history = model_6.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(SAVE_DIR, "tf_hub_sentence_encoder")])
```

    Saving TensorBoard log files to: model_logs/tf_hub_sentence_encoder/20210822-071154
    Epoch 1/5
    215/215 [==============================] - 10s 31ms/step - loss: 0.5060 - accuracy: 0.7787 - val_loss: 0.4518 - val_accuracy: 0.8005
    Epoch 2/5
    215/215 [==============================] - 4s 18ms/step - loss: 0.4145 - accuracy: 0.8183 - val_loss: 0.4373 - val_accuracy: 0.8110
    Epoch 3/5
    215/215 [==============================] - 4s 18ms/step - loss: 0.4018 - accuracy: 0.8206 - val_loss: 0.4317 - val_accuracy: 0.8150
    Epoch 4/5
    215/215 [==============================] - 4s 17ms/step - loss: 0.3938 - accuracy: 0.8269 - val_loss: 0.4276 - val_accuracy: 0.8176
    Epoch 5/5
    215/215 [==============================] - 4s 17ms/step - loss: 0.3888 - accuracy: 0.8278 - val_loss: 0.4275 - val_accuracy: 0.8202
    

Diğer modellerimizde yaptığımız gibi onunla da bazı tahminler yapalım ve onları değerlendirelim.


```python
# USE TF Hub modeli ile tahminler yapın
model_6_pred_probs = model_6.predict(val_sentences)
model_6_pred_probs[:10]
```




    array([[0.16858765],
           [0.78706455],
           [0.9866457 ],
           [0.18491879],
           [0.74524015],
           [0.7418657 ],
           [0.9791367 ],
           [0.97791415],
           [0.94224715],
           [0.0893833 ]], dtype=float32)




```python
# Tahmin olasılıklarını etiketlere dönüştürün
model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 1., 1., 0., 1., 1., 1., 1., 1., 0.], dtype=float32)>




```python
# Model 6 performans metriklerini hesaplayın
model_6_results = calculate_results(val_labels, model_6_preds)
model_6_results
```




    {'accuracy': 82.02099737532808,
     'f1': 0.8187283797448478,
     'precision': 0.8226273514784717,
     'recall': 0.8202099737532809}




```python
# Karşılaştırma
compare_baseline_to_new_results(baseline_results, model_6_results)
```

    Baseline accuracy: 79.27, New accuracy: 82.02, Difference: 2.76
    Baseline precision: 0.81, New precision: 0.82, Difference: 0.01
    Baseline recall: 0.79, New recall: 0.82, Difference: 0.03
    Baseline f1: 0.79, New f1: 0.82, Difference: 0.03
    

## Model 7

USE içindeki önceden eğitilmiş yerleştirmeler gibi transfer öğrenme yöntemlerini kullanmanın faydalarından biri, az miktarda veri üzerinde harika sonuçlar elde etme yeteneğidir (USE makalesi özette bundan bahseder).

Bunu test etmek için, eğitim verilerinin küçük bir alt kümesini (%10) oluşturacağız, bir model eğiteceğiz ve onu değerlendireceğiz.


```python
train_10_percent = train_df_shuffled[["text", "target"]].sample(frac=0.1, random_state=42)
train_sentences_10_percent = train_10_percent["text"].to_list()
train_labels_10_percent = train_10_percent["target"].to_list()
len(train_sentences_10_percent), len(train_labels_10_percent)

train_sentences_90_percent, train_sentences_10_percent, train_labels_90_percent, train_labels_10_percent = train_test_split(np.array(train_sentences),
                                                                                                                            train_labels,
                                                                                                                            test_size=0.1,
                                                                                                                            random_state=42)
print(f"Total training examples: {len(train_sentences)}")
print(f"Length of 10% training examples: {len(train_sentences_10_percent)}")
```

    Total training examples: 6851
    Length of 10% training examples: 686
    

Eğitim örneklerinin rastgele bir alt kümesini seçtiğimiz için, sınıfların kabaca dengelenmesi gerekir (tam eğitim veri kümesinde olduğu gibi).


```python
# Veri alt kümemizdeki hedef sayısını kontrol edin
# (bu, orijinal train_labels içindeki etiketlerin dağılımına yakın olmalıdır)
pd.Series(train_labels_10_percent).value_counts()
```




    0    415
    1    271
    dtype: int64



Modelimizin tam eğitim kümesinden öğrenme yeteneği ile %10 alt kümeden öğrenme yeteneği arasında uygun bir karşılaştırma yaptığımızdan emin olmak için, [`tf.keras.models.clone_model()` kullanarak USE modelimizi ("model_6") kullanarak klonlayacağız.`](https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model)

Bunu yapmak aynı mimariyi yaratacak ancak klon hedefinin öğrenilen ağırlıklarını sıfırlayacaktır (USE'den gelen önceden eğitilmiş ağırlıklar kalacak, ancak diğerleri sıfırlanacaktır).


```python
model_7 = tf.keras.models.clone_model(model_6)

model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_7.summary()
```

    Model: "model_6_USE"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    USE (KerasLayer)             (None, 512)               256797824 
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                32832     
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 256,830,721
    Trainable params: 32,897
    Non-trainable params: 256,797,824
    _________________________________________________________________
    

`model_7` düzeninin `model_6` ile aynı olduğuna dikkat edin. Şimdi yeni oluşturulan modeli %10 eğitim verisi alt kümemizde eğitelim.


```python
model_7_history = model_7.fit(
    x=train_sentences_10_percent,
    y=train_labels_10_percent,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[create_tensorboard_callback(SAVE_DIR, "10_percent_tf_hub_sentence_encoder")])
```

    Saving TensorBoard log files to: model_logs/10_percent_tf_hub_sentence_encoder/20210822-071224
    Epoch 1/5
    22/22 [==============================] - 6s 154ms/step - loss: 0.6651 - accuracy: 0.6851 - val_loss: 0.6443 - val_accuracy: 0.6929
    Epoch 2/5
    22/22 [==============================] - 1s 46ms/step - loss: 0.5920 - accuracy: 0.7959 - val_loss: 0.5918 - val_accuracy: 0.7310
    Epoch 3/5
    22/22 [==============================] - 1s 35ms/step - loss: 0.5166 - accuracy: 0.8163 - val_loss: 0.5415 - val_accuracy: 0.7638
    Epoch 4/5
    22/22 [==============================] - 1s 33ms/step - loss: 0.4554 - accuracy: 0.8382 - val_loss: 0.5058 - val_accuracy: 0.7795
    Epoch 5/5
    22/22 [==============================] - 1s 32ms/step - loss: 0.4120 - accuracy: 0.8382 - val_loss: 0.4925 - val_accuracy: 0.7730
    

Daha az miktarda eğitim verisi nedeniyle eğitim, eskisinden daha hızlı bitti. Eğitim verilerinin %10'unu öğrendikten sonra modelimizin performansını değerlendirelim.


```python
model_7_pred_probs = model_7.predict(val_sentences)
model_7_pred_probs[:10]
```




    array([[0.26725993],
           [0.7663658 ],
           [0.8697842 ],
           [0.30411467],
           [0.5334834 ],
           [0.82516   ],
           [0.8102658 ],
           [0.84461933],
           [0.8153697 ],
           [0.10006839]], dtype=float32)




```python
model_7_preds = tf.squeeze(tf.round(model_7_pred_probs))
model_7_preds[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 1., 1., 0., 1., 1., 1., 1., 1., 0.], dtype=float32)>




```python
model_7_results = calculate_results(val_labels, model_7_preds)
model_7_results
```




    {'accuracy': 77.29658792650919,
     'f1': 0.7698502254147366,
     'precision': 0.7770640736660165,
     'recall': 0.7729658792650919}




```python
compare_baseline_to_new_results(baseline_results, model_7_results)
```

    Baseline accuracy: 79.27, New accuracy: 77.30, Difference: -1.97
    Baseline precision: 0.81, New precision: 0.78, Difference: -0.03
    Baseline recall: 0.79, New recall: 0.77, Difference: -0.02
    Baseline f1: 0.79, New f1: 0.77, Difference: -0.02
    

# Modellerimizin Her Birinin Performansını Karşılaştırma

Şimdi modelimizin sonuçlarını karşılaştırma zamanı. Ancak bundan hemen önce, bu tür bir uygulamanın standart bir derin öğrenme iş akışı olduğunu belirtmekte fayda var. Çeşitli farklı modelleri eğitin, ardından hangisinin en iyi performansı gösterdiğini görmek için bunları karşılaştırın ve gerekirse onu eğitmeye devam edin.

Unutulmaması gereken önemli nokta, tüm modelleme deneylerimiz için aynı eğitim verilerini kullandığımızdır (eğitim verilerinin %10'unu kullandığımız 'model_7' hariç).

Modelimizin performanslarını görselleştirmek için, result sözlüklerimiz olan bir pandas DataFrame oluşturalım ve sonra onu çizelim.


```python
all_model_results = pd.DataFrame({"baseline": baseline_results,
                                  "simple_dense": model_1_results,
                                  "lstm": model_2_results,
                                  "gru": model_3_results,
                                  "bidirectional": model_4_results,
                                  "conv1d": model_5_results,
                                  "tf_hub_sentence_encoder": model_6_results,
                                  "tf_hub_10_percent_data": model_7_results})
all_model_results = all_model_results.transpose()
all_model_results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>79.265092</td>
      <td>0.811139</td>
      <td>0.792651</td>
      <td>0.786219</td>
    </tr>
    <tr>
      <th>simple_dense</th>
      <td>78.477690</td>
      <td>0.789165</td>
      <td>0.784777</td>
      <td>0.781896</td>
    </tr>
    <tr>
      <th>lstm</th>
      <td>78.083990</td>
      <td>0.781487</td>
      <td>0.780840</td>
      <td>0.779482</td>
    </tr>
    <tr>
      <th>gru</th>
      <td>77.165354</td>
      <td>0.771295</td>
      <td>0.771654</td>
      <td>0.771282</td>
    </tr>
    <tr>
      <th>bidirectional</th>
      <td>76.115486</td>
      <td>0.761888</td>
      <td>0.761155</td>
      <td>0.759376</td>
    </tr>
    <tr>
      <th>conv1d</th>
      <td>75.721785</td>
      <td>0.758599</td>
      <td>0.757218</td>
      <td>0.754947</td>
    </tr>
    <tr>
      <th>tf_hub_sentence_encoder</th>
      <td>82.020997</td>
      <td>0.822627</td>
      <td>0.820210</td>
      <td>0.818728</td>
    </tr>
    <tr>
      <th>tf_hub_10_percent_data</th>
      <td>77.296588</td>
      <td>0.777064</td>
      <td>0.772966</td>
      <td>0.769850</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Doğruluğu diğer metriklerle aynı ölçeğe indirin
all_model_results["accuracy"] = all_model_results["accuracy"]/100
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));
```


    
![png](8-NLP%27ye%20Giris_files/8-NLP%27ye%20Giris_169_0.png)
    


Önceden eğitilmiş USE TensorFlow Hub modellerimiz en iyi performansa sahip gibi görünüyor, eğitim verilerinin yalnızca %10'una sahip olan model bile diğer modellerden daha iyi performans gösteriyor. Bu, transfer öğrenmenin gücünü gösterir.

Detaylara inip her modelin F1 puanlarını almaya ne dersiniz?


```python
all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7));
```


    
![png](8-NLP%27ye%20Giris_files/8-NLP%27ye%20Giris_171_0.png)
    


Tek bir ölçümde detaya indiğimizde, USE TensorFlow Hub modellerimizin diğer tüm modellerden daha iyi performans gösterdiğini görüyoruz. İlginç bir şekilde, temelin F1 puanı, daha derin modellerin geri kalanından çok uzakta değil.

## Modellerimizi Birleştirmek

Birçok üretim sistemi, bir tahmin yapmak için bir **ensemble** (birden çok farklı modelin bir araya getirilmesi) modellerini kullanır.

Model istiflemenin ardındaki fikir, birbiriyle ilişkisiz birkaç modelin bir tahmin üzerinde anlaşmaya varması durumunda, tahminin tekil bir model tarafından yapılan bir tahminden daha sağlam olması gerektiğidir.

Yukarıdaki cümledeki anahtar kelime **uncorrelated**, bu da farklı model türleri demenin başka bir yoludur. Örneğin, bizim durumumuzda taban çizgimizi, çift yönlü modelimizi ve TensorFlow Hub USE modelimizi birleştirebiliriz.

Bu modellerin hepsi aynı veriler üzerinde eğitilmiş olsa da, hepsinin farklı bir kalıp bulma yolu vardır.

Üç LSTM modeli gibi benzer şekilde eğitilmiş üç model kullanacak olsaydık, çıktı tahminleri muhtemelen çok benzer olacaktır.

Bunu arkadaşlarınızla nerede yemek yiyeceğinize karar vermeye çalışmak olarak düşünün. Hepinizin zevkleri benzerse, muhtemelen hepiniz aynı restoranı seçeceksiniz. Ama hepinizin farklı zevkleri varsa ve yine de aynı restoranı seçerseniz, restoran iyi olmalı.

Bir sınıflandırma problemi ile çalıştığımız için modellerimizi birleştirmenin birkaç yolu vardır:
1. **Ortalama** - Her örnek için her modelin çıktı tahmin olasılıklarını alın, birleştirin ve ardından ortalamasını alın.
2. **Çoğunluk oyu (mod)** - Modellerinizin her biri ile tüm örneklerde sınıf tahminleri yapın, tahmin edilen sınıf çoğunlukta olandır. Örneğin, üç farklı model sırasıyla `[1, 0, 1]` değerini tahmin ederse, çoğunluk sınıfı `1` olur, bu nedenle bu tahmin edilen etiket olacaktır.
3. **Model yığınlama** - Seçtiğiniz modellerin her birinin çıktılarını alın ve bunları başka bir modele girdi olarak kullanın.

> 📖 **Kaynak:** Model istifleme/birleştirme için yukarıdaki yöntemler, Andriy Burkov tarafından [Machine Learning Engineering Book](http://www.mlebook.com/wiki/doku.php) Bölüm 6'dan uyarlanmıştır. Makine öğrenimi mühendisliği alanına girmek, yalnızca modeller oluşturmak değil, aynı zamanda üretim ölçeğinde makine öğrenimi sistemleri kurmak istiyorsanız, tamamını okumanızı şiddetle tavsiye ederim.

Yine, model istifleme kavramı en iyi eylemde görülür.

Temel modelimizi (`model_0`), LSTM modelimizi (`model_2`) ve tam eğitim verisi (`model_6`) üzerinde eğitilmiş USE modelimizi, her birinin birleşik tahmin olasılıklarının ortalamasını alarak birleştireceğiz.


```python
# temel modelden tahmin olasılıklarını alın
baseline_pred_probs = np.max(model_0.predict_proba(val_sentences), axis=1)
combined_pred_probs = baseline_pred_probs + tf.squeeze(model_2_pred_probs, axis=1) + tf.squeeze(model_6_pred_probs)
# tahmin sınıfları için olasılıklarını ortalamasını alın ve yuvarlayın
combined_preds = tf.round(combined_pred_probs/3)
combined_preds[:20]
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1.], dtype=float32)>



Olağanüstü! Farklı sınıflardan oluşan birleştirilmiş bir tahminler dizimiz var, bunları gerçek etiketlere göre değerlendirelim ve yığılmış modelimizin sonuçlarını `all_model_results` DataFrame'imize ekleyelim.


```python
ensemble_results = calculate_results(val_labels, combined_preds)
ensemble_results
```




    {'accuracy': 79.92125984251969,
     'f1': 0.7991404257926901,
     'precision': 0.7990931458872615,
     'recall': 0.7992125984251969}




```python
# Birleştirilmiş modelimizin sonuçlarını DataFrame sonuçlarına ekleyin
all_model_results.loc["ensemble_results"] = ensemble_results
# Doğruluğu, sonuçların geri kalanıyla aynı ölçeğe dönüştürün
all_model_results.loc["ensemble_results"]["accuracy"] = all_model_results.loc["ensemble_results"]["accuracy"]/100

all_model_results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.792651</td>
      <td>0.811139</td>
      <td>0.792651</td>
      <td>0.786219</td>
    </tr>
    <tr>
      <th>simple_dense</th>
      <td>0.784777</td>
      <td>0.789165</td>
      <td>0.784777</td>
      <td>0.781896</td>
    </tr>
    <tr>
      <th>lstm</th>
      <td>0.780840</td>
      <td>0.781487</td>
      <td>0.780840</td>
      <td>0.779482</td>
    </tr>
    <tr>
      <th>gru</th>
      <td>0.771654</td>
      <td>0.771295</td>
      <td>0.771654</td>
      <td>0.771282</td>
    </tr>
    <tr>
      <th>bidirectional</th>
      <td>0.761155</td>
      <td>0.761888</td>
      <td>0.761155</td>
      <td>0.759376</td>
    </tr>
    <tr>
      <th>conv1d</th>
      <td>0.757218</td>
      <td>0.758599</td>
      <td>0.757218</td>
      <td>0.754947</td>
    </tr>
    <tr>
      <th>tf_hub_sentence_encoder</th>
      <td>0.820210</td>
      <td>0.822627</td>
      <td>0.820210</td>
      <td>0.818728</td>
    </tr>
    <tr>
      <th>tf_hub_10_percent_data</th>
      <td>0.772966</td>
      <td>0.777064</td>
      <td>0.772966</td>
      <td>0.769850</td>
    </tr>
    <tr>
      <th>ensemble_results</th>
      <td>0.799213</td>
      <td>0.799093</td>
      <td>0.799213</td>
      <td>0.799140</td>
    </tr>
  </tbody>
</table>
</div>



Yığılmış model diğer modellere karşı nasıl bir sonuç verdi?

> 🔑 **Not:** Modelimizin sonuçlarının çoğu benzer görünüyor. Bu, verilerimizden öğrenilebileceklerin bazı sınırlamaları olduğu anlamına gelebilir. Modelleme denemelerinizin çoğu benzer sonuçlar verdiğinde, verilerinizi tekrar gözden geçirmek iyi bir fikirdir.

## Eğitilmiş Bir Modeli Kaydetme ve Yükleme

Eğitim süresi çok uzun sürmese de, yeniden eğitmek zorunda kalmamak için eğitilmiş modellerinizi kaydetmek iyi bir uygulamadır.

Modellerinizi kaydetmek, aynı zamanda, bir web uygulamasında olduğu gibi, dizüstü bilgisayarınızın dışında başka bir yerde kullanmak üzere dışa aktarmanıza da olanak tanır.

[TensorFlow'da bir modeli kaydetmenin] iki ana yolu vardır(https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model):
1. `HDF5` biçimi.
2. `KayıtlıModel` biçimi (varsayılan).

İkisine de bir göz atalım.


```python
model_6.save("model_6.h5")
```

Bir modeli `HDF5` olarak kaydederseniz, tekrar yüklerken TensorFlow'a kullandığınız özel nesneler hakkında bilgi vermeniz gerekir.


```python
# Modeli özel Hub Katmanı ile yükleyin (HDF5 formatı için gereklidir)
loaded_model_6 = tf.keras.models.load_model(
    "model_6.h5", 
    custom_objects={"KerasLayer": hub.KerasLayer})
```


```python
# Yüklenen modelimiz nasıl performans gösteriyor?
loaded_model_6.evaluate(val_sentences, val_labels)
```

    24/24 [==============================] - 1s 16ms/step - loss: 0.4275 - accuracy: 0.8202
    




    [0.4274521470069885, 0.8202099800109863]



Hedef modelimizde `save()` yöntemini çağırmak ve ona bir dosya yolu iletmek, modelimizi `SavedModel` formatında kaydetmemizi sağlar.


```python
model_6.save("model_6_SavedModel_format")
```

    WARNING:absl:Function `_wrapped_model` contains input name(s) USE_input with unsupported characters which will be renamed to use_input in the SavedModel.
    

    INFO:tensorflow:Assets written to: model_6_SavedModel_format/assets
    

    INFO:tensorflow:Assets written to: model_6_SavedModel_format/assets
    

SavedModel biçimini (varsayılan) kullanırsanız, `tensorflow.keras.models.load_model()` işlevini kullanarak özel nesneler belirtmeden modelinizi yeniden yükleyebilirsiniz.


```python
# TF Hub Cümle Kodlayıcıyı Yükle SavedModel
loaded_model_6_SavedModel = tf.keras.models.load_model("model_6_SavedModel_format")
```


```python
# Yüklenen SavedModel biçimini değerlendirin
loaded_model_6_SavedModel.evaluate(val_sentences, val_labels)
```

    24/24 [==============================] - 1s 15ms/step - loss: 0.4275 - accuracy: 0.8202
    




    [0.4274521470069885, 0.8202099800109863]



Gördüğünüz gibi, modelimizi her iki formatta da kaydedip yüklemek aynı performansı veriyor.

> 🤔 **Soru:** "loadModel" biçimini mi yoksa "HDF5" biçimini mi kullanmalısınız?

Çoğu kullanım durumu için `SavedModel` formatı yeterli olacaktır. Ancak bu, TensorFlow'a özel bir standarttır. Daha genel amaçlı bir veri standardına ihtiyacınız varsa, "HDF5" daha iyi olabilir.

## En Yanlış Örnekleri Bulma

Daha önce bahsetmiştik ki, modelleme deneylerimizin çoğu, farklı türde modeller kullanmamıza rağmen benzer sonuçlar veriyorsa, verilere geri dönüp bunun neden olabileceğini incelemenin iyi bir fikir olduğundan bahsetmiştik.

Verilerinizi incelemenin en iyi yollarından biri, modelinizin tahminlerini sıralamak ve onun en yanlış yaptığı örnekleri bulmaktır, yani hangi tahminlerin yüksek tahmin olasılığı vardı ama yanlış çıktı.

Bir kez daha, görselleştirme sizin arkadaşınızdır. Görselleştirin, görselleştirin, görselleştirin.

İşleri görsel hale getirmek için, en iyi performans gösteren modelimizin tahmin olasılıklarını ve sınıflarını doğrulama örnekleriyle (metin ve kesin doğruluk etiketleri) birlikte alalım ve bunları bir panda DataFrame'de birleştirelim.

* En iyi modelimiz hala mükemmel değilse, hangi örnekler yanlış gidiyor?
* Hangileri en yanlış?
* Yanlış olan bazı etiketler var mı? Örneğin. model doğru anlıyor ancak temel doğruluk etiketi bunu yansıtmıyor


```python
val_df = pd.DataFrame({"text": val_sentences,
                       "target": val_labels,
                       "pred": model_6_preds,
                       "pred_prob": tf.squeeze(model_6_pred_probs)})
val_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
      <th>pred</th>
      <th>pred_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DFR EP016 Monthly Meltdown - On Dnbheaven 2015...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.168588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FedEx no longer to transport bioterror germs i...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.787065</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gunmen kill four in El Salvador bus attack: Su...</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.986646</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@camilacabello97 Internally and externally scr...</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.184919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Radiation emergency #preparedness starts with ...</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.745240</td>
    </tr>
  </tbody>
</table>
</div>



Şimdi modelimizin yanlış tahminlerini bulalım (burada `target != pred`) ve bunları tahmin olasılıklarına göre sıralayalım (`pred_prob` sütunu).


```python
most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)
most_wrong[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
      <th>pred</th>
      <th>pred_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>? High Skies - Burning Buildings ? http://t.co...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.915444</td>
    </tr>
    <tr>
      <th>759</th>
      <td>FedEx will no longer transport bioterror patho...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.892491</td>
    </tr>
    <tr>
      <th>49</th>
      <td>@madonnamking RSPCA site multiple 7 story high...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.862208</td>
    </tr>
    <tr>
      <th>628</th>
      <td>@noah_anyname That's where the concentration c...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.851035</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Ashes 2015: AustraliaÛªs collapse at Trent Br...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.840667</td>
    </tr>
    <tr>
      <th>393</th>
      <td>@SonofLiberty357 all illuminated by the bright...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.826442</td>
    </tr>
    <tr>
      <th>109</th>
      <td>[55436] 1950 LIONEL TRAINS SMOKE LOCOMOTIVES W...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.825141</td>
    </tr>
    <tr>
      <th>251</th>
      <td>@AshGhebranious civil rights continued in the ...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.813603</td>
    </tr>
    <tr>
      <th>698</th>
      <td>åÈMGN-AFRICAå¨ pin:263789F4 åÈ Correction: Ten...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.791938</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FedEx no longer to transport bioterror germs i...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.787065</td>
    </tr>
  </tbody>
</table>
</div>



Son olarak, örnek metni, doğruluk etiketini, tahmin sınıfını ve tahmin olasılığını görselleştirmek için bazı kodlar yazabiliriz. Örneklerimizi tahmin olasılığına göre sıraladığımız için, `en yanlış` DataFrame'imizin başındaki örneklere bakmak bize yanlış pozitifler gösterecektir.

Bir hatırlatıcı:
* `0` = Gerçek bir felaket Tweet değil
* `1` = Gerçek felaket Tweet


```python
for row in most_wrong[:10].itertuples(): 
  _, text, target, pred, prob = row
  print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
  print(f"Text:\n{text}\n")
  print("----\n")
```

    Target: 0, Pred: 1, Prob: 0.9154438972473145
    Text:
    ? High Skies - Burning Buildings ? http://t.co/uVq41i3Kx2 #nowplaying
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8924906253814697
    Text:
    FedEx will no longer transport bioterror pathogens in wake of anthrax lab mishaps http://t.co/lHpgxc4b8J
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8622081279754639
    Text:
    @madonnamking RSPCA site multiple 7 story high rise buildings next to low density character residential in an area that floods
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8510348796844482
    Text:
    @noah_anyname That's where the concentration camps and mass murder come in. 
     
    EVERY. FUCKING. TIME.
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8406673073768616
    Text:
    Ashes 2015: AustraliaÛªs collapse at Trent Bridge among worst in history: England bundled out Australia for 60 ... http://t.co/t5TrhjUAU0
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8264415860176086
    Text:
    @SonofLiberty357 all illuminated by the brightly burning buildings all around the town!
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.8251414895057678
    Text:
    [55436] 1950 LIONEL TRAINS SMOKE LOCOMOTIVES WITH MAGNE-TRACTION INSTRUCTIONS http://t.co/xEZBs3sq0y http://t.co/C2x0QoKGlY
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.813603401184082
    Text:
    @AshGhebranious civil rights continued in the 60s. And what about trans-generational trauma? if anything we should listen to the Americans.
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.791938304901123
    Text:
    åÈMGN-AFRICAå¨ pin:263789F4 åÈ Correction: Tent Collapse Story: Correction: Tent Collapse story åÈ http://t.co/fDJUYvZMrv @wizkidayo
    
    ----
    
    Target: 0, Pred: 1, Prob: 0.7870645523071289
    Text:
    FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps http://t.co/qZQc8WWwcN via @usatoday
    
    ----
    
    


```python
# En yanlış yanlış negatifleri kontrol edin (model 1 tahmin etmeliyken 0 tahmin etti)
for row in most_wrong[-10:].itertuples():
  _, text, target, pred, prob = row
  print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
  print(f"Text:\n{text}\n")
  print("----\n")
```

    Target: 1, Pred: 0, Prob: 0.05815977230668068
    Text:
    'The way you move is like a full on rainstorm and I'm a house of cards'
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.0545167438685894
    Text:
    @DavidVonderhaar At least you were sincere ??
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.05418562889099121
    Text:
    going to redo my nails and watch behind the scenes of desolation of smaug ayyy
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.05028621107339859
    Text:
    You can never escape me. Bullets don't harm me. Nothing harms me. But I know pain. I know pain. Sometimes I share it. With someone like you.
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.04881070926785469
    Text:
    @willienelson We need help! Horses will die!Please RT &amp; sign petition!Take a stand &amp; be a voice for them! #gilbert23 https://t.co/e8dl1lNCVu
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.04162179306149483
    Text:
    I get to smoke my shit in peace
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.03738167881965637
    Text:
    Why are you deluged with low self-image? Take the quiz: http://t.co/XsPqdOrIqj http://t.co/CQYvFR4UCy
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.036325037479400635
    Text:
    Reddit Will Now QuarantineÛ_ http://t.co/pkUAMXw6pm #onlinecommunities #reddit #amageddon #freespeech #Business http://t.co/PAWvNJ4sAP
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.035058073699474335
    Text:
    @SoonerMagic_ I mean I'm a fan but I don't need a girl sounding off like a damn siren
    
    ----
    
    Target: 1, Pred: 0, Prob: 0.027997875586152077
    Text:
    Ron &amp; Fez - Dave's High School Crush https://t.co/aN3W16c8F6 via @YouTube
    
    ----
    
    

En yanlış örneklerle ilgili ilginç bir şey fark ettiniz mi? Etiketler doğru mu? Geri dönüp olmayan etiketleri düzeltirsek ne olur sizce?

## Test Veri Seti Üzerinde Tahminler Yapmak

Pekala, modelimizin doğrulama setinde nasıl performans gösterdiğini gördük. Peki ya test veri seti?

Test veri seti için etiketlerimiz yok, bu yüzden bazı tahminler yapmamız ve bunları kendimiz incelememiz gerekecek. Test veri setinden rastgele örnekler üzerinde tahminler yapmak için bazı kodlar yazalım ve görselleştirelim.


```python
test_sentences = test_df["text"].to_list()
test_samples = random.sample(test_sentences, 10)
for test_sample in test_samples:
  pred_prob = tf.squeeze(model_6.predict([test_sample]))
  pred = tf.round(pred_prob)
  print(f"Pred: {int(pred)}, Prob: {pred_prob}")
  print(f"Text:\n{test_sample}\n")
  print("----\n")
```

    Pred: 0, Prob: 0.02110372669994831
    Text:
    @noobde this monkey will be a good character in mkx lol banana fatality
    
    ----
    
    Pred: 0, Prob: 0.08776362240314484
    Text:
    genuine Leather man Bag Messenger fit iPad mini 4 tablet case cross body air jp - Full reaÛ_ http://t.co/Vl26HSrq4E http://t.co/ryl0Y88fKM
    
    ----
    
    Pred: 0, Prob: 0.3871268928050995
    Text:
    It was finally demolished in the spring of 2013 and the property has sat vacant since. The justÛ_: saddlebrooke... http://t.co/bd5B5yffyb
    
    ----
    
    Pred: 1, Prob: 0.7170340418815613
    Text:
    Bioterrorism and Ebola. http://t.co/ORIOVftLK4 RT #STOPIslam #TCOT #CCOT #MakeDCListen #TeaParty
    
    ----
    
    Pred: 1, Prob: 0.9093793630599976
    Text:
    Evacuation order lifted for town of Roosevelt - Washington Times http://t.co/Kue48Nmjxh
    
    ----
    
    Pred: 0, Prob: 0.21379774808883667
    Text:
    It's nice out. Guessing the heat wave is over.
    
    ----
    
    Pred: 0, Prob: 0.13666385412216187
    Text:
    @DukeSkywalker @facialabuse you should do a competetion between @xxxmrbootleg &amp; #ClaudioMeloni (ultimate throat penetrator) to a wreck off.
    
    ----
    
    Pred: 0, Prob: 0.3559725284576416
    Text:
    CommoditiesåÊAre Crashing Like It's 2008 All Over Again http://t.co/EM1cN7alGk
    
    ----
    
    Pred: 0, Prob: 0.1974869817495346
    Text:
    Serial arsonist gets no bail not jail release http://t.co/rozs6aumsS
    
    ----
    
    Pred: 0, Prob: 0.13319750130176544
    Text:
    Choking Hazard Prompts Recall Of Kraft Cheese Singles http://t.co/98nOsYzu58
    
    ----
    
    

Modelinizin görünmeyen veriler üzerinde nasıl performans gösterdiğine ve ardından gerçek testte nasıl performans gösterebileceğine bir göz atmak için bu tür görselleştirme kontrollerini mümkün olduğunca sık yapmak önemlidir.

## Hız/Puan Dengesi

Yapacağımız son testlerden biri, en iyi modelimiz ve temel modelimiz arasındaki hız/puan dengelerini bulmaktır.

Bu neden önemli?

Deneme yoluyla bulduğunuz en iyi performans gösteren modeli seçmek cazip gelse de, bu model aslında bir üretim ortamında çalışmayabilir.

Bu şekilde ifade edin, Twitter olduğunuzu ve saatte 1 milyon Tweet aldığınızı hayal edin (bu uydurma bir sayıdır, gerçek sayı çok daha yüksektir). Ve Tweet'leri okumak ve bir felaketle ilgili ayrıntıları gerçek zamanlıya yakın bir şekilde yetkilileri uyarmak için bir felaket algılama sistemi oluşturmaya çalışıyorsunuz.

İşlem gücü ücretsiz değildir, bu nedenle proje için tek bir işlem makinesiyle sınırlısınız. Bu makinede, modellerinizden biri %80 doğrulukla saniyede 10.000 tahminde bulunurken, modellerinizden biri (daha büyük bir model) %85 doğrulukla saniyede 100 tahmin yapar.

Hangi modeli seçersiniz?

İkinci modelin performans artışı, ekstra kapasiteyi kaçırmaya değer mi? Tabii ki burada deneyebileceğiniz birçok seçenek var, ilk modele mümkün olduğunca çok Tweet göndermek ve ardından modelin en az emin olduğu şeyleri ikinci modele göndermek gibi.

Buradaki amaç, deney yoluyla bulduğunuz en iyi modeli göstermektir, üretimde kullandığınız model olmayabilir.

Bunu daha somut hale getirmek için, bir model ve bir dizi örnek alacak bir fonksiyon yazalım.


```python
import time
def pred_timer(model, samples):
  start_time = time.perf_counter() 
  model.predict(samples) 
  end_time = time.perf_counter() 
  total_time = end_time-start_time 
  time_per_pred = total_time/len(val_sentences)
  return total_time, time_per_pred
```

İyi görünüyor!

Şimdi en iyi performans gösteren modelimizin (`model_6`) ve temel modelimizin (`model_0`) tahmin sürelerini değerlendirmek için `pred_timer()` fonksiyonumuzu kullanalım.


```python
model_6_total_pred_time, model_6_time_per_pred = pred_timer(model_6, val_sentences)
model_6_total_pred_time, model_6_time_per_pred
```




    (0.3335412910000173, 0.0004377182296588153)




```python
baseline_total_pred_time, baseline_time_per_pred = pred_timer(model_0, val_sentences)
baseline_total_pred_time, baseline_time_per_pred
```




    (0.023066670999980943, 3.0271221784751892e-05)



Mevcut donanımımızla (benim durumumda bir Google Colab not defteri kullanıyorum) en iyi performans gösteren modelimiz, temel modelimiz olarak tahminler yapmak için 10 kat daha fazla zaman alıyor. Bu ekstra tahmin süresi buna değer mi?

Modelimizin F1 puanlarıyla tahmin başına süreyi karşılaştıralım.


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(baseline_time_per_pred, baseline_results["f1"], label="baseline")
plt.scatter(model_6_time_per_pred, model_6_results["f1"], label="tf_hub_sentence_encoder")
plt.legend()
plt.title("F1-score versus time per prediction")
plt.xlabel("Time per prediction")
plt.ylabel("F1-Score");
```


    
![png](8-NLP%27ye%20Giris_files/8-NLP%27ye%20Giris_207_0.png)
    


Elbette, bu noktaların her biri için ideal konum, grafiğin sol üst köşesinde olmaktır (tahmin başına düşük süre, yüksek F1 puanı).

Bizim durumumuzda, tahmin ve performans başına süre için açık bir fark var. En iyi performans gösteren modelimiz, tahmin başına bir büyüklük sırası daha uzun sürüyor, ancak yalnızca birkaç F1 puanı artışıyla sonuçlanıyor.

Bu tür bir fark, makine öğrenimi modellerini kendi uygulamalarınıza dahil ederken aklınızda bulundurmanız gereken bir şeydir.
