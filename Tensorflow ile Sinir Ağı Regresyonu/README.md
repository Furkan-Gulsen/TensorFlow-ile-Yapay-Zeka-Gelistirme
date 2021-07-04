Bu konuya başlamadan önce aşağıda bulunan kavramlar çok iyi bir şekilde anlaşılması gerekiyor.

## Regresyon Analizi Nedir?

**Regresyon analizi, iki ya da daha çok nicel değişken arasındaki ilişkiyi ölçmek için kullanılan analiz metodudur.** Eğer tek bir değişken kullanılarak analiz yapılıyorsa buna tek değişkenli regresyon, birden çok değişken kullanılıyorsa çok değişkenli regresyon analizi olarak isimlendirilir. Regresyon analizi ile değişkenler arasındaki ilişkinin varlığı, eğer ilişki var ise bunun gücü hakkında bilgi edinilebilir. 

Örneğin; <br>
Bir daire satın almayı düşünüyorsunuz. Bu dairenin fiyatını belirleyen birden çok unsur vardır. Örnek:
- Toplu taşımaya yakınlığı
- Manzarası
- Kaçıncı katta olduğu
- Binanın yapım yılı

gibi benzeri bir çok unsur binanın fiyatını belirleyen önemli faktörlerdir. 

Regresyonda, değişkenlerden biri bağımlı diğerleri bağımsız değişken olmalıdır. 


## Yapay Sinir Ağı Nedir?

Yapay sinir ağları (YSA), insan beyninin bilgi işleme tekniğinden esinlenerek geliştirilmiş bir bilgi işlem teknolojisidir. YSA ile basit biyolojik sinir sisteminin çalışma şekli taklit edilir. Yani biyolojik nöron hücrelerinin ve bu hücrelerin birbirleri ile arasında kurduğu sinaptik bağın dijital olarak modellenmesidir.

Nöronlar çeşitli şekillerde birbirlerine bağlanarak ağlar oluştururlar. Bu ağlar öğrenme, hafızaya alma ve veriler arasındaki ilişkiyi ortaya çıkarma kapasitesine sahiptirler. Diğer bir ifadeyle, YSA'lar, normalde bir insanın düşünme ve gözlemlemeye yönelik doğal yeteneklerini gerektiren problemlere çözüm üretmektedir. Bir insanın, düşünme ve gözlemleme yeteneklerini gerektiren problemlere yönelik çözümler üretebilmesinin temel sebebi ise insan beyninin ve dolayısıyla insanın sahip olduğu yaşayarak veya deneyerek öğrenme yeteneğidir.

Yapay Sinir Ağlarının Avantajları
- Yapay Sinir Ağları bir çok hücreden meydana gelir ve bu hücreler eş zamanlı çalışarak karmaşık işleri gerçekleştirir.
- Öğrenme kabiliyeti vardır ve farklı öğrenme algoritmalarıyla öğrenebilirler.
- Görülmemiş çıktılar için sonuç (bilgi) üretebilirler. Gözetimsiz öğrenim söz konusudur.
- Örüntü tanıma ve sınıflandırma yapabilirler. Eksik örüntüleri tamamlayabilirler.
- Hata toleransına sahiptirler. Eksik veya belirsiz bilgiyle çalışabilirler. Hatalı durumlarda dereceli bozulma (graceful degradation) gösterirler.
- Paralel çalışabilmekte ve gerçek zamanlı bilgiyi işleyebilmektedirler.

### Yapay Sinir Ağlarının Sınıflandırılması

#### Tek Katmanlı Yapay Sinir Ağları
Tek katmanlı yapay sinir ağları sadece girdi ve çıktı katmanlarından oluşur. Çıktı üniteleri bütün girdi ünitelerine (X) bağlanmaktadır ve her bağlantının bir ağırlığı (W) vardır. 

#### Çok Katmanlı Yapay Sinir Ağları

Doğrusal olmayan problemlerin çözümü için uygun bir ağ yapısıdır. Bu sebeple daha karışık problemlerin çözümünde kullanılır. Karışık problemlerin modeli olması ağın eğitimini zorlaştıran bir yapıya bürünmesine sebep olur. Tek katmanlı ağ yapısına göre daha karmaşık bir yapıdadır. Fakat problem çözümlerinde genellikle çok katmanlı ağ yapısı kullanılır çünkü tek katmanlı yapılara göre daha başarılı sonuçlar verir. Çok katmanlı yapay sinir ağları modellerinde en az 1 adet gizli katman bulunur.

### Yapay Sinir Hücresi

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Neuron_Hand-tuned.svg/1200px-Neuron_Hand-tuned.svg.png" />

Canlılardaki sinir hücrelerinin biyolojik görünümü yukarıda gördüğümüz şekildeki gibidir. Çekirdeğimiz var ve bir akson boyunca iletim yapılıyor. Burada çıkış terminallerinde dentrit uclarından elde edilen sensör verilerimiz çekirdekte ağırlandırılarak akson boyunca iletiliyor ve başka sinir hücresine bağlanıyor. Bu şekilde sinirler arası iletişim sağlanmış oluyor.

İnsandaki bir sinir hücresinin matematiksel modeli ise şu şekilde gösterilebilir:

<img src="https://mesutpiskin.com/blog/wp-content/uploads/2017/08/ysa_matematiksel_modeli.png" />

Dentrit dediğimiz yollar boyunca ağırlıklarımız mevcut ve bu dentritlere giren bir başka nörondan da gelmiş olabilecek bir giriş değerimiz (x0 ) var. Giriş değerimiz ve dentritteki ağırlığımız(w0) çarpıldıktan sonra( w0x0)  sinir hücresine iletilir ve sinir hücresinde bu çarpma işlemi yapılıyor ve tüm dentritlerden gelen ağırlık ile giriş çarpımları toplanır. Yani ağırlıklı toplama işlemi yapılır. Ardından bir bias(b) ile toplandıktan sonra aktivasyon fonksiyonu ardından çıkışa aktarılır. Bu çıkış nihai çıkış olabileceği gibi bir başka hücrenin girişi olabilir. Matematiksel olarak ağırlıklar ile girişler çarpılır artı bir bias eklenir. Böylelikle basit bir matematiksel model elde edilir.

Yapay Sinir Ağlarında yapılan temel işlem; modelin en iyi skoru vereceği w(ağırlık parametresi) ve b(bias değeri) parametrelerinin hesabını yapmaktır.                     

Her bir sinir hücresi aynı şekilde hesaplanır ve bunlar birbirine seri ya da paralel şekilde bağlanır.

Bir yapay sinir hücresi beş bölümden oluşmaktadır;

1. **Girdiler:**<br> Girdiler nöronlara gelen verilerdir. Bu girdilerden gelen veriler biyolojik sinir hücrelerinde olduğu gibi toplanmak üzere nöron çekirdeğine gönderilir.

2. **Ağırlıklar:**<br> Yapay sinir hücresine gelen bilgiler girdiler üzerinden çekirdeğe ulaşmadan önce geldikleri bağlantıların ağırlığıyla çarpılarak çekirdeğe iletilir. Bu sayede girdilerin üretilecek çıktı üzerindeki etkisi ayarlanabilinmektedir.

3. **Toplama Fonksiyonu (Birleştirme Fonksiyonu):**<br> Toplama fonksiyonu bir yapay sinir hücresine ağırlıklarla çarpılarak gelen girdileri toplayarak o hücrenin net girdisini hesaplayan bir fonksiyondur

4. **Aktivasyon fonksiyonu:**<br> Önceki katmandaki tüm girdilerin ağırlıklı toplamını alan ve daha sonra bir çıkış değeri (tipik olarak doğrusal olmayan) üreten ve bir sonraki katmana geçiren bir fonksiyondur. (örneğin, ReLU veya sigmoid ).

5. **Çıktılar:**<br> Aktivasyon fonksiyonundan çıkan değer hücrenin çıktı değeri olmaktadır. Her hücrenin birden fazla girdisi olmasına rağmen bir tek çıktısı olmaktadır. Bu çıktı istenilen sayıda hücreye bağlanabilir.

### Yapay Sinir Ağlarını Bağlantılarına Göre Sınıflandırma

Yapay sinir ağları kendi arasında bağlantılar içerir. Bunlar ileri beslemeli ve geri beslemeli ağlar olarak sınıflandırılırlar.

#### İleri Beslemeli Ağlar

<img src="https://i.hizliresim.com/qLLIIK.png" />

- Tek yönlü bilgi akışı söz konusudur.
- Bu ağ modelinde Girdi tabakasından alınan bilgiler Gizli katmana iletilir.
- Gizli ve Çıktı tabakalarından bilginin işlenmesi ile çıkış değeri belirlenir.

#### Geri Beslemeli Ağlar

<img src="https://www.derinogrenme.com/wp-content/uploads/2017/02/gsa.png"/>

- Bir geri beslemeli sinir ağı, çıkış ve ara katlardaki çıkışların, giriş birimlerine veya önceki ara katmanlara geri beslendiği bir ağ yapısıdır. Böylece, girişler hem ileri yönde hem de geri yönde aktarılmış olur.
- Bu çeşit YSA’ların dinamik hafızaları vardır ve bir andaki çıkış hem o andaki hem de önceki girişleri yansıtır. Bundan dolayı, özellikle önceden tahmin uygulamaları için uygundurlar.


## Input ve Output Değeri
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# değerleri oluşturma
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# etiketleri oluşturma
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# hadi şimdi görselleştirelim
plt.scatter(X, y);
```
> <img src="https://i.ibb.co/M9y3mmD/chapter1-img1.png" />

Yukarıda ki modellemede X ve y arasında ki matematiksel örüntüyü hesaplayabilir misiniz?

Örneğin; X'e 40 değerini verirsek y değeri ne olur ? Ya da y değerini 30 yapan X değeri nedir?

Bunu neden elle (klasik) hesaplayalım. Bu sadece 2 değişkeni olan basit bir yapı. Ya 100 değişkeni olsaydı? O zamanda elle hesaplayabiliriz cümlesini kurabilirmiydik? Tabiki de HAYIR. O zaman bunun için neden bir sinir ağı eğitmiyoruz? Hadi başlayalım...

```python
""" 
Input  : Modele giren verilerimizin şekli
Output : Modelimizden çıkmasını istediğimiz verilerin şekli 
Bunlar, üzerinde çalıştığımız soruna bağlı olarak farklılık gösterecektir.
Sinir ağları tensor (veya dizi) olarak temsilsil edilir.
"""

# Örnek bir tensör oluşturalım
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
house_info, house_price
```
> (<tf.Tensor: shape=(3,), dtype=string, numpy=array([b'bedroom', b'bathroom', b'garage'], dtype=object)>,
 <tf.Tensor: shape=(1,), dtype=int32, numpy=array([939700], dtype=int32)>)

```python
house_info.shape
```
> TensorShape([3])

```python
# Yukarıdaki örneği numpy kütüphanesi ile yapmıştık. Bunu tensör yapısına çevirelim
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
plt.scatter(X, y);
```
> <img src="https://i.ibb.co/M9y3mmD/chapter1-img1.png" />

Tensörde oluşturmayı hatırladığımıza göre şimdi yukarıda bahsettiğimiz X ve y 
sorununa dair bir sinir ağı eğitelim. 

Amacımız kısaca; verilen X değerine göre y değerini bulmak. Burada input değerimiz X, output değerimiz y'dir. 

```python
# Tek bir X örneğinin shape değeri
input_shape = X[0].shape

# Tek bir y örneğinin shape değeri
output_shape = y[0].shape

# bunların ikisi de skalerdir (shape değeri yoktur)
input_shape, output_shape
```
> (TensorShape([]), TensorShape([]))

Neden input ve output değerlimizin bir şekli yok.

Bunun nedeni, modelimize ne tür veriler ilettiğimiz önemli değil, her zaman 
input olarak alacak ve output olarak bir tür tensör olarak geri dönecektir.

Ancak bizim durumumuzda veri kümemiz nedeniyle (sadece 2 küçük sayı listesi), 
özel bir tür tensöre bakıyoruz, daha spesifik olarak bir 
rank'ı 0 tensör veya bir skaler.

```python
X[0], y[0]
```
> (<tf.Tensor: shape=(), dtype=float32, numpy=-7.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=3.0>)
 
- X[0] = -7.0
- y[0] = 3.0
Bir y değerini tahmin etmek için bir X değeri kullanıyoruz. Bu bizim modelimizin en basit denklemi. 

Burada anlatmak istediğim nokta; bir input değeri ile output değeri kullanmaktır. Klasik programlama da bir input ve bir fonksiyon kullanarak bir output değeri elde edersiniz. Bizim sinir ağlarında (yada ML) yapmak istediğimiz şey; bir input ve bir output değeri vererek arada ki fonksiyonunu kendi üretmesini istiyoruz.

<img src="https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/01-input-and-output-shapes-housing-prices.png" />

*Konut fiyatlarını tahmin etmek için bir makine öğrenimi algoritması oluşturmaya çalışıyorsanız, girdileriniz yatak odası sayısı, banyo sayısı ve garaj sayısı olabilir ve size 3 (3 farklı özellik) girdi şekli verir. Ve evin fiyatını tahmin etmeye çalıştığınız için çıktı şekliniz 1 olur.*

## Modellemedeki Adımlar

Artık elimizde hangi verilere, girdi ve çıktı şekillerine sahip olduğumuzu biliyoruz, onu modellemek için nasıl bir sinir ağı kuracağımıza bakalım.

TensorFlow'da bir model oluşturmak ve eğitmek için tipik olarak 3 temel adım vardır.

- **Bir model oluşturma**<br>
Bir sinir ağının katmanlarını kendiniz bir araya getirin ([Functional](https://www.tensorflow.org/guide/keras/functional) veya [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)'yi kullanarak) veya önceden oluşturulmuş bir modeli içe aktarın (transfer learning olarak bilinir). 
- **Model derleme**<br>
Bir model performansının nasıl ölçüleceğini (kayıp metrikler) tanımlamanın yanı sıra nasıl iyileştirileceğini (optimize) tanımlama. 
- **Model uydurma**<br>
Modelin verilerdeki kalıpları bulmaya çalışmasına izin vermek (X, y'ye nasıl ulaşır). 

Regresyon verilerimiz için bir model oluşturmak üzere [Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)'sini kullanarak bunları çalışırken görelim. Ve sonra her birinin üzerinden geçeceğiz.

```python
"""
Biz her modeli çalıştırdığımızda model, belirli bir metolojide çalışacak.
Ama burada şu sıkıntı var: Modeli her run ettiğimizde farklı bir sonuç alacağız.
İşte burada tf.random.set_seed(number) kullanarak o rastgeleliği belirli bir
yolla bağlamış oluyoruz. Bu sayede her run ettiğimizde aynı sonucu alacağız.
"""
tf.random.set_seed(42)

# Sequential API'yi kullanarak bir model oluşturun
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Modeli derleme
model.compile(loss=tf.keras.losses.mae, # mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # stochastic gradient descent
              metrics=["mae"])

# modeli fit etme
model.fit(X, y, epochs=5)
```

İşte bu kadar basit :) 

X'e bağlı bir y değeri oluşturan bir modeli geliştirdik.

```python
# X ve y değerlerini kontrol edelim
X, y
```
> (<tf.Tensor: shape=(8,), dtype=float32, numpy=array([-7., -4., -1.,  2.,  5.,  8., 11., 14.], dtype=float32)>,
 <tf.Tensor: shape=(8,), dtype=float32, numpy=array([ 3.,  6.,  9., 12., 15., 18., 21., 24.], dtype=float32)>)

```python
# Var olan bir X değeri ile modelimiz doğru bir y değeri üretecek mi?
model.predict([8.0])
```
Ama ama bu niye böyle oldu :(  Her şeyi doğru yaptık gibi. Modele bir input ve output değeri verdik. Bunu bir sinir ağına bağladık. Fakat doğru sonuçla alakası bile olmayan bir output değeri verdi bize. 

> Bu soruya cevap vermeden önce size kısa bir soru sormak istiyorum. TensorFlow içerisinde hep Keras dediğimiz yapıları görüyoruz. Bu keras nedir? [Cevap](https://i.ibb.co/LNScsJd/cevap1.png)


## Bir Model Geliştirmek

Model istediğimiz sonucu vermeyi bırakın, yakınına dahi yanaşamadı. Peki burada çözüm ne?

Doğru tahmin ettiniz. Fine tuning yani ince ayar yapmak. Modeli geliştirmek için hangi adımları uyguladık:

1. **Model oluşturma**<br>
Burada daha fazla katman eklemek, her katmandaki gizli birimlerin (nöronlar olarak da adlandırılır) sayısını artırmak, her katmanın etkinleştirme işlevlerini değiştirmek isteyebilirsiniz.
2. **Model derleme**<br>
Optimizasyon fonksiyonunu seçmek veya belki de optimizasyon fonksiyonunun öğrenme oranını değiştirmek isteyebilirsiniz.
3. **Modeli fit etme**<br>
Daha fazla epoch veya daha fazla veri ile daha iyi sonuçlar almak isteyebilirsiniz.

Vay. Az önce bir dizi olası adımı tanıttık. Hatırlanması gereken önemli şey, bunların her birini nasıl değiştireceğiniz, üzerinde çalıştığınız soruna bağlı olacaktır.

```python
# yukarıda bu yapıyı ayrıntısıyla anlattım
tf.random.set_seed(42)

# bir önceki modelin aynısını uygulayalım
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# modeli aynı şekilde derleyelim
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# şimdi fit edelim, ama bu sefer 100 epoch kullanarak
model.fit(X, y, epochs=100)
```
Yukarı da ki kod bloğunu çalıştırdığınızda MAE değerinin yani kayıp fonksiyonun adım adım düştüğünü gözlemleyeceksiniz. İşte bu anda geriye yaslanıp işlemin bitmesini bekleyebilirsiniz çünkü bu doğru gittiğinizi gösteriyor.

Yukarıda 5 epoch ile eğittimizde hiç güzel bir tahmin değeri almadık. Peki ya şimdi?

```python
model.predict([8.0])
```
İşte buuu 💪 0.33 lük bir sapma var ama modelimiz resmen input ve output değerlerini ile doğru eğitilmiş.

Şimdi de modelimizde olmayan bir sayı ile deneme yapalım. 
Modelimize tekrar göz atalım. Test etmek için arada ki bağlantıyı bulmamız gerekiyor.
```python
X, y
```
> (<tf.Tensor: shape=(8,), dtype=float32, numpy=array([-7., -4., -1.,  2.,  5.,  8., 11., 14.], dtype=float32)>,
 <tf.Tensor: shape=(8,), dtype=float32, numpy=array([ 3.,  6.,  9., 12., 15., 18., 21., 24.], dtype=float32)>)
 
- -7 --> 3
- -4 --> 6
- -1 --> 9

Yukarıda ki eşitliklere baktığımızda x + 10 = y gibi bir eşitiğin olduğunu hemen anlayabiliriz. Denklemi çıkardığımıza göre şimdi dizide olmayan bir sayıda nasıl performans gösterdiğini gözlemleyebiliriz.

```python
model.predict([20]) # cevabın 20+10= 30 olmasını bekliyoruz
```
Hımm. Yaklaşık bir sonuç ama tam da istediğimiz bir cevap değil. Modelimizi bir değerlendirelim daha sonra nasıl daha iyi sonuç alacağımızı düşünürüz.
 
 
## Modeli Değerlendirme

Sinir ağı oluştururken takip edilen tipik bir akış var:
```
Bir model yarat -> Onu değerlendir -> Bir model yarat -> Onu değerlendir -> Bir model yarat -> Onu değerlendir ...
```
Fine tuning (ince ayarlama), sıfırdan bir model oluşturmak değil, mevcut bir model üzerinde ayarlamalar yapmaktır.

Modeli değerlendirirken yapılacak en güzel davranışlardan bazıları şunlardır:
- Görselleştirin
- Görselleştirin
- Görselleştirin

Lütfen modeli görselleştirin. Görselleştirmeniz gereken bazı fikirler:
- Veriler - hangi verilerle çalışıyorsunuz? Nasıl görünüyor?
- Modelin kendisi - mimari neye benziyor? Farklı şekiller nelerdir?
- Bir modelin eğitimi - bir model öğrenirken nasıl performans gösterir?
- Bir modelin tahminleri - bir modelin tahminleri temel gerçeğe (orijinal etiketler) karşı nasıl sıralanır?

Görselleştirmeyi 1 adım sonraya erteliyoruz çünkü yukarıda eğittiğimiz model tam da istediğimiz sonucu vermedi. Bu yüzden yukarıda ki modeli daha fazla veri ile tekrar eğitmek iyi olabilir.

```python
# Daha büyük bir veriseti yaratma
X = np.arange(-100, 100, 4)
X
```
> array([-100,  -96,  -92,  -88,  -84,  -80,  -76,  -72,  -68,  -64,  -60,
        -56,  -52,  -48,  -44,  -40,  -36,  -32,  -28,  -24,  -20,  -16,
        -12,   -8,   -4,    0,    4,    8,   12,   16,   20,   24,   28,
         32,   36,   40,   44,   48,   52,   56,   60,   64,   68,   72,
         76,   80,   84,   88,   92,   96])

```python
# şimdi de etiketlerini oluşturalım
y = np.arange(-90, 110, 4)
y
```
> array([-90, -86, -82, -78, -74, -70, -66, -62, -58, -54, -50, -46, -42,
       -38, -34, -30, -26, -22, -18, -14, -10,  -6,  -2,   2,   6,  10,
        14,  18,  22,  26,  30,  34,  38,  42,  46,  50,  54,  58,  62,
        66,  70,  74,  78,  82,  86,  90,  94,  98, 102, 106])

`x + 10 = y` eşitliği sağladık gibi duruyor. 

### Verileri Train ve Test Olarak Ayırma

Bir makine öğrenimi projesindeki diğer en yaygın ve önemli adımlardan biri, bir eğitim ve test seti (ve gerektiğinde bir doğrulama seti) oluşturmaktır.

Her set belirli bir amaca hizmet eder:

- **Eğitim seti**<br>
Model, genellikle mevcut toplam verilerin (epoch boyunca çalıştığınız ders materyalleri gibi) %70-80'i olan bu verilerden öğrenir.
- **Doğrulama seti**<br> 
Model, genellikle mevcut toplam verilerin %10-15'i olan bu verilere göre ayarlanır (final sınavından önce girdiğiniz alıştırma sınavı gibi).
- **Test seti**<br>
Model, öğrendiklerini test etmek için bu veriler üzerinde değerlendirilir, genellikle mevcut toplam verilerin %10-15'i kadardır (dönem sonunda girdiğiniz final sınavı gibi).

Şimdilik sadece bir eğitim ve test seti kullanacağız, bu, modelimizin öğrenilmesi ve değerlendirilmesi için bir veri setimiz olacağı anlamına geliyor.

X ve y dizilerimizi bölerek bunları oluşturabiliriz.

> 🔑 Not: Gerçek dünya verileriyle uğraşırken, bu adım tipik olarak bir projenin hemen başlangıcında yapılır (test seti her zaman diğer tüm verilerden ayrı tutulmalıdır). Modelimizin eğitim verilerini öğrenmesini ve ardından görünmeyen örneklere ne kadar iyi genelleştiğine dair bir gösterge elde etmek için test verileri üzerinde değerlendirmesini istiyoruz.

```python
# verisetimizin büyüklüğüne bakalım
len(X)
```
> 50


```python
# verileri train ve test olarak ayıralım
X_train = X[:40] # verilerin %80'ine denk geliyor
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

len(X_train), len(X_test)
```
> (40, 10)


### Verileri Görselleştirme

Artık eğitim ve test verilerimiz var, artık bunu görselleştirmek iyi bir fikir.

Neyin ne olduğunu ayırt etmek için güzel renklerle çizelim.

```python
plt.figure(figsize=(10, 7))
# train verileri mavi olsun
plt.scatter(X_train, y_train, c='b', label='Training data')
# test verileri yeşil olsun
plt.scatter(X_test, y_test, c='g', label='Testing data')
plt.legend();
```
> <img src="https://i.ibb.co/xDL5jBD/indir.png" />

Güzel! Verilerinizi, modelinizi, herhangi bir şeyi görselleştirebildiğiniz her an, bu iyi bir fikirdir.

Bu grafiği göz önünde bulundurarak, yeşil noktaları (X_test) çizmek için mavi noktalardaki (X_train) deseni öğrenen bir model oluşturmaya çalışacağız.

```python
tf.random.set_seed(42)

# bir model yaratma
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# modeli derleme
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# modeli fit etme
# model.fit(X_train, y_train, epochs=100) # commented out on purpose (not fitting it just yet)
```

### Modeli Görselleştirme

Bir model oluşturduktan sonra, ona bir göz atmak isteyebilirsiniz (özellikle daha önce çok model oluşturmadıysanız).

Modelinizin katmanlarını ve şekillerini, üzerinde Summary()'i arayarak inceleyebilirsiniz.

🔑 Not: Bir modeli görselleştirmek, özellikle girdi ve çıktı şekli uyumsuzluklarıyla karşılaştığınızda faydalıdır.

```python
# çalışmayacak (modeli fit etmedik)
model.summary()
```
> ValueError

Sizce yukarıda ki hatanın sebebi modeli fit etmememiz mi? Hımm. Hata mesajını okuduğumuzda `input_shape` değerinin olmadığını söylüyor. 

`Input_shape` değeri ilk katmana girilir. Şimdi deneyelim ve bakalım hata gideriliyor mu?


```python
tf.random.set_seed(42)

# bir model yaratma
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1])
])

# modeli derleme
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# modeli fit etme
# model.fit(X_train, y_train, epochs=100) # commented out on purpose (not fitting it just yet)
```

```python
model.summary()
```
> <img src="https://i.ibb.co/8cZRccR/Ekran-g-r-nt-s-2021-07-04-120440.png" />

Modelimizde `summary()` işlevini çağırmak bize içerdiği katmanları, çıktı şeklini ve parametre sayısını gösterir.

- **Toplam parametreler**<br>
Modeldeki toplam parametre sayısı.
- **Eğitilebilir parametreler**<br>
Bunlar, modelin eğitirken güncelleyebileceği parametrelerdir (kalıplardır).
- **Eğitilemez parametreler**<br>
Bu parametreler eğitim sırasında güncellenmez (bu, transfer learninig sırasında diğer modellerden önceden öğrenilmiş kalıpları getirdiğinizde tipiktir).

> 📖 Kaynak: Bir katmandaki eğitilebilir parametrelere daha derinlemesine bir genel bakış için [MIT'nin derin öğrenme videosuna](https://www.youtube.com/watch?v=njKP3FqW3Sk) girişine göz atın.

> 🛠 Alıştırma: Dense katmandaki gizli birimlerin sayısıyla oynamayı deneyin (örn. `Dense(2)`, `Dense(3)`). Bu, Toplam/Eğitilebilir parametreleri nasıl değiştirir? Değişikliğe neyin sebep olduğunu araştırın.

Şimdilik, bu parametreler hakkında düşünmeniz gereken tek şey, bunların verilerdeki öğrenilebilir kalıplarıdır.

Modelimizi eğitim verileriyle fir edelim şimdi.


```python
# modeli eğitim verileriyle fit etme
model.fit(X_train, y_train, epochs=100, verbose=0)
```

Özetin yanı sıra plot_model() kullanarak modelin 2D grafiğini de görüntüleyebilirsiniz.

```python
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)
```
> <img src="https://i.ibb.co/8cZRccR/Ekran-g-r-nt-s-2021-07-04-120440.png" />

Bizim durumumuzda, kullandığımız modelin yalnızca bir girdisi ve bir çıktısı var, ancak daha karmaşık modelleri görselleştirmek hata ayıklama için çok yardımcı olabilir.

### Tahminleri Görselleştirme

Şimdi eğitilmiş bir modelimiz var, hadi bazı tahminleri görselleştirelim.

Tahminleri görselleştirmek için, onları temel gerçek etiketlerine göre planlamak her zaman iyi bir fikirdir.

Bunu genellikle y_test ve y_pred (gerçek ve tahminler) şeklinde görürsünüz.

İlk olarak, test verileri (X_test) üzerinde bazı tahminler yapacağız, modelin test verilerini hiç görmediğini unutmayın.


```python
# modeli predict edelim (X_test verileri ile)
y_preds = model.predict(X_test)

# tahminleri görelim
y_preds
```
> array([[53.57109 ],
       [57.05633 ],
       [60.541573],
       [64.02681 ],
       [67.512054],
       [70.99729 ],
       [74.48254 ],
       [77.96777 ],
       [81.45301 ],
       [84.938255]], dtype=float32)

Bunları gerçek değerler ile karşılaştırıp modelin doğruluğunu anlamak için bir fonksiyon yaratalım:

```python
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):
  """
  Eğitim verilerini, test verilerini görselleştirir ve tahminleri karşılaştırır.
  """
  plt.figure(figsize=(10, 7))
  # train verileri mavi olsun
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # test verileri yeşil olsun
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # tahmin değerleri kırmızı olsun
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  plt.legend();
  
plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)
```
> <img src="https://i.ibb.co/Vv70dWN/indir-2.png" />

### Tahminleri Değerlendirme

Görselleştirmelerin yanı sıra değerlendirme metrikleri, modelinizi değerlendirmek için alternatif en iyi seçeneğinizdir.

Üzerinde çalıştığınız soruna bağlı olarak, farklı modellerin farklı değerlendirme ölçütleri vardır.

Regresyon problemleri için kullanılan ana metriklerden ikisi şunlardır:

- **Mean absolute error (MAE)**<br>
Tahminlerin her biri arasındaki ortalama fark.
- **Mean squared error (MSE)**<br>
Tahminler arasındaki kare ortalama fark.

Bu değerlerin her biri ne kadar düşükse, o kadar iyidir.

Ayrıca, derleme adımı sırasında herhangi bir ölçüm ayarının yanı sıra modelin kaybını döndürecek olan `model.evaluate()` öğesini de kullanabilirsiniz.

```python
model.evaluate(X_test, y_test)
```
> [18.74532699584961, 18.74532699584961]

Biz MAE(`metrics=['MAE']`) değerini kullandığımız için evaluate fonksiyonu bize MAE değerini döndürecektir.

TensorFlow'da ayrıca MSE ve MAE için ayrı olarak fonksiyonlar vardır. Bunlar değerlendirme için ayrıca kullanılabilir.


```python
# MAE değerini fonksiyon ile hesaplama
mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=y_preds)
mae
```
> <tf.Tensor: shape=(10,), dtype=float32, numpy=
array([34.42891 , 30.943668, 27.45843 , 23.97319 , 20.487946, 17.202168,
       14.510478, 12.419336, 11.018796, 10.212349], dtype=float32)>

Aaa. Neden bir çıktı yerine on farklı çıktı aldık?

Bunun nedeni, y_test ve y_preds tensorlerinin farklı şekillerde olmasından kaynaklanıyor.

```python
# y etiket tensorünü kontrol edelim
y_test
```
> array([ 70,  74,  78,  82,  86,  90,  94,  98, 102, 106])

```python
# tahminleri kontrol edelim
y_preds
```
> array([[53.57109 ],
       [57.05633 ],
       [60.541573],
       [64.02681 ],
       [67.512054],
       [70.99729 ],
       [74.48254 ],
       [77.96777 ],
       [81.45301 ],
       [84.938255]], dtype=float32)

```python
# tesorlerin şekillerini kontrol edelim
y_test.shape, y_preds.shape
```
> ((10,), (10, 1))

Hatırlarsanız en başta Input ve Outpu değerlerini konuşmuşduk. Ve o sorun geldi çattı. Bu değerlerin aynı şekillere sahip olmasoı gerekiyor yoksa değerlendirmemiz imkansız.

`squeeze()` kullanarak bunu düzeltebiliriz, 1 boyutunu y_preds tensörümüzden kaldıracak ve onu y_test ile aynı şekle getirecektir.

```python
# squeeze() kullanmadan önce
y_preds.shape
```
> (10, 1)


```python
# squeeze() kullandıktan sonra
y_preds.squeeze().shape
```
> (10, )

```python
# verilere  ayrıntılı bakalım
y_test, y_preds.squeeze()
```
> (array([ 70,  74,  78,  82,  86,  90,  94,  98, 102, 106]),
 array([53.57109 , 57.05633 , 60.541573, 64.02681 , 67.512054, 70.99729 ,
        74.48254 , 77.96777 , 81.45301 , 84.938255], dtype=float32))

Tamamdır, şimdi y_test ve y_preds tensorlerımızı nasıl aynı şekle getireceğimizi biliyoruz, hadi değerlendirme metriklerimizi kullanalım.

```python
# MAE değerini hesaplama
mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=y_preds.squeeze())

# MSE değerini hesaplama
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=y_preds.squeeze())
mse, mae
```
> (<tf.Tensor: shape=(), dtype=float32, numpy=18.745327>, <tf.Tensor: shape=(), dtype=float32, numpy=353.57336> )

MAE'yi saf TensorFlow işlevlerini kullanarak da hesaplayabiliriz.

```python
tf.reduce_mean(tf.abs(y_test-y_preds.squeeze()))
```
> <tf.Tensor: shape=(), dtype=float64, numpy=18.745327377319335>

Yine, tekrar kullanabileceğinizi (veya kendinizi tekrar tekrar kullanırken bulabileceğinizi) düşündüğünüz herhangi bir şeyi işlevsel hale getirmek iyi bir fikirdir.

Değerlendirme metriklerimiz için fonksiyonlar yaratalım

```python
def mae(y_test, y_pred):
  """
  y_test ve y_preds arasındaki ortalama mutlak hatayı hesaplar.
  """
  return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mse(y_test, y_pred):
  """
  y_test ve y_preds arasındaki ortalama karesel hatayı hesaplar
  """
  return tf.metrics.mean_squared_error(y_test,
                                       y_pred)
```

### Bir Modeli Geliştirmek İçin Denemeler Yapmak

Değerlendirme metriklerini ve modelinizin yaptığı tahminleri gördükten sonra, muhtemelen modeli geliştirmek isteyeceksiniz.

Yine, bunu yapmanın birçok farklı yolu vardır, ancak bunlardan başlıca 3 tanesi şunlardır:

- **Daha fazla veri elde edin**<br>
Modeliniz için daha fazla örnek alın (kalıpları öğrenmek için daha fazla fırsat).
- **Modelinizi büyütün (daha karmaşık bir model kullanın)**<br>
Bu, her katmanda daha fazla katman veya daha fazla gizli birim şeklinde olabilir.
- **Daha uzun süre eğitin**<br>
Modelinize verilerdeki kalıpları bulma şansı verin.

Veri kümemizi oluşturduğumuzdan, kolayca daha fazla veri üretebiliyorduk, ancak gerçek dünya veri kümeleriyle çalışırken durum her zaman böyle olmuyor.

Şimdi 2 ve 3'ü kullanarak modelimizi nasıl geliştirebileceğimize bir göz atalım.

Bunu yapmak için 3 model oluşturacağız ve sonuçlarını karşılaştıracağız:

- `model_1` - orijinal modelle aynı, 1 katman, 100 epoch için eğitilmiş.
- `model_2` - 100 epoch için eğitilmiş 2 katman.
- `model_3` - 500 epoch için eğitilmiş 2 katman.

`Model_1` 


```python
tf.random.set_seed(42)

# Orijinal modeli çoğaltıyoruz
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# modeli derleme
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# modeli fit etme
model_1.fit(X_train, y_train, epochs=100, verbose=0)
```

```python
# tahminleri model_1 için görselleştirelim
y_preds_1 = model_1.predict(X_test)
plot_predictions(predictions=y_preds_1)
```
> <img src="https://i.ibb.co/Vv70dWN/indir-2.png" />


```python
mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
mae_1, mse_1
```
> (18.745327, 353.57336)



```python

```




```python

```



```python

```



```python

```



```python

```



```python

```



```python

```




