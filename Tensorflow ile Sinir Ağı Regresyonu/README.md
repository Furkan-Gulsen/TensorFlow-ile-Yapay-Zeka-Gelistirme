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

... devam edecek










