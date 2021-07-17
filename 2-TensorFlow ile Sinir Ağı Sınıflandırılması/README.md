# TensorFlow ile Sinir Ağı Sınıflandırılması

Tüm derin sinir ağlarının içerdiği bazı temeller vardır:
- Bir giriş katmanı
- Bazı gizli katmanlar
- Bir çıktı katmanı

Geri kalanın çoğu, modeli oluşturan veri analistine kalmış.

Sınıflandırma, bazı girdiler verilen bir sınıf etiketinin çıktısını almayı içeren tahmine dayalı bir modelleme problemidir.

Aşağıdakiler, sınıflandırma sinir ağlarınızda sıklıkla kullanacağınız bazı standart değerlerdir.

| **Hyperparameter** | **İkili Sınıflandırma** | **Çoklu Sınıflandırma** |
| --- | --- | --- |
| Input layer shape | Özellik sayısı ile aynı (örn; yaş için 5, cinsiyet, kilo, boy, sigara içenlerin kalp krizi riski) | İkili sınıflandırma ile aynı |
| Gizli katman | Probleme özel, minimum = 1, maksimum = sınırsız | İkili sınıflandırma ile aynı |
| Gizli katman başına nöron sayısı | Probleme özel, genellikle 10 ila 100 | İkili sınıflandırma ile aynı |
| Çıktı katmanı şekli | 1 (bir sınıf veya diğer) | Sınıf başına 1 adet (örneğin yemek, kişi veya köpek fotoğrafı için 3) |
| Gizli aktivasyon | Genelde [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning)  | İkili sınıflandırma ile aynı |
| Çıkış aktivasyonu | [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) | [Softmax](https://en.wikipedia.org/wiki/Softmax_function) |
| kayıp fonksiyonu | [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) ([`tf.keras.losses.BinaryCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) in TensorFlow) | Cross entropy ([`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) in TensorFlow) |
| Optimize Edici | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) | İkili sınıflandırma ile aynı |

## Veri Oluşturma

Bir sınıflandırma veri setini içe aktararak başlayabiliriz, ancak hadi kendi sınıflandırma verilerimizden oluşturmaya çalışalım.

Scikit-Learn'ün `make_circles()` işlevini veriseti oluşturmak için kullanalım.


```python
from sklearn.datasets import make_circles
import tensorflow as tf

# 1000 örnek
n_samples = 1000

# verisetini oluşturma
X, y = make_circles(n_samples, 
                    noise=0.03, 
                    random_state=42)
```

Verisetini oluşturuk gibi. Şimdi X ve y değerlerini kontrol edelim.


```python
# özelliklere göz atalım (x1, x2)
X
```




    array([[ 0.75424625,  0.23148074],
           [-0.75615888,  0.15325888],
           [-0.81539193,  0.17328203],
           ...,
           [-0.13690036, -0.81001183],
           [ 0.67036156, -0.76750154],
           [ 0.28105665,  0.96382443]])




```python
# ilk 10'nun etiketi
y[:10]
```




    array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])



Tamam, bazı verilerimizi ve etiketlerimizi gördük, görselleştirmeye geçmeye ne dersiniz?

> 🔑 Not: Herhangi bir tür makine öğrenimi projesi başlatmanın önemli bir adımı, verilerle bir olmaktır. Bunu yapmanın en iyi yollarından biri, üzerinde çalıştığınız verileri mümkün olduğunca görselleştirmektir.Slognamızı unutmayalım: "görselleştir, görselleştir, görselleştir".

Bir DataFrame yaratma ile başlayalım.


```python
import pandas as pd

circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
circles.head()
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
      <th>X0</th>
      <th>X1</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.754246</td>
      <td>0.231481</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.756159</td>
      <td>0.153259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.815392</td>
      <td>0.173282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.393731</td>
      <td>0.692883</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.442208</td>
      <td>-0.896723</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# etiketimize göre veri sayılarımız
circles.label.value_counts()
```




    1    500
    0    500
    Name: label, dtype: int64



Pekala, bir ikili (binary) sınıflandırma problemiyle uğraşıyoruz gibi görünüyor. İkilidir çünkü yalnızca iki etiket vardır (0 veya 1).

Daha fazla etiket seçeneği olsaydı (ör. 0, 1, 2, 3 veya 4), çok sınıflı (multi-label) sınıflandırma olarak adlandırılırdı.

Görselleştirmemizi bir adım daha ileri götürelim ve verilerimizi çizelim.


```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_10_0.png)
    


Maviyi kırmızı noktalardan ayırt edebilen bir model yaratalım.

## Girdi (Input) ve Çıkış (Output) Şekilleri

Sinir ağları oluştururken karşılaşacağınız en yaygın sorunlardan biri şekil uyumsuzluklarıdır.

Daha spesifik olarak, girdi verilerinin şekli ve çıktı verilerinin şekli.

Bizim durumumuzda, X'i girmek ve modelimizin y'yi tahmin etmesini sağlamak istiyoruz.

Şimdi X ve y'nin şekillerini kontrol edelim.



```python
# Özelliklerimizin ve etiketlerimizin şekillerini kontrol edelim
X.shape, y.shape
```




    ((1000, 2), (1000,))



Hmm, bu rakamlar nereden geliyor?


```python
# Kaç örneğimiz olduğunu kontrol edelim
len(X), len(y)
```




    (1000, 1000)



Yani y değeri kadar X değerimiz var, bu mantıklı.

Her birinin bir örneğini inceleyelim.


```python
X[0], y[0]
```




    (array([0.75424625, 0.23148074]), 1)



Bir y değeri sonucunu veren iki X özelliğimiz var.

Bu, sinir ağımızın girdi şeklinin en az bir boyutu iki olan bir tensörü kabul etmesi ve en az bir değere sahip bir tensör çıkışı vermesi gerektiği anlamına gelir.

> 🤔 Not: (1000,) şeklinde bir y olması kafa karıştırıcı görünebilir. Ancak bunun nedeni, tüm y değerlerinin aslında skaler (tek değerler) olması ve bu nedenle bir boyutu olmamasıdır. Şimdilik, çıktı şeklinizin en azından bir y örneğiyle aynı değerde olduğunu düşünün (bizim durumumuzda, sinir ağımızın çıktısı en az bir değer olmalıdır).

## Modellemedeki adımlar

Artık elimizde hangi verilere ve girdi ve çıktı şekillerine sahip olduğumuzu biliyoruz, onu modellemek için nasıl bir sinir ağı kuracağımıza bakalım.

TensorFlow'da bir model oluşturmak ve eğitmek için tipik olarak 3 temel adım vardır.

- **Bir model oluşturma**<br>
Bir sinir ağının katmanlarını kendiniz bir araya getirin (işlevsel veya sıralı API'yi kullanarak) veya önceden oluşturulmuş bir modeli içe aktarın (aktarım öğrenimi (transfer learning) olarak bilinir).
- **Bir model derleme**<br>
Bir modelin performansının nasıl ölçüleceğini (kayıp/metrikler) tanımlamanın yanı sıra nasıl iyileştirileceğini (optimizer) tanımlama.
- **Modeli fit etme**<br>
Modelin verilerdeki kalıpları bulmaya çalışmasına izin vermek (X, y'ye nasıl ulaşır).

Sıralı API'yi kullanarak bunları çalışırken görelim. Ve sonra her birinin üzerinden geçeceğiz.


```python
tf.random.set_seed(42)

# 1. Sıralı API'yi kullanarak modeli oluşturma
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Modeli derleme
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

# Modeli fit etme
model_1.fit(X, y, epochs=5)
```

    Epoch 1/5
    32/32 [==============================] - 1s 1ms/step - loss: 2.8544 - accuracy: 0.4600
    Epoch 2/5
    32/32 [==============================] - 0s 1ms/step - loss: 0.7131 - accuracy: 0.5430
    Epoch 3/5
    32/32 [==============================] - 0s 1ms/step - loss: 0.6973 - accuracy: 0.5090
    Epoch 4/5
    32/32 [==============================] - 0s 2ms/step - loss: 0.6950 - accuracy: 0.5010
    Epoch 5/5
    32/32 [==============================] - 0s 1ms/step - loss: 0.6942 - accuracy: 0.4830





    <tensorflow.python.keras.callbacks.History at 0x7f64a380b710>



Accuracy metriğine bakıldığında, modelimiz zayıf bir performans sergiliyor (ikili sınıflandırma probleminde %50 doğruluk, tahmin etmeye eşdeğerdir), ama ya onu daha uzun süre eğitirsek?


```python
# Modelimizi daha uzun süre eğitme
model_1.fit(X, y, epochs=200, verbose=0)
model_1.evaluate(X, y)
```

    32/32 [==============================] - 0s 1ms/step - loss: 0.6935 - accuracy: 0.5000





    [0.6934829950332642, 0.5]



Verilerin 200 geçişinden sonra bile, hala tahmin ediyormuş gibi performans gösteriyor.

Fazladan bir katman ekleyip biraz daha uzun süre eğitsek ne olur?


```python
tf.random.set_seed(42)

# 1. Sıralı API'yi kullanarak modeli oluşturma
model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1), # ek olarak bir katman ekleme
  tf.keras.layers.Dense(1)
])

# Modeli derleme
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

# Modeli fit etme
model_2.fit(X, y, epochs=100, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f64a14ff3d0>




```python
# modeli değerlendirme
model_2.evaluate(X, y)
```

    32/32 [==============================] - 0s 1ms/step - loss: 0.6933 - accuracy: 0.5000





    [0.6933314800262451, 0.5]



## Bir Modeli Geliştirmek

Modelimizi geliştirmek için daha önce geçtiğimiz 3 adımın neredeyse her bölümünü değiştirebiliriz.

- **Bir model oluşturma** <br>
Burada daha fazla katman eklemek, her katmandaki gizli birimlerin (nöronlar olarak da adlandırılır) sayısını artırmak, her katmanın etkinleştirme işlevlerini değiştirmek isteyebilirsiniz.
- **Bir model derleme** <br>
Farklı bir optimizasyon işlevi (genellikle birçok sorun için oldukça iyi olan Adam optimize edici gibi) seçmek veya belki de optimizasyon işlevinin öğrenme oranını değiştirmek isteyebilirsiniz.
- **Bir modeli fit etme** <br>
Belki de bir modeli daha fazla epoch'a sığdırabilirsiniz (onu daha uzun süre eğitmeye bırakın).

Daha fazla nöron, fazladan bir katman ve Adam optimize edici eklemeye ne dersiniz?

Elbette bunu yapmak tahmin etmekten daha iyidir...


```python
tf.random.set_seed(42)

# 1. Sıralı API'yi kullanarak modeli oluşturma
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 yoğun nöron ekleyin
  tf.keras.layers.Dense(10), # 10 nöronlu başka bir katman ekleyin
  tf.keras.layers.Dense(1)
])

# Modeli derleme
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Modeli fit etme
model_3.fit(X, y, epochs=100, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f64a24d2c10>




```python
#  Modeli değerlendirme
model_3.evaluate(X, y)
```

    32/32 [==============================] - 0s 1ms/step - loss: 0.6980 - accuracy: 0.5080





    [0.6980254650115967, 0.5080000162124634]



Birkaç numara çıkardık ama modelimiz tahmin etmekten bile daha iyi değil.

Neler olduğunu görmek için bazı görselleştirmeler yapalım.

> 🔑 Not: Modeliniz garip bir şekilde performans gösterdiğinde veya verilerinizle ilgili tam olarak emin olmadığınız bir şeyler olduğunda, şu üç kelimeyi hatırlayın: görselleştir, görselleştir, görselleştir. Verilerinizi inceleyin, modelinizi inceleyin, modelinizin tahminlerini inceleyin.

Modelimizin tahminlerini görselleştirmek için bir `plot_decision_boundary()` fonksiyonu oluşturacağız ve bu fonksiyon:

Eğitilmiş bir modeli, özellikleri (X) ve etiketleri (y) alır.
- Farklı X değerlerinden oluşan bir ağ ızgarası oluşturur.
- [Meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) üzerinden tahminler yapar.
- Tahminleri ve farklı bölgeler (her benzersiz sınıfın düştüğü yer) arasında bir çizgi çizer.

> 🔑 Not: Bir işlevin ne yaptığından emin değilseniz, ne yaptığını görmek için onu çözmeyi ve satır satır yazmayı deneyin. Küçük parçalara ayırın ve her bir parçanın çıktısını görün.


```python
import numpy as np

def plot_decision_boundary(model, X, y):

  # Çizimin eksen sınırlarını tanımlayın ve bir ağ ızgarası oluşturun
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # x değerlerini yaratın
  x_in = np.c_[xx.ravel(), yy.ravel()] 
  
  # Eğitilmiş modeli kullanarak tahminler yapın
  y_pred = model.predict(x_in)

  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
```


```python
# Modelimizin yaptığı tahminlere göz atalım
plot_decision_boundary(model_3, X, y)
```

    doing binary classifcation...



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_31_1.png)
    


Görünüşe göre modelimiz veriler arasında düz bir çizgi çizmeye çalışıyor. 

Düz çizgi çizmeye çalışmasının hatası ne? <br>
Ana sorun, verilerimizin düz bir çizgiyle ayrılamamasıdır. Eğer verilerimiz bir regresyon tanımlasaydı modelimiz doğru bir yöntem uygulamış olacaktı. Hadi deneyelim


```python
tf.random.set_seed(42)

# regresyon verisi yaratma
X_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# Eğitim ve test setlerine ayırma
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# modeli yeni veriyle ffit etme
model_3.fit(X_reg_train, y_reg_train, epochs=100)
```

    Epoch 1/100



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-18-87efe03ef25e> in <module>()
         12 
         13 # modeli yeni veriyle ffit etme
    ---> 14 model_3.fit(X_reg_train, y_reg_train, epochs=100)
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1181                 _r=1):
       1182               callbacks.on_train_batch_begin(step)
    -> 1183               tmp_logs = self.train_function(iterator)
       1184               if data_handler.should_sync:
       1185                 context.async_wait()


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        887 
        888       with OptionalXlaContext(self._jit_compile):
    --> 889         result = self._call(*args, **kwds)
        890 
        891       new_tracing_count = self.experimental_get_tracing_count()


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        915       # In this case we have created variables on the first call, so we run the
        916       # defunned version which is guaranteed to never create variables.
    --> 917       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        918     elif self._stateful_fn is not None:
        919       # Release the lock early so that multiple threads can perform the call


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py in __call__(self, *args, **kwargs)
       3020     with self._lock:
       3021       (graph_function,
    -> 3022        filtered_flat_args) = self._maybe_define_function(args, kwargs)
       3023     return graph_function._call_flat(
       3024         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py in _maybe_define_function(self, args, kwargs)
       3439               call_context_key in self._function_cache.missed):
       3440             return self._define_function_with_shape_relaxation(
    -> 3441                 args, kwargs, flat_args, filtered_flat_args, cache_key_context)
       3442 
       3443           self._function_cache.missed.add(call_context_key)


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py in _define_function_with_shape_relaxation(self, args, kwargs, flat_args, filtered_flat_args, cache_key_context)
       3361 
       3362     graph_function = self._create_graph_function(
    -> 3363         args, kwargs, override_flat_arg_shapes=relaxed_arg_shapes)
       3364     self._function_cache.arg_relaxed[rank_only_cache_key] = graph_function
       3365 


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py in _create_graph_function(self, args, kwargs, override_flat_arg_shapes)
       3287             arg_names=arg_names,
       3288             override_flat_arg_shapes=override_flat_arg_shapes,
    -> 3289             capture_by_value=self._capture_by_value),
       3290         self._function_attributes,
       3291         function_spec=self.function_spec,


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes)
        997         _, original_func = tf_decorator.unwrap(python_func)
        998 
    --> 999       func_outputs = python_func(*func_args, **func_kwargs)
       1000 
       1001       # invariant: `func_outputs` contains only Tensors, CompositeTensors,


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py in wrapped_fn(*args, **kwds)
        670         # the function a weak reference to itself to avoid a reference cycle.
        671         with OptionalXlaContext(compile_with_xla):
    --> 672           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
        673         return out
        674 


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py in wrapper(*args, **kwargs)
        984           except Exception as e:  # pylint:disable=broad-except
        985             if hasattr(e, "ag_error_metadata"):
    --> 986               raise e.ag_error_metadata.to_exception(e)
        987             else:
        988               raise


    ValueError: in user code:
    
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:855 train_function  *
            return step_function(self, iterator)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:845 step_function  **
            outputs = model.distribute_strategy.run(run_step, args=(data,))
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:1285 run
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:2833 call_for_each_replica
            return self._call_for_each_replica(fn, args, kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:3608 _call_for_each_replica
            return fn(*args, **kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:838 run_step  **
            outputs = model.train_step(data)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:795 train_step
            y_pred = self(x, training=True)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/base_layer.py:1013 __call__
            input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/input_spec.py:255 assert_input_compatibility
            ' but received input with shape ' + display_shape(x.shape))
    
        ValueError: Input 0 of layer sequential_2 is incompatible with the layer: expected axis -1 of input shape to have value 2 but received input with shape (None, 1)



Hata almak mı? Fark ettiniz değil mi? Modelimiz sınıflandırma için eğitilmiş bir model. Şimdi o modeli regresyon için eğitelim.


```python
tf.random.set_seed(42)

# Modeli yeniden oluşturma
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# derlenmiş modelimizin kaybını ve ölçümlerini değiştirme
model_3.compile(loss=tf.keras.losses.mae, 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae']) 

# modeli fit etme
model_3.fit(X_reg_train, y_reg_train, epochs=100, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f64a1c3abd0>



Tamam, modelimiz bir şeyler öğreniyor gibi görünüyor (mae değeri her epoch'ta aşağı doğru eğilim gösteriyor), hadi tahminlerini çizelim.


```python
# Eğitimli modelimiz ile tahminler yapma
y_reg_preds = model_3.predict(y_reg_test)

# Modelin tahminlerini regresyon verilerimize göre çizme
plt.figure(figsize=(10, 7))
plt.scatter(X_reg_train, y_reg_train, c='b', label='Training data')
plt.scatter(X_reg_test, y_reg_test, c='g', label='Testing data')
plt.scatter(X_reg_test, y_reg_preds.squeeze(), c='r', label='Predictions')
plt.legend();
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_37_0.png)
    


Tahminler mükemmel olmamakla birlikte fena da sayılmaz. Yani bu, modelimizin bir şeyler öğreniyor olması gerektiği anlamına geliyor. Sınıflandırma problemimiz için gözden kaçırdığımız bir şey olmalı.

## Doğrusal Olmama (Non-linearity)

Tamam, sinir ağımızın düz çizgileri modelleyebildiğini gördük. Düz olmayan (doğrusal olmayan) çizgiler ne olacak?

Sınıflandırma verilerimizi (kırmızı ve mavi daireleri) modelleyeceksek, bazı doğrusal olmayan çizgilere ihtiyacımız olacak.


```python
tf.random.set_seed(42)

# bir model yaratma
model_5 = tf.keras.Sequential([
  tf.keras.layers.Dense(1, activation=tf.keras.activations.linear), # 1 lineer aktivasyonlu gizli katman
  tf.keras.layers.Dense(1) # çıktı katmanı
])

# modeli derleme
model_5.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])

# modeli fit etme
history = model_5.fit(X, y, epochs=100, verbose=0)
```

Model çıktısına ayrıntılı olarak bakarsanız modelin hala öğrenmediğini göreceksiniz. Nöron ve katmanların sayısını artırırsak sizce öğrenmesini sağlayabilir miyiz?


```python
tf.random.set_seed(42)

# bir model yaratma
model_6 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu), # gizli katmanı 1, 4 nöronlu bir relu aktivasyonu
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu), # gizli katmanı 2, 4 nöronlu bir relu aktivasyonu
  tf.keras.layers.Dense(1) # çıktı katmanı
])

# modeli derleme
model_6.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])

# modeli fit etme
history = model_6.fit(X, y, epochs=100, verbose=0)
```


```python
# modeli değerlendirme
model_6.evaluate(X, y)
```

    32/32 [==============================] - 0s 1ms/step - loss: 7.7125 - accuracy: 0.5000





    [7.712474346160889, 0.5]



%50 doğruluğa ulaşıyoruz, modelimiz hala berbat sonuçlar veriyor.

Tahminler nasıl görünüyor?


```python
# 2 gizli katman kullanarak tahminlere göz atın
plot_decision_boundary(model_6, X, y)
```

    doing binary classifcation...



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_45_1.png)
    


İdeal olarak, sarı çizgiler kırmızı dairenin ve mavi dairenin iç kısmına gider.

Tamam, hadi bu daireyi bir kereliğine modelleyelim.

Bir model daha (söz veriyorum... aslında, bu sözü bozmak zorunda kalacağım... daha birçok model üreteceğiz).

Bu sefer çıktı katmanımızdaki aktivasyon fonksiyonunu da değiştireceğiz. Bir sınıflandırma modelinin mimarisini hatırlıyor musunuz? İkili sınıflandırma için, çıktı katmanı aktivasyonu genellikle Sigmoid aktivasyon fonksiyonudur.


```python
tf.random.set_seed(42)

# modeli yaratma
model_7 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
  ])

# modeli derleme
model_7.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# modeli fit etme
history = model_7.fit(X, y, epochs=100, verbose=0)
```


```python
# modeli değerlendirme
model_7.evaluate(X, y)
```

    32/32 [==============================] - 0s 1ms/step - loss: 0.2948 - accuracy: 0.9910





    [0.2948004901409149, 0.9909999966621399]



Süper. Modelimiz harika bir accuracy değeri verdi. Görselleştirip nasıl göründüğüne bakalım hemen.


```python
plot_decision_boundary(model_7, X, y)
```

    doing binary classifcation...



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_50_1.png)
    


Yapmak istediğimiz sınıflandırma işlemi gerçekleşmiş. Mavi ve kırmızıyı ayırmayı başardık. 

> 🤔 Soru: Yaptığımız tahminlerde yanlış olan ne? Burada modelimizi gerçekten doğru değerlendiriyor muyuz? İpucu: Model hangi verileri öğrendi ve biz neyi tahmin ettik?

Buna cevap vermeden önce, az önce ele aldığımız şeyin farkına varmak önemlidir.

> 🔑 Not: Doğrusal (düz çizgiler) ve doğrusal olmayan (düz olmayan çizgiler) işlevlerin birleşimi, sinir ağlarının temel temellerinden biridir.

Bunu şöyle düşünün:

Size sınırsız sayıda düz çizgi ve düz olmayan çizgi vermiş olsaydım, ne tür desenler çizebilirdiniz? Esasen sinir ağlarının verilerdeki kalıpları bulmak için yaptığı şey budur.

Kullandığımız aktivasyon fonksiyonları sayesinde hedeflediğimiz modele ulaşdık. Şimdi fikir edinmek adına aktivasyon fonksiyonları oluşturup, oluşturduğumuz bu fonksiyonların sonuçlarını değerlendirelim.


```python
A = tf.cast(tf.range(-10, 10), tf.float32)
A
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,
             1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
          dtype=float32)>




```python
# Görselleştirelim
plt.plot(A);
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_53_0.png)
    


Düz (doğrusal) bir çizgi! Güzel, şimdi sigmoid fonksiyonunu ile bu modeli yeniden oluşturalım ve verilerimize ne yaptığını görelim. 


```python
# Sigmoid - https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
def sigmoid(x):
  return 1 / (1 + tf.exp(-x))

sigmoid(A)
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([4.5397872e-05, 1.2339458e-04, 3.3535014e-04, 9.1105117e-04,
           2.4726233e-03, 6.6928510e-03, 1.7986210e-02, 4.7425874e-02,
           1.1920292e-01, 2.6894143e-01, 5.0000000e-01, 7.3105860e-01,
           8.8079703e-01, 9.5257413e-01, 9.8201376e-01, 9.9330717e-01,
           9.9752742e-01, 9.9908900e-01, 9.9966466e-01, 9.9987662e-01],
          dtype=float32)>




```python
# sigmoid fonksiyona soktuğumuz değerlerin çıktısını görselleştirelim
plt.plot(sigmoid(A));
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_56_0.png)
    


Düz olmayan (doğrusal olmayan) bir çizgi!

Tamam, ReLU işlevine ne dersiniz (ReLU tüm negatifleri 0'a çevirir ve pozitif sayılar aynı kalır)?


```python
# ReLU - https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu
def relu(x):
  return tf.maximum(0, x)

relu(A)
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6.,
           7., 8., 9.], dtype=float32)>




```python
# relu fonksiyona soktuğumuz değerlerin çıktısını görselleştirelim
plt.plot(relu(A));
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_59_0.png)
    


Düz olmayan bir çizgi daha!

Peki, TensorFlow'un lineer aktivasyon fonksiyonuna ne dersiniz?



```python
tf.keras.activations.linear(A)
```




    <tf.Tensor: shape=(20,), dtype=float32, numpy=
    array([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,
             1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
          dtype=float32)>



Girdiğimiz değerler olduğu gibi çıktı. Bunu kontrol edelim.


```python
A == tf.keras.activations.linear(A)
```




    <tf.Tensor: shape=(20,), dtype=bool, numpy=
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True])>



Tamam, bu nedenle, modelin yalnızca doğrusal etkinleştirme fonksiyonlarını kullanırken gerçekten hiçbir şey öğrenmemesi mantıklıdır, çünkü doğrusal etkinleştirme işlevi, giriş verilerimizi hiçbir şekilde değiştirmez.

Doğrusal olmayan fonksiyonlarımızla verilerimiz manipüle edilir. Bir sinir ağı, girdileri ve çıktıları arasındaki desenleri çizmek için bu tür dönüşümleri büyük ölçekte kullanır.

Şimdi, sinir ağlarının derinlerine dalmak yerine, öğrendiklerimizi farklı problemlere uygulayarak kodlamaya devam edeceğiz, ancak sahne arkasında neler olup bittiğine daha derinlemesine bakmak istiyorsanız, aşağıdaki bölüme göz atabilirsiniz.

> 📖 Kaynak: Aktivasyon işlevleri hakkında daha fazla bilgi için, bunlarla ilgili [makine öğrenimi sayfasına ](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#)bakın.

## Sınıflandırma Modelimizin Değerlendirilmesi Ve İyileştirilmesi

Yukarıdaki soruya cevap verdiyseniz, yanlış yaptığımızı anlamış olabilirsiniz. Modelimizi, train ettiğimiz aynı veriler üzerinde değerlendiriyorduk.

Verilerimizi train, validation (isteğe bağlı) ve test kümelerine bölmek daha iyi bir yaklaşım olacaktır.

Bunu yaptıktan sonra, modelimizi train setinde eğiteceğiz (verilerdeki desenleri bulmasına izin verin) ve ardından test setindeki değerleri tahmin etmek için kullanarak modelleri ne kadar iyi öğrendiğini göreceğiz.

Hadi yapalım.


```python
# Tüm veri setinde kaç örnek var?
len(X)
```




    1000




```python
# Verileri train ve test setlerine ayırın
X_train, y_train = X[:800], y[:800] 
X_test, y_test = X[800:], y[800:] 

# Verilerin şekillerini kontrol edin
X_train.shape, X_test.shape 
```




    ((800, 2), (200, 2))



Harika, şimdi train ve test setlerimiz var, hadi train verilerini modelleyelim ve modelimizin test setinde öğrendiklerini değerlendirelim.


```python
tf.random.set_seed(42)

# model yaratma
model_8 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary output değeri aldığı için
])

# modeli derleme
model_8.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])

# modeli fit etme
history = model_8.fit(X_train, y_train, epochs=25)
```

    Epoch 1/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6847 - accuracy: 0.5425
    Epoch 2/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6777 - accuracy: 0.5525
    Epoch 3/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6736 - accuracy: 0.5512
    Epoch 4/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6681 - accuracy: 0.5775
    Epoch 5/25
    25/25 [==============================] - 0s 2ms/step - loss: 0.6633 - accuracy: 0.5850
    Epoch 6/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6546 - accuracy: 0.5838
    Epoch 7/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6413 - accuracy: 0.6750
    Epoch 8/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.6264 - accuracy: 0.7013
    Epoch 9/25
    25/25 [==============================] - 0s 2ms/step - loss: 0.6038 - accuracy: 0.7487
    Epoch 10/25
    25/25 [==============================] - 0s 2ms/step - loss: 0.5714 - accuracy: 0.7738
    Epoch 11/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.5404 - accuracy: 0.7650
    Epoch 12/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.5015 - accuracy: 0.7837
    Epoch 13/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.4683 - accuracy: 0.7975
    Epoch 14/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.4113 - accuracy: 0.8450
    Epoch 15/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.3625 - accuracy: 0.9125
    Epoch 16/25
    25/25 [==============================] - 0s 2ms/step - loss: 0.3209 - accuracy: 0.9312
    Epoch 17/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.2847 - accuracy: 0.9488
    Epoch 18/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.2597 - accuracy: 0.9525
    Epoch 19/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.2375 - accuracy: 0.9563
    Epoch 20/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.2135 - accuracy: 0.9663
    Epoch 21/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.1938 - accuracy: 0.9775
    Epoch 22/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.1752 - accuracy: 0.9737
    Epoch 23/25
    25/25 [==============================] - 0s 3ms/step - loss: 0.1619 - accuracy: 0.9787
    Epoch 24/25
    25/25 [==============================] - 0s 2ms/step - loss: 0.1550 - accuracy: 0.9775
    Epoch 25/25
    25/25 [==============================] - 0s 1ms/step - loss: 0.1490 - accuracy: 0.9762



```python
# Modelimizi test setinde değerlendirelim
loss, accuracy = model_8.evaluate(X_test, y_test)
print(f"Test setinde model kaybı (loss): {loss}")
print(f"Test setinde model doğruluğu (accuracy): {100*accuracy:.2f}%")
```

    7/7 [==============================] - 0s 2ms/step - loss: 0.1247 - accuracy: 1.0000
    Test setinde model kaybı (loss): 0.1246885135769844
    Test setinde model doğruluğu (accuracy): 100.00%


%100 doğruluk? Güzel!

Şimdi model_8'i oluşturmaya başladığımızda model_7 ile aynı olacağını söylemiştik ama bunu biraz yalan bulmuş olabilirsiniz.

Çünkü birkaç şeyi değiştirdik:

- **Aktivasyon parametresi**<br>
TensorFlow'da kitaplık yolları (tf.keras.activations.relu) yerine dizeler ("relu" & "sigmoid") kullandık, ikisi de aynı işlevselliği sunar.
- **Learning_rate (ayrıca lr) parametresi**<br> 
Adam optimizer'deki öğrenme oranı parametresini 0,001 yerine 0,01'e yükselttik (10x artış).
  - Öğrenme oranını, bir modelin ne kadar hızlı öğrendiği olarak düşünebilirsiniz. Öğrenme oranı ne kadar yüksek olursa, modelin öğrenme kapasitesi o kadar hızlı olur, ancak, bir modelin çok hızlı öğrenmeye çalıştığı ve hiçbir şey öğrenmediği, çok yüksek bir öğrenme oranı gibi bir şey vardır. (overfitting ve underfitting kavramları)
- **Epoch sayısı**<br>
Epoch sayısını (epochs parametresini kullanarak) 100'den 25'e düşürdük, ancak modelimiz hem eğitim hem de test setlerinde hala inanılmaz bir sonuç aldı.
  - Modelimizin eskisinden bile daha az epoch sayısı ile iyi performans göstermesinin nedenlerinden biri (tek bir epoch, modelin verideki kalıpları bir kez bakarak öğrenmeye çalışması olduğunu unutmayın, bu nedenle 25 epoch, modelin 25 şansı olduğu anlamına gelir) öncekinden daha fazla öğrenme oranı.

Modelimizin değerlendirme ölçütlerine göre iyi performans gösterdiğini biliyoruz ancak görsel olarak nasıl performans gösterdiğini görelim.


```python
# Eğitim ve test setleri için karar sınırlarını çizin
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_8, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_8, X=X_test, y=y_test)
plt.show()
```

    doing binary classifcation...
    doing binary classifcation...



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_72_1.png)
    


İnce ayar (fine tuning) ile neredeyse mükemmel bir model eğittik.

## Kayıp (Loss) Eğrilerini Görselleştirin

Yukarıdaki grafiklere baktığımızda modelimizin çıktılarının çok iyi olduğunu görebiliriz. Ama modelimiz öğrenirken nasıl bir yol izledi?

Olduğu gibi, modelin verilere bakma şansı olduğu her seferinde (her epoch bir kez) performans nasıl değişti? Bunu anlamak için kayıp (loss) eğrilerini kontrol edebiliriz.Bir modelde fit() işlevini çağırırken değişken geçmişini kullandığımızı görmüş olabilirsiniz (fit() bir Geçmiş nesnesi döndürür).

Modelimizin öğrenirken nasıl performans gösterdiğine dair bilgileri buradan alacağız. Bakalım nasıl kullanacağız...


```python
# history niteliğini kullanarak geçmiş değişkenindeki bilgilere erişebilirsiniz
pd.DataFrame(history.history)
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
      <th>loss</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.684651</td>
      <td>0.54250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.677721</td>
      <td>0.55250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.673595</td>
      <td>0.55125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.668149</td>
      <td>0.57750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.663269</td>
      <td>0.58500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.654567</td>
      <td>0.58375</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.641258</td>
      <td>0.67500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.626428</td>
      <td>0.70125</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.603831</td>
      <td>0.74875</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.571404</td>
      <td>0.77375</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.540443</td>
      <td>0.76500</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.501504</td>
      <td>0.78375</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.468332</td>
      <td>0.79750</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.411302</td>
      <td>0.84500</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.362506</td>
      <td>0.91250</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.320904</td>
      <td>0.93125</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.284708</td>
      <td>0.94875</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.259720</td>
      <td>0.95250</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.237469</td>
      <td>0.95625</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.213520</td>
      <td>0.96625</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.193820</td>
      <td>0.97750</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.175244</td>
      <td>0.97375</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.161893</td>
      <td>0.97875</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.154989</td>
      <td>0.97750</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.148973</td>
      <td>0.97625</td>
    </tr>
  </tbody>
</table>
</div>



Çıktıları inceleyerek kayıp değerlerinin düştüğünü ve doğruluğun arttığını görebiliriz.

Nasıl görünüyor (görselleştirin, görselleştirin, görselleştirin)?



```python
pd.DataFrame(history.history).plot()
plt.title("Model_8 training curves")
```




    Text(0.5, 1.0, 'Model_8 training curves')




    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_77_1.png)
    


Güzel. Bu, bir sınıflandırma problemiyle uğraşırken aradığımız ideal görsellemedir, kayıp azalır, doğruluk artar.

> 🔑 Not: Birçok problem için, kayıp fonksiyonunun düşmesi, modelin iyileştiği anlamına gelir.

## En İyi Öğrenme Oranını (Learning Rate) Bulma

Mimarinin (katmanlar, nöronların sayısı, aktivasyonlar vb.) yanı sıra, sinir ağı modelleriniz için ayarlayabileceğiniz en önemli hiperparametre öğrenme oranıdır (lr). 

model_8'de Adam optimizer'ın öğrenme oranını varsayılan 0,001'den (varsayılan) 0,01'e indirdiğimizi gördünüz. Ve bunu neden yaptığımızı merak ediyor olabilirsiniz. Şanslı bir tahmindi.

Daha düşük bir öğrenme oranı denemeye ve modelin nasıl gittiğini görmeye karar verdim. Şimdi "Cidden mi? Bunu yapabilir misin?" diye düşünüyor olabilirsiniz. Ve cevap evet. Sinir ağlarınızın hiperparametrelerinden herhangi birini değiştirebilirsiniz. Pratik yaparak, ne tür hiperparametrelerin işe yarayıp nelerin yaramadığını görmeye başlayacaksınız.

Bu, genel olarak makine öğrenimi ve derin öğrenme hakkında anlaşılması gereken önemli bir şeydir. Bir model kuruyorsunuz ve onu değerlendiriyorsunuz, bir model kuruyorsunuz ve onu değerlendiriyorsunuz ... bu döngü en iyi veya ideal sonuca erişene dek devam ediyor.

Bununla birlikte, ileriye dönük modelleriniz için en uygun öğrenme oranını (en azından eğitime başlamak için) bulmanıza yardımcı olacak bir numara tanıtmak istiyorum. Bunu yapmak için aşağıdakileri kullanacağız:

- Bir [learning rate callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler).
  - Geri aramayı (backpropagation), eğitim sırasında modelinize ekleyebileceğiniz ekstra bir işlevsellik parçası olarak düşünebilirsiniz.
- Başka bir model (yukarıdakilerin aynısını kullanabiliriz, burada model oluşturma alıştırması yapıyoruz).
- Değiştirilmiş bir kayıp eğrileri grafiği

Her birinin üzerinden kodlarla geçeceğiz, sonra da neler olduğunu açıklayacağız.

> 🔑 Not: TensorFlow'daki birçok sinir ağı yapı taşının varsayılan hiperparametreleri, genellikle kutudan çıkar çıkmaz çalışacak şekilde kurulur (örneğin, Adam optimizer'ın varsayılan ayarları genellikle birçok veri kümesinde iyi sonuçlar alabilir). Bu nedenle, önce varsayılanları denemek ve ardından gerektiği gibi ayarlamak (fine tuning) iyi bir fikirdir.


```python
tf.random.set_seed(42)

# bir model yaratalım (model_8'in aynısı)
model_9 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# modeli derleyelim
model_9.compile(loss="binary_crossentropy", 
              optimizer="Adam", 
              metrics=["accuracy"]) 

# öğrenme oranını belli bir kurala göre azaltacak fonksiyonu ekleme
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) 
# 1e-4'ten başlayarak her epochta 10**(epoch/20) artan bir dizi öğrenme oranı (learning rate)

# modeli fit etme
history = model_9.fit(X_train, 
                      y_train, 
                      epochs=100,
                      callbacks=[lr_scheduler])
```

    Epoch 1/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6945 - accuracy: 0.4988
    Epoch 2/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6938 - accuracy: 0.4975
    Epoch 3/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6930 - accuracy: 0.4963
    Epoch 4/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6922 - accuracy: 0.4975
    Epoch 5/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6914 - accuracy: 0.5063
    Epoch 6/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6906 - accuracy: 0.5013
    Epoch 7/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6898 - accuracy: 0.4950
    Epoch 8/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6889 - accuracy: 0.5038
    Epoch 9/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6880 - accuracy: 0.5013
    Epoch 10/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6871 - accuracy: 0.5050
    Epoch 11/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6863 - accuracy: 0.5200
    Epoch 12/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6856 - accuracy: 0.5163
    Epoch 13/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6847 - accuracy: 0.5175
    Epoch 14/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6842 - accuracy: 0.5200
    Epoch 15/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6835 - accuracy: 0.5213
    Epoch 16/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6829 - accuracy: 0.5213
    Epoch 17/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6826 - accuracy: 0.5225
    Epoch 18/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6819 - accuracy: 0.5300
    Epoch 19/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6816 - accuracy: 0.5312
    Epoch 20/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6811 - accuracy: 0.5387
    Epoch 21/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6806 - accuracy: 0.5400
    Epoch 22/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6801 - accuracy: 0.5412
    Epoch 23/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6796 - accuracy: 0.5400
    Epoch 24/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6790 - accuracy: 0.5425
    Epoch 25/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6784 - accuracy: 0.5450
    Epoch 26/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6778 - accuracy: 0.5387
    Epoch 27/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6770 - accuracy: 0.5425
    Epoch 28/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6760 - accuracy: 0.5537
    Epoch 29/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6754 - accuracy: 0.5512
    Epoch 30/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6739 - accuracy: 0.5575
    Epoch 31/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6726 - accuracy: 0.5500
    Epoch 32/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6711 - accuracy: 0.5512
    Epoch 33/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6688 - accuracy: 0.5562
    Epoch 34/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6672 - accuracy: 0.5612
    Epoch 35/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6660 - accuracy: 0.5888
    Epoch 36/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6625 - accuracy: 0.5625
    Epoch 37/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6560 - accuracy: 0.5813
    Epoch 38/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6521 - accuracy: 0.6025
    Epoch 39/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6415 - accuracy: 0.7088
    Epoch 40/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6210 - accuracy: 0.7113
    Epoch 41/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.5904 - accuracy: 0.7487
    Epoch 42/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.5688 - accuracy: 0.7312
    Epoch 43/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.5346 - accuracy: 0.7563
    Epoch 44/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.4533 - accuracy: 0.8150
    Epoch 45/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.3455 - accuracy: 0.9112
    Epoch 46/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.2570 - accuracy: 0.9463
    Epoch 47/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1968 - accuracy: 0.9575
    Epoch 48/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1336 - accuracy: 0.9700
    Epoch 49/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.1310 - accuracy: 0.9613
    Epoch 50/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.1002 - accuracy: 0.9700
    Epoch 51/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1166 - accuracy: 0.9638
    Epoch 52/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1368 - accuracy: 0.9513
    Epoch 53/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.0879 - accuracy: 0.9787
    Epoch 54/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1187 - accuracy: 0.9588
    Epoch 55/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.0733 - accuracy: 0.9712
    Epoch 56/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1132 - accuracy: 0.9550
    Epoch 57/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.1057 - accuracy: 0.9613
    Epoch 58/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.0664 - accuracy: 0.9750
    Epoch 59/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1898 - accuracy: 0.9275
    Epoch 60/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1895 - accuracy: 0.9312
    Epoch 61/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.4131 - accuracy: 0.8612
    Epoch 62/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1707 - accuracy: 0.9725
    Epoch 63/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.0569 - accuracy: 0.9937
    Epoch 64/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.1007 - accuracy: 0.9638
    Epoch 65/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.1323 - accuracy: 0.9488
    Epoch 66/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.1819 - accuracy: 0.9375
    Epoch 67/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6672 - accuracy: 0.7613
    Epoch 68/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.5301 - accuracy: 0.6687
    Epoch 69/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.4140 - accuracy: 0.7925
    Epoch 70/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.4574 - accuracy: 0.7412
    Epoch 71/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.4759 - accuracy: 0.7262
    Epoch 72/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.3748 - accuracy: 0.8112
    Epoch 73/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.4710 - accuracy: 0.8150
    Epoch 74/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.4143 - accuracy: 0.8087
    Epoch 75/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.5961 - accuracy: 0.7412
    Epoch 76/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.4787 - accuracy: 0.7713
    Epoch 77/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.4720 - accuracy: 0.7113
    Epoch 78/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.2565 - accuracy: 0.8675
    Epoch 79/100
    25/25 [==============================] - 0s 1ms/step - loss: 1.1824 - accuracy: 0.6275
    Epoch 80/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.6873 - accuracy: 0.5425
    Epoch 81/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.7068 - accuracy: 0.5575
    Epoch 82/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6879 - accuracy: 0.5838
    Epoch 83/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6996 - accuracy: 0.5700
    Epoch 84/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.6471 - accuracy: 0.5863
    Epoch 85/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7457 - accuracy: 0.5312
    Epoch 86/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7546 - accuracy: 0.5038
    Epoch 87/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7681 - accuracy: 0.5063
    Epoch 88/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7596 - accuracy: 0.4963
    Epoch 89/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.7778 - accuracy: 0.5063
    Epoch 90/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7741 - accuracy: 0.4787
    Epoch 91/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.7851 - accuracy: 0.5163
    Epoch 92/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7441 - accuracy: 0.4888
    Epoch 93/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.7354 - accuracy: 0.5163
    Epoch 94/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7548 - accuracy: 0.4938
    Epoch 95/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.8087 - accuracy: 0.4863
    Epoch 96/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.7714 - accuracy: 0.4638
    Epoch 97/100
    25/25 [==============================] - 0s 1ms/step - loss: 0.8001 - accuracy: 0.5013
    Epoch 98/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.9554 - accuracy: 0.4963
    Epoch 99/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.9268 - accuracy: 0.4913
    Epoch 100/100
    25/25 [==============================] - 0s 2ms/step - loss: 0.8563 - accuracy: 0.4663


Modelimizin eğitimi bitti, şimdi eğitim grafiğine bir göz atalım.


```python
pd.DataFrame(history.history).plot(figsize=(10,7), xlabel="epochs");
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_82_0.png)
    


Gördüğünüz gibi epoch sayısı arttıkça öğrenme oranı katlanarak artıyor. Ve öğrenme oranı yavaşça arttığında belirli bir noktada modelin doğruluğunun arttığını (ve kaybın azaldığını) görebilirsiniz.

Bu çarpma noktasının nerede olduğunu bulmak için, günlük ölçekli öğrenme oranına karşı kaybı çizebiliriz.



```python
# Kayba karşı öğrenme oranını çizin
lrs = 1e-4 * (10 ** (np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_84_0.png)
    


Öğrenme hızının ideal değerini bulmak için (en azından modelimizi eğitmeye başlamak için ideal değer), temel kural, kaybın hala azalmakta olduğu ancak tam olarak düzleşmediği eğriyi kullanmaktır. Bu durumda ideal öğrenme oranımız 0,01  ile 0,02 arasında olur.

Şimdi modelimiz için ideal öğrenme oranını (0,02 kullanacağız) tahmin ettik, hadi bu değerle yeniden eğitelim.


```python
tf.random.set_seed(42)

# bir model yaratma
model_10 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# yeni lr değeri ile modeli derleme
model_10.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                metrics=["accuracy"])

history = model_10.fit(X_train, y_train, epochs=20)
```

    Epoch 1/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.6837 - accuracy: 0.5600
    Epoch 2/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.6744 - accuracy: 0.5750
    Epoch 3/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.6626 - accuracy: 0.5875
    Epoch 4/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.6332 - accuracy: 0.6388
    Epoch 5/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.5830 - accuracy: 0.7563
    Epoch 6/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.4907 - accuracy: 0.8313
    Epoch 7/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.4251 - accuracy: 0.8450
    Epoch 8/20
    25/25 [==============================] - 0s 2ms/step - loss: 0.3596 - accuracy: 0.8875
    Epoch 9/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.3152 - accuracy: 0.9100
    Epoch 10/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.2512 - accuracy: 0.9500
    Epoch 11/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.2152 - accuracy: 0.9500
    Epoch 12/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.1721 - accuracy: 0.9750
    Epoch 13/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.1443 - accuracy: 0.9837
    Epoch 14/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.1232 - accuracy: 0.9862
    Epoch 15/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.1085 - accuracy: 0.9850
    Epoch 16/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.0940 - accuracy: 0.9937
    Epoch 17/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.0827 - accuracy: 0.9962
    Epoch 18/20
    25/25 [==============================] - 0s 2ms/step - loss: 0.0798 - accuracy: 0.9937
    Epoch 19/20
    25/25 [==============================] - 0s 1ms/step - loss: 0.0845 - accuracy: 0.9875
    Epoch 20/20
    25/25 [==============================] - 0s 2ms/step - loss: 0.0790 - accuracy: 0.9887


Güzel! Biraz daha yüksek öğrenme oranıyla (0,01 yerine 0,02), daha az epoch (25 yerine 20) ile model_8'den daha yüksek bir doğruluğa ulaşıyoruz.


```python
# Modeli test veri seti ile değerlendirin
model_10.evaluate(X_test, y_test)
```

    7/7 [==============================] - 0s 3ms/step - loss: 0.0574 - accuracy: 0.9900





    [0.05740184709429741, 0.9900000095367432]



Bakalım tahminler göreselleştirme üzerinde nasıl duruyor.


```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_10, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_10, X=X_test, y=y_test)
plt.show()
```

    doing binary classifcation...
    doing binary classifcation...



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_90_1.png)
    


Ve gördüğümüz gibi, yine neredeyse mükemmel. Bunlar, kendi modellerinizi oluştururken sıklıkla gerçekleştireceğiniz deneylerdir. Varsayılan ayarlarla başlayın ve bunların verilerinizde nasıl performans gösterdiğini görün. Ve istediğiniz kadar iyi performans göstermiyorlarsa, geliştirin. Sınıflandırma modellerimizi değerlendirmenin birkaç yoluna daha bakalım.


## Daha Fazla Sınıflandırma Değerlendirme Yöntemi

Yaptığımız görselleştirmelerin yanı sıra, sınıflandırma modellerimizi değerlendirmek için kullanabileceğimiz bir dizi farklı değerlendirme ölçütü var.

| **Metrik adı/Değerlendirme yöntemi*** | **Tanım** | **Kod** |
| --- | --- | --- |
| Accuracy | Modeliniz 100 tahminden kaçı doğru çıkıyor? Örneğin. %95 doğruluk, 95/100 tahminlerin doğru olduğu anlamına gelir. | [`sklearn.metrics.accuracy_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) veya [`tf.keras.metrics.Accuracy()`](tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy) |
| Precision | Gerçek pozitiflerin toplam örnek sayısına oranı. Daha yüksek hassasiyet, daha az yanlış pozitife yol açar (model, 0 olması gerektiğinde 1'i tahmin eder). | [`sklearn.metrics.precision_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) veya [`tf.keras.metrics.Precision()`](tensorflow.org/api_docs/python/tf/keras/metrics/Precision) |
| Recall | Gerçek pozitiflerin toplam gerçek pozitif ve yanlış negatif sayısı üzerindeki oranı (model, 1 olması gerektiğinde 0'ı tahmin eder). Daha yüksek hatırlama, daha az yanlış negatife yol açar. | [`sklearn.metrics.recall_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) or [`tf.keras.metrics.Recall()`](tensorflow.org/api_docs/python/tf/keras/metrics/Recall) |
| F1-score | Kesinlik ve geri çağırmayı tek bir metrikte birleştirir. 1 en iyisidir, 0 en kötüsüdür. | [`sklearn.metrics.f1_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| [Confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)  | Tahmin edilen değerleri gerçek değerlerle tablo şeklinde karşılaştırır, %100 doğruysa matristeki tüm değerler sol üstten sağ alta olacaktır. | Custom function or [`sklearn.metrics.plot_confusion_matrix()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html) |
| Classification report | Kesinlik, geri çağırma ve f1 puanı gibi bazı ana sınıflandırma ölçütlerinin toplanması. | [`sklearn.metrics.classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |



> 🔑 Not: Her sınıflandırma problemi, farklı türde değerlendirme yöntemleri gerektirecektir. Ama en azından yukarıdakilere aşina olmalısınız.

Doğrulukla başlayalım.

Modelimizi derlerken metrics parametresine `["accuracy"]` ilettiğimiz için, üzerinde `evaluate()` öğesinin çağrılması, doğruluğun yanı sıra kaybı da döndürecektir.


```python
loss, accuracy = model_10.evaluate(X_test, y_test)
print(f"Test setinde model kaybı:: {loss}")
print(f"Test setinde model doğruluğu: {(accuracy*100):.2f}%")
```

    7/7 [==============================] - 0s 2ms/step - loss: 0.0574 - accuracy: 0.9900
    Test setinde model kaybı:: 0.05740184709429741
    Test setinde model doğruluğu: 99.00%



```python
from sklearn.metrics import confusion_matrix

# tahminler yapma
y_preds = model_10.predict(X_test)

# bir confusion matrix oluşturma
confusion_matrix(y_test, y_preds)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-50-f9255bdba7ac> in <module>()
          5 
          6 # bir confusion matrix oluşturma
    ----> 7 confusion_matrix(y_test, y_preds)
    

    /usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py in confusion_matrix(y_true, y_pred, labels, sample_weight, normalize)
        266 
        267     """
    --> 268     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        269     if y_type not in ("binary", "multiclass"):
        270         raise ValueError("%s is not supported" % y_type)


    /usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py in _check_targets(y_true, y_pred)
         88     if len(y_type) > 1:
         89         raise ValueError("Classification metrics can't handle a mix of {0} "
    ---> 90                          "and {1} targets".format(type_true, type_pred))
         91 
         92     # We can't have more than one value on y_type => The set is no more needed


    ValueError: Classification metrics can't handle a mix of binary and continuous targets


Ahh, görünüşe göre tahminlerimiz olması gereken formatta değil. Onları kontrol edelim.



```python
# İlk 10 tahmine göz atalım
y_preds[:10]
```




    array([[9.8526537e-01],
           [9.9923790e-01],
           [9.9032348e-01],
           [9.9706942e-01],
           [3.9622977e-01],
           [1.8126935e-02],
           [9.6829069e-01],
           [1.9746721e-02],
           [9.9967170e-01],
           [5.6460500e-04]], dtype=float32)




```python
# ilk 10 test etiketine göz atalım
y_test[:10]
```




    array([1, 1, 1, 1, 0, 0, 1, 0, 1, 0])



Tahminlerimizi ikili formata (0 veya 1) almamız gerekiyor gibi görünüyor.

Ama merak ediyor olabilirsiniz, şu anda hangi formattalar?  Mevcut formatlarında (9.8526537e-01), tahmin olasılıkları adı verilen bir formdalar.

Bunu sinir ağlarının çıktılarında sıklıkla göreceksiniz. Genellikle kesin değerler olmayacaklar, ancak daha çok, şu veya bu değer olma olasılıklarının bir olasılığı olacaktır. Bu nedenle, bir sinir ağı ile tahminler yaptıktan sonra sıklıkla göreceğiniz adımlardan biri, tahmin olasılıklarını etiketlere dönüştürmektir. Bizim durumumuzda, temel doğruluk etiketlerimiz (y_test) ikili (0 veya 1) olduğundan, tf.round() kullanarak tahmin olasılıklarını ikili biçimlerine dönüştürebiliriz.



```python
tf.round(y_preds)[:10]
```




    <tf.Tensor: shape=(10, 1), dtype=float32, numpy=
    array([[1.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.]], dtype=float32)>



İşte şimdi oldu. confusion matrix uygulayabiliriz.


```python
# bir confusion matrix oluşturma
confusion_matrix(y_test, tf.round(y_preds))
```




    array([[99,  2],
           [ 0, 99]])



Pekala, en yüksek sayıların sol üst ve sağ alta olduğunu görebiliyoruz, yani bu iyi bir işaret, ancak matrisin geri kalanı bize pek bir şey söylemiyor.

Karışıklık matrisimizi biraz daha görsel hale getirmek için bir fonksiyon yazmaua ne dersiniz?



```python
import itertools

figsize = (10, 10)

# confusion matrix oluşturma
cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize etme
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize=figsize)
cax = ax.matshow(cm, cmap=plt.cm.Blues) 
fig.colorbar(cax)

classes = False

if classes:
  labels = classes
else:
  labels = np.arange(cm.shape[0])

# eksenleri etiketleme
ax.set(title="Confusion Matrix",
       xlabel="Predicted label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

# x ekseni etiketlerini en alta ayarla
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

# Etiket boyutunu ayarla
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(20)

# Farklı renkler için eşik ayarla
threshold = (cm.max() + cm.min()) / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
           horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=15)
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_104_0.png)
    


Bu çok daha iyi görünüyor. Görünüşe göre modelimiz, iki yanlış pozitif (sağ üst köşe) dışında test setinde neredeyse mükemmel tahminler yaptı.

## Daha Büyük Bir Örnekle Çalışma (Multi-Class Classification)

İkili bir sınıflandırma örneği gördük (bir veri noktasının kırmızı bir dairenin mi yoksa mavi bir dairenin mi parçası olduğunu tahmin ederek) ama ya birden fazla farklı nesne sınıfınız varsa?

Örneğin, bir moda şirketi olduğunuzu ve bir giysinin ayakkabı mı, gömlek mi yoksa ceket mi olduğunu tahmin etmek için bir sinir ağı kurmak istediğinizi varsayalım (3 farklı seçenek). Seçenek olarak ikiden fazla sınıfınız olduğunda, buna çok sınıflı sınıflandırma denir. İyi haber şu ki, şimdiye kadar öğrendiklerimiz (birkaç ince ayar ile) çok sınıflı sınıflandırma problemlerine de uygulanabilir.


```python
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# verileri değişenlere atayalım
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 0s 0us/step



```python
# İlk train örneğine göz atalım
print(f"Training sample:\n{train_data[0]}\n") 
print(f"Training label: {train_labels[0]}")
```

    Training sample:
    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
        0   1   4   0   0   0   0   1   1   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
       54   0   0   0   1   3   4   0   0   3]
     [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
      144 123  23   0   0   0   0  12  10   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
      107 156 161 109  64  23  77 130  72  15]
     [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
      216 163 127 121 122 146 141  88 172  66]
     [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
      223 223 215 213 164 127 123 196 229   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
      235 227 224 222 224 221 223 245 173   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
      180 212 210 211 213 223 220 243 202   0]
     [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
      169 227 208 218 224 212 226 197 209  52]
     [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
      198 221 215 213 222 220 245 119 167  56]
     [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
      232 213 218 223 234 217 217 209  92   0]
     [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
      222 221 216 223 229 215 218 255  77   0]
     [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
      211 218 224 223 219 215 224 244 159   0]
     [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
      224 234 176 188 250 248 233 238 215   0]
     [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
      255 255 221 234 221 211 220 232 246   0]
     [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
      188 154 191 210 204 209 222 228 225   0]
     [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
      168 219 221 215 217 223 223 224 229  29]
     [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
      239 223 218 212 209 222 220 221 230  67]
     [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
      199 206 186 181 177 172 181 205 206 115]
     [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
      195 191 198 192 176 156 167 177 210  92]
     [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
      210 210 211 188 188 194 192 216 170   0]
     [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
      182 182 181 176 166 168  99  58   0   0]
     [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]
    
    Training label: 9



```python
# Verilerimizin şeklini kontrol edelim
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
```




    ((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))




```python
# Tek bir örneğin şeklini kontrol edelim
train_data[0].shape, train_labels[0].shape
```




    ((28, 28), ())



Tamam, her biri şekil (28, 28) ve bir etiket içeren 60.000 train örneğinin yanı sıra 10.000 şekil test örneği (28, 28) vardır.

Ama bunlar sadece rakamlar, hadi görselleştirelim.



```python
# Bir örneğe göz atalım hemen
import matplotlib.pyplot as plt
plt.imshow(train_data[7]);
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_112_0.png)
    



```python
# etiketini kontrol edelim
train_labels[7]
```




    2



Görünüşe göre etiketlerimiz sayısal biçimde. Ve bu bir sinir ağı için iyi olsa da, onları insan tarafından okunabilir biçimde almak isteyebilirsiniz.

Sınıf adlarının küçük bir listesini oluşturalım (bunları veri kümesinin [`GitHub sayfasında`](https://github.com/zalandoresearch/fashion-mnist#labels) bulabiliriz).

🔑 Not: Bu veri kümesi bizim için hazırlanmış ve kullanıma hazır olsa da, birçok veri kümesinin bunun gibi kullanmaya hazır olmayacağını unutmamak önemlidir. Genellikle bir sinir ağıyla kullanıma hazır hale getirmek için birkaç ön işleme adımı yapmanız gerekir (daha sonra kendi verilerimizle çalışırken bunun daha fazlasını göreceğiz).


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

len(class_names)
```




    10



Şimdi bunlara sahibiz, başka bir örnek çizelim.

> 🤔 Soru: Üzerinde çalıştığımız verilerin nasıl göründüğüne özellikle dikkat edin. Sadece düz çizgiler mi? Yoksa düz olmayan çizgileri de var mı? Giysilerin fotoğraflarında (aslında piksel koleksiyonları olan) desenler bulmak istesek, modelimizin doğrusal olmayanlara (düz olmayan çizgiler) ihtiyacı olacak mı, olmayacak mı?




```python
# Örnek bir resim ve etiketini görüntüleme
plt.imshow(train_data[17], cmap=plt.cm.binary) # renkleri siyah beyaz olarak değiştirme
plt.title(class_names[train_labels[17]]);
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_117_0.png)
    



```python
# Moda MNIST'in birden fazla rastgele görüntü
import random
plt.figure(figsize=(7, 7))
for i in range(4):
  ax = plt.subplot(2, 2, i + 1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
  plt.title(class_names[train_labels[rand_index]])
  plt.axis(False)
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_118_0.png)
    


Pekala, piksel değerleri ve etiketleri arasındaki ilişkiyi bulmak için bir model oluşturalım.

Bu çok sınıflı bir sınıflandırma problemi olduğundan, mimarimizde birkaç değişiklik yapmamız gerekecek (yukarıdaki Tablo 1 ile aynı hizada):

- Giriş şekli 28x28 tensörlerle (resimlerimizin yüksekliği ve genişliği) uğraşmak zorunda kalacak.
Aslında girdiyi (784) bir tensöre (vektöre) sıkıştıracağız.
Çıktı şeklinin 10 olması gerekecek çünkü modelimizin 10 farklı sınıf için tahmin yapmasına ihtiyacımız var.
- Ayrıca çıktı katmanımızın aktivasyon parametresini 'sigmoid' yerine "softmax" olarak değiştireceğiz. 
  - Göreceğimiz gibi, "softmax" etkinleştirme işlevi 0 ve 1 arasında bir dizi değer verir (çıktı şekliyle aynı şekil, toplamları ~1'e eşittir. En yüksek değere sahip indeks, model tarafından tahmin edilir.
- Kayıp fonksiyonumuzu ikili kayıp fonksiyonundan çok sınıflı kayıp fonksiyonuna değiştirmemiz gerekecek.
  - Daha spesifik olarak, etiketlerimiz tamsayı biçiminde olduğundan, etiketlerimiz one-hot encoding (örneğin, [0, 0, 1, 0, 0 gibi görünüyorlardı) tf.keras.losses.SparseCategoricalCrossentropy() kullanacağız. ..]), tf.keras.losses.CategoricalCrossentropy() kullanırdık.
- fit() işlevini çağırırken validation_data parametresini de kullanacağız. Bu bize eğitim sırasında modelin test setinde nasıl performans gösterdiği hakkında bir fikir verecektir.



```python
tf.random.set_seed(42)

# bir model oluşturma
model_11 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax") 
])

# modeli derleme
model_11.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# modeli fit etme
non_norm_history = model_11.fit(train_data,
                                train_labels,
                                epochs=10,
                                validation_data=(test_data, test_labels))
```

    Epoch 1/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 2.1671 - accuracy: 0.1606 - val_loss: 1.7959 - val_accuracy: 0.2046
    Epoch 2/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.7066 - accuracy: 0.2509 - val_loss: 1.6567 - val_accuracy: 0.2805
    Epoch 3/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.6321 - accuracy: 0.2806 - val_loss: 1.6094 - val_accuracy: 0.2857
    Epoch 4/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.6052 - accuracy: 0.2833 - val_loss: 1.6041 - val_accuracy: 0.2859
    Epoch 5/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5975 - accuracy: 0.2862 - val_loss: 1.6064 - val_accuracy: 0.2756
    Epoch 6/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5950 - accuracy: 0.2920 - val_loss: 1.5747 - val_accuracy: 0.2994
    Epoch 7/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5775 - accuracy: 0.3040 - val_loss: 1.6030 - val_accuracy: 0.3000
    Epoch 8/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5708 - accuracy: 0.3175 - val_loss: 1.5635 - val_accuracy: 0.3315
    Epoch 9/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5638 - accuracy: 0.3280 - val_loss: 1.5534 - val_accuracy: 0.3334
    Epoch 10/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.5432 - accuracy: 0.3346 - val_loss: 1.5390 - val_accuracy: 0.3549



```python
# Modelimizin şekillerini kontrol edelim
model_11.summary()
```

    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense_26 (Dense)             (None, 4)                 3140      
    _________________________________________________________________
    dense_27 (Dense)             (None, 4)                 20        
    _________________________________________________________________
    dense_28 (Dense)             (None, 10)                50        
    =================================================================
    Total params: 3,210
    Trainable params: 3,210
    Non-trainable params: 0
    _________________________________________________________________


Pekala, modelimiz, ikili sınıflandırma problemimizde kullandığımıza benzer bir stil modeli kullanarak 10 epochtan sonra yaklaşık %35 doğruluğa ulaşıyor.

Hangisi tahmin etmekten daha iyidir (10 sınıfla tahmin etmek yaklaşık %10 doğrulukla sonuçlanır), ancak daha iyisini yapabiliriz.

0 ile 1 arasındaki sayıları tercih eden sinir ağlarından bahsettiğimizi hatırlıyor musunuz? (eğer hatırlamıyorsanız bunu bir hatırlatma olarak kabul edin)

Şu anda elimizdeki veriler 0 ile 1 arasında değil, başka bir deyişle, normalleştirilmedi (bu nedenle fit()'i çağırırken non_norm_history değişkenini kullandık). Piksel değerleri 0 ile 255 arasındadır.


```python
# Eğitim verilerinin minimum ve maksimum değerlerini kontrol edin
train_data.min(), train_data.max()
```




    (0, 255)



Bu değerleri 0 ile 1 arasında, tüm diziyi maksimuma bölerek elde edebiliriz: 0-255.

Bunu yapmak, tüm verilerimizin 0 ile 1 arasında olmasına neden olur (ölçeklendirme veya normalleştirme olarak bilinir).


```python
# traini bölün ve görüntüleri maksimum değere göre test edin (normalleştirin)
train_data = train_data / 255.0
test_data = test_data / 255.0

# Eğitim verilerinin minimum ve maksimum değerlerini kontrol edin
train_data.min(), train_data.max()
```




    (0.0, 1.0)



Güzel! Şimdi verilerimiz 0 ile 1 arasında. Modellediğimiz zaman bakalım ne olacak. Daha önce olduğu gibi (model_11) aynı modeli kullanacağız, ancak bu sefer veriler normalize edilecek.


```python
tf.random.set_seed(42)

# bir model oluşturma
model_12 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax") 
])

# modeli derleme
model_12.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# modeli fit etme
norm_history = model_12.fit(train_data,
                                train_labels,
                                epochs=10,
                                validation_data=(test_data, test_labels))
```

    Epoch 1/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 1.0348 - accuracy: 0.6474 - val_loss: 0.6937 - val_accuracy: 0.7617
    Epoch 2/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6376 - accuracy: 0.7757 - val_loss: 0.6400 - val_accuracy: 0.7820
    Epoch 3/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5942 - accuracy: 0.7914 - val_loss: 0.6247 - val_accuracy: 0.7783
    Epoch 4/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5750 - accuracy: 0.7979 - val_loss: 0.6078 - val_accuracy: 0.7881
    Epoch 5/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5641 - accuracy: 0.8006 - val_loss: 0.6169 - val_accuracy: 0.7881
    Epoch 6/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5544 - accuracy: 0.8043 - val_loss: 0.5855 - val_accuracy: 0.7951
    Epoch 7/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5488 - accuracy: 0.8063 - val_loss: 0.6097 - val_accuracy: 0.7836
    Epoch 8/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5428 - accuracy: 0.8077 - val_loss: 0.5787 - val_accuracy: 0.7971
    Epoch 9/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5373 - accuracy: 0.8097 - val_loss: 0.5698 - val_accuracy: 0.7977
    Epoch 10/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5360 - accuracy: 0.8124 - val_loss: 0.5658 - val_accuracy: 0.8014


Vayy, daha önce olduğu gibi aynı modeli kullandık ama normalleştirilmiş verilerle artık çok daha yüksek bir doğruluk değeri görüyoruz!

Her modelin history değerlerini (kayıp eğrilerini) görselleştirelim.



```python
import pandas as pd
pd.DataFrame(non_norm_history.history).plot(title="Non-normalized Data")
pd.DataFrame(norm_history.history).plot(title="Normalized data");
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_129_0.png)
    



    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_129_1.png)
    


Vay. Bu iki grafikten, normalize verili modelimizin (model_12) normalleştirilmemiş verili modele (model_11) göre ne kadar hızlı geliştiğini görebiliriz.

> 🔑 Not: Biraz farklı verilere sahip aynı model, önemli ölçüde farklı sonuçlar üretebilir. Bu nedenle, modelleri karşılaştırırken, onları aynı kriterlere göre karşılaştırdığınızdan emin olmanız önemlidir (örneğin, aynı mimari ancak farklı veriler veya aynı veriler ancak farklı mimari). İdeal öğrenme oranını bulup ne olduğunu görmeye ne dersiniz? Kullanmakta olduğumuz mimarinin aynısını kullanacağız.


```python
tf.random.set_seed(42)

# bir model oluşturma
model_13 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax") 
])

# modeli derleme
model_13.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# modeli fit etme
find_lr_history = model_13.fit(train_data,
                                train_labels,
                                epochs=40,
                                validation_data=(test_data, test_labels),
                                callbacks=[lr_scheduler])
```

    Epoch 1/40
    1875/1875 [==============================] - 4s 2ms/step - loss: 1.0348 - accuracy: 0.6474 - val_loss: 0.6937 - val_accuracy: 0.7617
    Epoch 2/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6366 - accuracy: 0.7759 - val_loss: 0.6400 - val_accuracy: 0.7808
    Epoch 3/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5934 - accuracy: 0.7911 - val_loss: 0.6278 - val_accuracy: 0.7770
    Epoch 4/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5749 - accuracy: 0.7969 - val_loss: 0.6122 - val_accuracy: 0.7871
    Epoch 5/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5655 - accuracy: 0.7987 - val_loss: 0.6061 - val_accuracy: 0.7913
    Epoch 6/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5569 - accuracy: 0.8022 - val_loss: 0.5917 - val_accuracy: 0.7940
    Epoch 7/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5542 - accuracy: 0.8036 - val_loss: 0.5898 - val_accuracy: 0.7896
    Epoch 8/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5509 - accuracy: 0.8039 - val_loss: 0.5829 - val_accuracy: 0.7949
    Epoch 9/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5468 - accuracy: 0.8047 - val_loss: 0.6036 - val_accuracy: 0.7833
    Epoch 10/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5478 - accuracy: 0.8058 - val_loss: 0.5736 - val_accuracy: 0.7974
    Epoch 11/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5446 - accuracy: 0.8059 - val_loss: 0.5672 - val_accuracy: 0.8016
    Epoch 12/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5432 - accuracy: 0.8067 - val_loss: 0.5773 - val_accuracy: 0.7950
    Epoch 13/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5425 - accuracy: 0.8056 - val_loss: 0.5775 - val_accuracy: 0.7992
    Epoch 14/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5407 - accuracy: 0.8078 - val_loss: 0.5616 - val_accuracy: 0.8075
    Epoch 15/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5408 - accuracy: 0.8052 - val_loss: 0.5773 - val_accuracy: 0.8039
    Epoch 16/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5437 - accuracy: 0.8058 - val_loss: 0.5682 - val_accuracy: 0.8015
    Epoch 17/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5419 - accuracy: 0.8075 - val_loss: 0.5995 - val_accuracy: 0.7964
    Epoch 18/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5488 - accuracy: 0.8058 - val_loss: 0.5544 - val_accuracy: 0.8087
    Epoch 19/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5506 - accuracy: 0.8042 - val_loss: 0.6068 - val_accuracy: 0.7864
    Epoch 20/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5537 - accuracy: 0.8030 - val_loss: 0.5597 - val_accuracy: 0.8076
    Epoch 21/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5572 - accuracy: 0.8036 - val_loss: 0.5998 - val_accuracy: 0.7934
    Epoch 22/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5615 - accuracy: 0.8013 - val_loss: 0.5756 - val_accuracy: 0.8034
    Epoch 23/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5655 - accuracy: 0.8017 - val_loss: 0.6386 - val_accuracy: 0.7668
    Epoch 24/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5819 - accuracy: 0.7963 - val_loss: 0.6356 - val_accuracy: 0.7869
    Epoch 25/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5810 - accuracy: 0.7977 - val_loss: 0.6481 - val_accuracy: 0.7865
    Epoch 26/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5960 - accuracy: 0.7901 - val_loss: 0.6997 - val_accuracy: 0.7802
    Epoch 27/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6101 - accuracy: 0.7870 - val_loss: 0.6124 - val_accuracy: 0.7917
    Epoch 28/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6178 - accuracy: 0.7846 - val_loss: 0.6137 - val_accuracy: 0.7962
    Epoch 29/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6357 - accuracy: 0.7771 - val_loss: 0.6655 - val_accuracy: 0.7621
    Epoch 30/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6671 - accuracy: 0.7678 - val_loss: 0.7597 - val_accuracy: 0.7194
    Epoch 31/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6836 - accuracy: 0.7585 - val_loss: 0.6958 - val_accuracy: 0.7342
    Epoch 32/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.7062 - accuracy: 0.7553 - val_loss: 0.7015 - val_accuracy: 0.7732
    Epoch 33/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.7383 - accuracy: 0.7500 - val_loss: 0.7146 - val_accuracy: 0.7706
    Epoch 34/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.8033 - accuracy: 0.7300 - val_loss: 0.8987 - val_accuracy: 0.6848
    Epoch 35/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.8429 - accuracy: 0.7110 - val_loss: 0.8750 - val_accuracy: 0.7053
    Epoch 36/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.8651 - accuracy: 0.7033 - val_loss: 0.8176 - val_accuracy: 0.6989
    Epoch 37/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.9203 - accuracy: 0.6837 - val_loss: 0.7876 - val_accuracy: 0.7333
    Epoch 38/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.2374 - accuracy: 0.5191 - val_loss: 1.3699 - val_accuracy: 0.4902
    Epoch 39/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.1828 - accuracy: 0.5311 - val_loss: 1.1010 - val_accuracy: 0.5819
    Epoch 40/40
    1875/1875 [==============================] - 3s 2ms/step - loss: 1.6640 - accuracy: 0.3303 - val_loss: 1.8528 - val_accuracy: 0.2779



```python
import numpy as np
import matplotlib.pyplot as plt
lrs = 1e-3 * (10**(np.arange(40)/20))
plt.semilogx(lrs, find_lr_history.history["loss"])
plt.xlabel("earning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate");
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_132_0.png)
    


Bu durumda, Adam optimizer'ın (0.001) varsayılan öğrenme oranına yakın bir yerde ideal öğrenme oranı gibi görünüyor.

İdeal öğrenme oranını kullanarak bir modeli yeniden yerleştirelim.



```python
tf.random.set_seed(42)

# bir model oluşturma
model_14 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax") 
])

# modeli derleme
model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# modeli fit etme
history = model_14.fit(train_data,
                                train_labels,
                                epochs=20,
                                validation_data=(test_data, test_labels),
                                callbacks=[lr_scheduler])
```

    Epoch 1/20
    1875/1875 [==============================] - 4s 2ms/step - loss: 1.0348 - accuracy: 0.6474 - val_loss: 0.6937 - val_accuracy: 0.7617
    Epoch 2/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.6366 - accuracy: 0.7759 - val_loss: 0.6400 - val_accuracy: 0.7808
    Epoch 3/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5934 - accuracy: 0.7911 - val_loss: 0.6278 - val_accuracy: 0.7770
    Epoch 4/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5749 - accuracy: 0.7969 - val_loss: 0.6122 - val_accuracy: 0.7871
    Epoch 5/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5655 - accuracy: 0.7987 - val_loss: 0.6061 - val_accuracy: 0.7913
    Epoch 6/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5569 - accuracy: 0.8022 - val_loss: 0.5917 - val_accuracy: 0.7940
    Epoch 7/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5542 - accuracy: 0.8036 - val_loss: 0.5898 - val_accuracy: 0.7896
    Epoch 8/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5509 - accuracy: 0.8039 - val_loss: 0.5829 - val_accuracy: 0.7949
    Epoch 9/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5468 - accuracy: 0.8047 - val_loss: 0.6036 - val_accuracy: 0.7833
    Epoch 10/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5478 - accuracy: 0.8058 - val_loss: 0.5736 - val_accuracy: 0.7974
    Epoch 11/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5446 - accuracy: 0.8059 - val_loss: 0.5672 - val_accuracy: 0.8016
    Epoch 12/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5432 - accuracy: 0.8067 - val_loss: 0.5773 - val_accuracy: 0.7950
    Epoch 13/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5425 - accuracy: 0.8056 - val_loss: 0.5775 - val_accuracy: 0.7992
    Epoch 14/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5407 - accuracy: 0.8078 - val_loss: 0.5616 - val_accuracy: 0.8075
    Epoch 15/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5408 - accuracy: 0.8052 - val_loss: 0.5773 - val_accuracy: 0.8039
    Epoch 16/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5437 - accuracy: 0.8058 - val_loss: 0.5682 - val_accuracy: 0.8015
    Epoch 17/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5419 - accuracy: 0.8075 - val_loss: 0.5995 - val_accuracy: 0.7964
    Epoch 18/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5488 - accuracy: 0.8058 - val_loss: 0.5544 - val_accuracy: 0.8087
    Epoch 19/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5506 - accuracy: 0.8042 - val_loss: 0.6068 - val_accuracy: 0.7864
    Epoch 20/20
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.5537 - accuracy: 0.8030 - val_loss: 0.5597 - val_accuracy: 0.8076


Şimdi ideale yakın bir öğrenme oranıyla eğitilmiş ve oldukça iyi performans gösteren bir modelimiz var, birkaç seçeneğimiz var.

Yapabiliriz:
- Diğer sınıflandırma ölçütlerini (karışıklık matrisi veya sınıflandırma raporu gibi) kullanarak performansını değerlendirin.
- Tahminlerinden bazılarını değerlendirin (görselleştirmeler aracılığıyla).
- Doğruluğunu artırın (daha uzun süre eğiterek veya mimariyi değiştirerek).
- Bir uygulamada kullanmak üzere kaydedin ve dışa aktarın.
İlk iki seçeneği inceleyelim.

İlk olarak, farklı sınıflardaki tahminlerini görselleştirmek için bir sınıflandırma matrisi oluşturacağız.


```python
import itertools
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15): 

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
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
```


```python
# En son modelle tahminler yapın
y_probs = model_14.predict(test_data)
y_probs[:5]
```




    array([[1.2742422e-09, 5.3467939e-08, 8.3849936e-06, 7.0589194e-06,
            9.7621414e-06, 5.4065306e-02, 7.2848763e-08, 6.3599236e-02,
            1.8366773e-03, 8.8047338e-01],
           [1.2383326e-05, 2.3110788e-17, 9.0172756e-01, 3.3549309e-06,
            5.8316510e-02, 1.4708356e-14, 3.9687403e-02, 3.2329574e-30,
            2.5285641e-04, 3.8686530e-21],
           [1.4126861e-04, 9.8552668e-01, 7.5693160e-06, 1.4010524e-02,
            2.0143854e-04, 9.7611138e-12, 1.0748472e-04, 4.0757726e-08,
            4.5021643e-06, 5.8545976e-07],
           [3.9738816e-06, 9.9383062e-01, 2.1684928e-06, 5.7096859e-03,
            2.5507511e-04, 4.3098709e-11, 1.8117787e-05, 1.1347497e-06,
            2.2362808e-06, 1.7706068e-04],
           [2.8454942e-01, 7.6419547e-06, 1.0144152e-01, 1.9927604e-02,
            3.2220736e-02, 1.3905409e-12, 5.6163126e-01, 1.6594300e-18,
            2.2172352e-04, 2.4364064e-15]], dtype=float32)



Modelimiz bir tahmin olasılıkları listesi verir, yani belirli bir sınıfın etiket olma olasılığının ne kadar muhtemel olduğuna dair bir sayı verir.

Tahmin olasılıkları listesindeki sayı ne kadar yüksekse, modelin doğru sınıf olduğuna inanması o kadar olasıdır.

En yüksek değeri bulmak için argmax() yöntemini kullanabiliriz.



```python
# İlk örnek için öngörülen sınıf numarasına ve etiketine bakın
y_probs[0].argmax(), class_names[y_probs[0].argmax()]
```




    (9, 'Ankle boot')




```python
# Olasılıklardan tüm tahminleri etiketlere dönüştürün
y_preds = y_probs.argmax(axis=1)

# İlk 10 tahmin etiketini görüntüleyin
y_preds[:10]
```




    array([9, 2, 1, 1, 6, 1, 4, 6, 5, 7])



Harika, şimdi modelimizin tahminlerini etiket biçiminde aldık, onları doğruluk etiketlerine karşı görmek için bir karışıklık matrisi oluşturalım.


```python
# Güzelleştirilmemiş karışıklık matrisine göz atın
confusion_matrix(y_true=test_labels, 
                 y_pred=y_preds)
```




    array([[868,   5,  17,  60,   1,   0,  36,   0,  12,   1],
           [  3, 951,   4,  29,   5,   5,   3,   0,   0,   0],
           [ 50,   3, 667,  11, 158,   2,  97,   0,  12,   0],
           [ 86,  18,  11, 824,  20,   1,  27,   2,  11,   0],
           [  6,   1, 116,  48, 728,   3,  93,   0,   5,   0],
           [  0,   2,   0,   0,   0, 862,   0,  78,   9,  49],
           [263,   5, 155,  45, 123,   5, 390,   0,  14,   0],
           [  0,   0,   0,   0,   0,  21,   0, 924,   1,  54],
           [ 15,   1,  35,  16,   3,   4,   5,   5, 916,   0],
           [  0,   3,   0,   0,   1,   8,   0,  39,   3, 946]])




```python
make_confusion_matrix(y_true=test_labels, 
                      y_pred=y_preds,
                      classes=class_names,
                      figsize=(15, 15),
                      text_size=10)
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_143_0.png)
    


Bu çok daha iyi görünüyor. Sonuçların o kadar iyi olmaması dışında... Görünüşe göre modelimizin Gömlek ve T-shirt/üst sınıflar arasında kafası karışıyor.

Bir karışıklık matrisi kullanarak model tahminlerimizin doğruluk etiketleriyle nasıl hizalandığını gördük, peki ya bazılarını görselleştirmeye ne dersiniz?

Tahmini ile birlikte rastgele bir görüntü çizmek için bir fonksiyon oluşturalım.

> 🔑 Not: Genellikle görüntülerle ve diğer görsel veri biçimleriyle çalışırken, verileri ve modelinizin çıktılarını daha iyi anlamak için mümkün olduğunca görselleştirmek iyi bir fikirdir.


```python
 import random

def plot_random_image(model, images, true_labels, classes):
  i = random.randint(0, len(images))
  
  # Tahminler ve hedefleri oluşturalım
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Hedef görüntüyü çizelim
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Tahminin doğru veya yanlış olmasına bağlı olarak başlıkların rengini değiştirelim
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
             color=color)
```


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=test_labels, 
                 y_pred=y_preds)
```




    array([[868,   5,  17,  60,   1,   0,  36,   0,  12,   1],
           [  3, 951,   4,  29,   5,   5,   3,   0,   0,   0],
           [ 50,   3, 667,  11, 158,   2,  97,   0,  12,   0],
           [ 86,  18,  11, 824,  20,   1,  27,   2,  11,   0],
           [  6,   1, 116,  48, 728,   3,  93,   0,   5,   0],
           [  0,   2,   0,   0,   0, 862,   0,  78,   9,  49],
           [263,   5, 155,  45, 123,   5, 390,   0,  14,   0],
           [  0,   0,   0,   0,   0,  21,   0, 924,   1,  54],
           [ 15,   1,  35,  16,   3,   4,   5,   5, 916,   0],
           [  0,   3,   0,   0,   1,   8,   0,  39,   3, 946]])




```python
plot_random_image(model=model_14, 
                  images=test_data, 
                  true_labels=test_labels, 
                  classes=class_names)
```


    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_147_0.png)
    


Yukarıdaki hücreyi birkaç kez çalıştırdıktan sonra, modelin tahminleri ile gerçek etiketler arasındaki ilişkiyi görsel olarak anlamaya başlayacaksınız.

Modelin hangi tahminlerde kafasının karıştığını anladınız mı?

Benzer sınıfları karıştırıyor gibi görünüyor, örneğin Sneaker with Ankle boot. Resimlere baktığınızda bunun nasıl olabileceğini görebilirsiniz. Bir Sneaker ve Ankle Boot'un genel şekli benzerdir. Genel şekil, modelin öğrendiği kalıplardan biri olabilir ve bu nedenle, iki görüntü benzer bir şekle sahip olduğunda, tahminleri karışır.


## Modelimiz Hangi Kalıpları Öğreniyor?

Bir sinir ağının sayılardaki kalıpları nasıl bulduğu hakkında çok konuştuk ama bu kalıplar tam olarak neye benziyor? Modellerimizden birini açıp öğrenelim.

İlk olarak, en son modelimizde (model_14) katmanlar özniteliğini kullanarak katmanların bir listesini alacağız.


```python
# En son modelimizin katmanlarını bulun
model_14.layers
```




    [<tensorflow.python.keras.layers.core.Flatten at 0x7f64a0b61b10>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f64a0b61550>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f64a0af9950>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f64a0acdf50>]



İndeksleme kullanarak bir hedef katmana erişebiliriz.


```python
# Belirli bir katmanı görüntüleme
model_14.layers[1]
```




    <tensorflow.python.keras.layers.core.Dense at 0x7f64a0b61550>



Ve `get_weights()` yöntemini kullanarak belirli bir katman tarafından öğrenilen kalıpları bulabiliriz.

`get_weights()` yöntemi, belirli bir katmanın ağırlıklarını (ağırlık matrisi olarak da bilinir) ve sapmalarını (önyargı vektörü olarak da bilinir) döndürür.



```python
weights, biases = model_14.layers[1].get_weights()
weights, weights.shape
```




    (array([[ 3.0885503 , -2.430857  ,  0.45438388, -3.0628507 ],
            [ 0.98286426, -2.71804   , -0.38760266, -1.1560956 ],
            [ 2.6185486 , -1.6931161 , -2.659585  , -2.343221  ],
            ...,
            [-0.5499583 ,  2.1220326 , -0.22042169,  0.75220233],
            [-0.5888785 ,  3.346401  ,  1.4520893 , -1.5131956 ],
            [ 0.90688974, -0.6245389 ,  0.64969605,  0.05348392]],
           dtype=float32), (784, 4))



Ağırlık matrisi, bizim durumumuzda 784 (28x28 piksel) olan giriş verileriyle aynı şekildedir. Ve seçilen katmandaki her nöron için ağırlık matrisinin bir kopyası var (seçilen katmanımızda 4 nöron var).

Ağırlık matrisindeki her değer, girdi verilerindeki belirli bir değerin ağın kararlarını nasıl etkilediğine karşılık gelir.

Bu değerler rastgele sayılar olarak başlar (bir katman oluştururken kernel_initializer parametresi tarafından ayarlanırlar, varsayılan "glorot_uniform"dur) ve daha sonra eğitim sırasında sinir ağı tarafından verilerin daha iyi temsili değerlerine (rastgele olmayan) güncellenir. 


```python
biases, biases.shape
```




    (array([ 2.1505804 ,  0.45967796, -0.38694024,  2.9040031 ], dtype=float32),
     (4,))



Her nöronun bir önyargı vektörü vardır. Bunların her biri bir ağırlık matrisi ile eşleştirilir. Önyargı değerleri varsayılan olarak sıfır olarak başlatılır (bias_initializer parametresi kullanılarak).

Önyargı vektörü, karşılık gelen ağırlık matrisindeki kalıpların bir sonraki katmanı ne kadar etkilemesi gerektiğini belirler.



```python
model_14.summary()
```

    Model: "sequential_13"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_3 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_35 (Dense)             (None, 4)                 3140      
    _________________________________________________________________
    dense_36 (Dense)             (None, 4)                 20        
    _________________________________________________________________
    dense_37 (Dense)             (None, 10)                50        
    =================================================================
    Total params: 3,210
    Trainable params: 3,210
    Non-trainable params: 0
    _________________________________________________________________


Şimdi birkaç derin öğrenme modeli oluşturduk, tüm girdiler ve çıktılar kavramının yalnızca bir modelin tamamıyla değil, bir model içindeki her katmanla da ilgili olduğunu belirtmenin tam zamanı.

Bunu zaten tahmin etmiş olabilirsiniz, ancak girdi katmanından başlayarak, sonraki her katmanın girdisi bir önceki katmanın çıktısıdır. Bunu `plot_model()` kullanarak açıkça görebiliriz.



```python
from tensorflow.keras.utils import plot_model

plot_model(model_14, show_shapes=True)
```




    
![png](TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_files/TensorFlow_ile_Sinir_A%C4%9F%C4%B1_S%C4%B1n%C4%B1fland%C4%B1r%C4%B1lmas%C4%B1_160_0.png)
    



Bir model nasıl öğrenir (kısaca):

Pekala, bir sürü model eğittik ama kaputun altında neler olduğunu hiç tartışmadık. Peki bir model tam olarak nasıl öğrenir?

Bir model, ağırlık matrislerini ve yanlılık değerlerini her çağda güncelleyerek ve geliştirerek öğrenir (bizim durumumuzda, fit() işlevini çağırdığımızda).

Bunu, veriler ve etiketler arasında öğrendiği kalıpları gerçek etiketlerle karşılaştırarak yapar.

Mevcut modeller (ağırlık matrisleri ve yanlılık değerleri) kayıp fonksiyonunda istenen bir azalmaya neden olmazsa (daha yüksek kayıp daha kötü tahminler anlamına gelir), optimize edici modeli, modellerini doğru şekilde güncellemek için yönlendirmeye çalışır (gerçek kullanarak referans olarak etiketler).

Modelin tahminlerini geliştirmek için gerçek etiketleri referans olarak kullanma sürecine geri yayılım (backpropagation) denir. Başka bir deyişle, veriler ve etiketler bir modelden geçer (ileri geçiş) ve veriler ile etiketler arasındaki ilişkiyi öğrenmeye çalışır. Ve eğer bu öğrenilen ilişki gerçek ilişkiye yakın değilse veya geliştirilebilirse, model bunu kendi içinden geçerek (geriye geçiş) ve verileri daha iyi temsil etmek için ağırlık matrislerini ve önyargı değerlerini değiştirerek yapar.
