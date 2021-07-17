# Evrişimsel Sinir Ağları (CNN)

Evrişimli sinir ağları görüntülerle çok iyi çalıştığından, onlar hakkında daha fazla bilgi edinmek için bir görüntü veri kümesiyle başlayacağız. Çalışacağımız görseller, 101.001 görselden, 101 farklı kategoriden oluştan [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) veri setindenden alınmıştır.

Başlangıç için, kategorilerden sadece ikisini kullanacağız: pizza ve biftek.

> 🔑 Not: Kullandığımız verileri hazırlamak için görüntüleri farklı alt küme klasörlerine taşıma gibi ön işleme adımları yapılmıştır.


```python
# google drive dosyalarına erişmek için yetki istiyoruz
from google.colab import drive
drive.mount("/content/gdrive")
```

    Mounted at /content/gdrive



```python
# zip şeklinde olan dosyayı unzipliyoruz
import zipfile

# zip'in path adresi
zir_path = "/content/gdrive/MyDrive/Colab Notebooks/TensorFlow Developer Certificate/eğitimler/pizza_steak.zip"
zip_ref = zipfile.ZipFile(zir_path, "r")
zip_ref.extractall()
zip_ref.close()
```

> Not: Google Colab kullanıyorsanız ve çalışma zamanınızın bağlantısı kesilirse, dosyaları yeniden indirmeniz gerekebilir. Bunu, yukarıdaki hücreyi yeniden çalıştırarak yapabilirsiniz.

## Verileri İnceleyin

Herhangi bir makine öğrenimi projesinin başlangıcında çok önemli bir adım, verilerle bir olmaktır. Bu genellikle, üzerinde çalıştığınız verileri anlamak için bol miktarda görselleştirme ve klasör taraması anlamına gelir. Bunu söylerken, az önce indirdiğimiz verileri inceleyelim.

Dosya yapısı, resimlerle çalışmak için kullanabileceğiniz tipik bir biçimde biçimlendirilmiştir.

Daha spesifik olarak:

Eğitim veri kümesindeki tüm görüntüleri içeren ve her biri o sınıfın görüntülerini içeren belirli bir sınıftan sonra adlandırılan alt dizinleri içeren bir train dizini.

Train dizini ile aynı yapıya sahip bir test dizini.


```
pizza_steak 
└───train 
│   └───pizza
│   │   │   1008104.jpg
│   │   │   1638227.jpg
│   │   │   ...      
│   └───steak
│       │   1000205.jpg
│       │   1647351.jpg
│       │   ...
│   
└───test 
│   └───pizza
│   │   │   1001116.jpg
│   │   │   1507019.jpg
│   │   │   ...      
│   └───steak
│       │   100274.jpg
│       │   1653815.jpg
│       │   ...    
```

İndirdiğimiz dizinlerin her birini inceleyelim. Bunu yapmak için, liste anlamına gelen `ls` komutunu kullanabiliriz.



```python
!ls pizza_steak
```

    test  train


Train ve test klasörlerimizi görüyoruz. Bakalım içlerinde ne varmış.


```python
!ls pizza_steak/train/
```

    pizza  steak


Peki ya steak içerisinde ne var?


```python
!ls pizza_steak/train/steak/
```

    1000205.jpg  1647351.jpg  2238681.jpg  2824680.jpg  3375959.jpg  417368.jpg
    100135.jpg   1650002.jpg  2238802.jpg  2825100.jpg  3381560.jpg  4176.jpg
    101312.jpg   165639.jpg   2254705.jpg  2826987.jpg  3382936.jpg  42125.jpg
    1021458.jpg  1658186.jpg  225990.jpg   2832499.jpg  3386119.jpg  421476.jpg
    1032846.jpg  1658443.jpg  2260231.jpg  2832960.jpg  3388717.jpg  421561.jpg
    10380.jpg    165964.jpg   2268692.jpg  285045.jpg   3389138.jpg  438871.jpg
    1049459.jpg  167069.jpg   2271133.jpg  285147.jpg   3393547.jpg  43924.jpg
    1053665.jpg  1675632.jpg  227576.jpg   2855315.jpg  3393688.jpg  440188.jpg
    1068516.jpg  1678108.jpg  2283057.jpg  2856066.jpg  3396589.jpg  442757.jpg
    1068975.jpg  168006.jpg   2286639.jpg  2859933.jpg  339891.jpg	 443210.jpg
    1081258.jpg  1682496.jpg  2287136.jpg  286219.jpg   3417789.jpg  444064.jpg
    1090122.jpg  1684438.jpg  2291292.jpg  2862562.jpg  3425047.jpg  444709.jpg
    1093966.jpg  168775.jpg   229323.jpg   2865730.jpg  3434983.jpg  447557.jpg
    1098844.jpg  1697339.jpg  2300534.jpg  2878151.jpg  3435358.jpg  461187.jpg
    1100074.jpg  1710569.jpg  2300845.jpg  2880035.jpg  3438319.jpg  461689.jpg
    1105280.jpg  1714605.jpg  231296.jpg   2881783.jpg  3444407.jpg  465494.jpg
    1117936.jpg  1724387.jpg  2315295.jpg  2884233.jpg  345734.jpg	 468384.jpg
    1126126.jpg  1724717.jpg  2323132.jpg  2890573.jpg  3460673.jpg  477486.jpg
    114601.jpg   172936.jpg   2324994.jpg  2893832.jpg  3465327.jpg  482022.jpg
    1147047.jpg  1736543.jpg  2327701.jpg  2893892.jpg  3466159.jpg  482465.jpg
    1147883.jpg  1736968.jpg  2331076.jpg  2907177.jpg  3469024.jpg  483788.jpg
    1155665.jpg  1746626.jpg  233964.jpg   290850.jpg   3470083.jpg  493029.jpg
    1163977.jpg  1752330.jpg  2344227.jpg  2909031.jpg  3476564.jpg  503589.jpg
    1190233.jpg  1761285.jpg  234626.jpg   2910418.jpg  3478318.jpg  510757.jpg
    1208405.jpg  176508.jpg   234704.jpg   2912290.jpg  3488748.jpg  513129.jpg
    1209120.jpg  1772039.jpg  2357281.jpg  2916448.jpg  3492328.jpg  513842.jpg
    1212161.jpg  1777107.jpg  2361812.jpg  2916967.jpg  3518960.jpg  523535.jpg
    1213988.jpg  1787505.jpg  2365287.jpg  2927833.jpg  3522209.jpg  525041.jpg
    1219039.jpg  179293.jpg   2374582.jpg  2928643.jpg  3524429.jpg  534560.jpg
    1225762.jpg  1816235.jpg  239025.jpg   2929179.jpg  3528458.jpg  534633.jpg
    1230968.jpg  1822407.jpg  2390628.jpg  2936477.jpg  3531805.jpg  536535.jpg
    1236155.jpg  1823263.jpg  2392910.jpg  2938012.jpg  3536023.jpg  541410.jpg
    1241193.jpg  1826066.jpg  2394465.jpg  2938151.jpg  3538682.jpg  543691.jpg
    1248337.jpg  1828502.jpg  2395127.jpg  2939678.jpg  3540750.jpg  560503.jpg
    1257104.jpg  1828969.jpg  2396291.jpg  2940544.jpg  354329.jpg	 561972.jpg
    126345.jpg   1829045.jpg  2400975.jpg  2940621.jpg  3547166.jpg  56240.jpg
    1264050.jpg  1829088.jpg  2403776.jpg  2949079.jpg  3553911.jpg  56409.jpg
    1264154.jpg  1836332.jpg  2403907.jpg  295491.jpg   3556871.jpg  564530.jpg
    1264858.jpg  1839025.jpg  240435.jpg   296268.jpg   355715.jpg	 568972.jpg
    127029.jpg   1839481.jpg  2404695.jpg  2964732.jpg  356234.jpg	 576725.jpg
    1289900.jpg  183995.jpg   2404884.jpg  2965021.jpg  3571963.jpg  588739.jpg
    1290362.jpg  184110.jpg   2407770.jpg  2966859.jpg  3576078.jpg  590142.jpg
    1295457.jpg  184226.jpg   2412263.jpg  2977966.jpg  3577618.jpg  60633.jpg
    1312841.jpg  1846706.jpg  2425062.jpg  2979061.jpg  3577732.jpg  60655.jpg
    1313316.jpg  1849364.jpg  2425389.jpg  2983260.jpg  3578934.jpg  606820.jpg
    1324791.jpg  1849463.jpg  2435316.jpg  2984311.jpg  358042.jpg	 612551.jpg
    1327567.jpg  1849542.jpg  2437268.jpg  2988960.jpg  358045.jpg	 614975.jpg
    1327667.jpg  1853564.jpg  2437843.jpg  2989882.jpg  3591821.jpg  616809.jpg
    1333055.jpg  1869467.jpg  2440131.jpg  2995169.jpg  359330.jpg	 628628.jpg
    1334054.jpg  1870942.jpg  2443168.jpg  2996324.jpg  3601483.jpg  632427.jpg
    1335556.jpg  187303.jpg   2446660.jpg  3000131.jpg  3606642.jpg  636594.jpg
    1337814.jpg  187521.jpg   2455944.jpg  3002350.jpg  3609394.jpg  637374.jpg
    1340977.jpg  1888450.jpg  2458401.jpg  3007772.jpg  361067.jpg	 640539.jpg
    1343209.jpg  1889336.jpg  2487306.jpg  3008192.jpg  3613455.jpg  644777.jpg
    134369.jpg   1907039.jpg  248841.jpg   3009617.jpg  3621464.jpg  644867.jpg
    1344105.jpg  1925230.jpg  2489716.jpg  3011642.jpg  3621562.jpg  658189.jpg
    134598.jpg   1927984.jpg  2490489.jpg  3020591.jpg  3621565.jpg  660900.jpg
    1346387.jpg  1930577.jpg  2495884.jpg  3030578.jpg  3623556.jpg  663014.jpg
    1348047.jpg  1937872.jpg  2495903.jpg  3047807.jpg  3640915.jpg  664545.jpg
    1351372.jpg  1941807.jpg  2499364.jpg  3059843.jpg  3643951.jpg  667075.jpg
    1362989.jpg  1942333.jpg  2500292.jpg  3074367.jpg  3653129.jpg  669180.jpg
    1367035.jpg  1945132.jpg  2509017.jpg  3082120.jpg  3656752.jpg  669960.jpg
    1371177.jpg  1961025.jpg  250978.jpg   3094354.jpg  3663518.jpg  6709.jpg
    1375640.jpg  1966300.jpg  2514432.jpg  3095301.jpg  3663800.jpg  674001.jpg
    1382427.jpg  1966967.jpg  2526838.jpg  3099645.jpg  3664376.jpg  676189.jpg
    1392718.jpg  1969596.jpg  252858.jpg   3100476.jpg  3670607.jpg  681609.jpg
    1395906.jpg  1971757.jpg  2532239.jpg  3110387.jpg  3671021.jpg  6926.jpg
    1400760.jpg  1976160.jpg  2534567.jpg  3113772.jpg  3671877.jpg  703556.jpg
    1403005.jpg  1984271.jpg  2535431.jpg  3116018.jpg  368073.jpg	 703909.jpg
    1404770.jpg  1987213.jpg  2535456.jpg  3128952.jpg  368162.jpg	 704316.jpg
    140832.jpg   1987639.jpg  2538000.jpg  3130412.jpg  368170.jpg	 714298.jpg
    141056.jpg   1995118.jpg  2543081.jpg  3136.jpg     3693649.jpg  720060.jpg
    141135.jpg   1995252.jpg  2544643.jpg  313851.jpg   3700079.jpg  726083.jpg
    1413972.jpg  199754.jpg   2547797.jpg  3140083.jpg  3704103.jpg  728020.jpg
    1421393.jpg  2002400.jpg  2548974.jpg  3140147.jpg  3707493.jpg  732986.jpg
    1428947.jpg  2011264.jpg  2549316.jpg  3142045.jpg  3716881.jpg  734445.jpg
    1433912.jpg  2012996.jpg  2561199.jpg  3142618.jpg  3724677.jpg  735441.jpg
    143490.jpg   2013535.jpg  2563233.jpg  3142674.jpg  3727036.jpg  740090.jpg
    1445352.jpg  2017387.jpg  256592.jpg   3143192.jpg  3727491.jpg  745189.jpg
    1446401.jpg  2018173.jpg  2568848.jpg  314359.jpg   3736065.jpg  752203.jpg
    1453991.jpg  2020613.jpg  2573392.jpg  3157832.jpg  37384.jpg	 75537.jpg
    1456841.jpg  2032669.jpg  2592401.jpg  3159818.jpg  3743286.jpg  756655.jpg
    146833.jpg   203450.jpg   2599817.jpg  3162376.jpg  3745515.jpg  762210.jpg
    1476404.jpg  2034628.jpg  2603058.jpg  3168620.jpg  3750472.jpg  763690.jpg
    1485083.jpg  2036920.jpg  2606444.jpg  3171085.jpg  3752362.jpg  767442.jpg
    1487113.jpg  2038418.jpg  2614189.jpg  317206.jpg   3766099.jpg  786409.jpg
    148916.jpg   2042975.jpg  2614649.jpg  3173444.jpg  3770370.jpg  80215.jpg
    149087.jpg   2045647.jpg  2615718.jpg  3180182.jpg  377190.jpg	 802348.jpg
    1493169.jpg  2050584.jpg  2619625.jpg  31881.jpg    3777020.jpg  804684.jpg
    149682.jpg   2052542.jpg  2622140.jpg  3191589.jpg  3777482.jpg  812163.jpg
    1508094.jpg  2056627.jpg  262321.jpg   3204977.jpg  3781152.jpg  813486.jpg
    1512226.jpg  2062248.jpg  2625330.jpg  320658.jpg   3787809.jpg  819027.jpg
    1512347.jpg  2081995.jpg  2628106.jpg  3209173.jpg  3788729.jpg  822550.jpg
    1524526.jpg  2087958.jpg  2629750.jpg  3223400.jpg  3790962.jpg  823766.jpg
    1530833.jpg  2088030.jpg  2643906.jpg  3223601.jpg  3792514.jpg  827764.jpg
    1539499.jpg  2088195.jpg  2644457.jpg  3241894.jpg  379737.jpg	 830007.jpg
    1541672.jpg  2090493.jpg  2648423.jpg  3245533.jpg  3807440.jpg  838344.jpg
    1548239.jpg  2090504.jpg  2651300.jpg  3245622.jpg  381162.jpg	 853327.jpg
    1550997.jpg  2125877.jpg  2653594.jpg  3247009.jpg  3812039.jpg  854150.jpg
    1552530.jpg  2129685.jpg  2661577.jpg  3253588.jpg  3829392.jpg  864997.jpg
    15580.jpg    2133717.jpg  2668916.jpg  3260624.jpg  3830872.jpg  885571.jpg
    1559052.jpg  2136662.jpg  268444.jpg   326587.jpg   38442.jpg	 907107.jpg
    1563266.jpg  213765.jpg   2691461.jpg  32693.jpg    3855584.jpg  908261.jpg
    1567554.jpg  2138335.jpg  2706403.jpg  3271253.jpg  3857508.jpg  910672.jpg
    1575322.jpg  2140776.jpg  270687.jpg   3274423.jpg  386335.jpg	 911803.jpg
    1588879.jpg  214320.jpg   2707522.jpg  3280453.jpg  3867460.jpg  91432.jpg
    1594719.jpg  2146963.jpg  2711806.jpg  3298495.jpg  3868959.jpg  914570.jpg
    1595869.jpg  215222.jpg   2716993.jpg  330182.jpg   3869679.jpg  922752.jpg
    1598345.jpg  2154126.jpg  2724554.jpg  3306627.jpg  388776.jpg	 923772.jpg
    1598885.jpg  2154779.jpg  2738227.jpg  3315727.jpg  3890465.jpg  926414.jpg
    1600179.jpg  2159975.jpg  2748917.jpg  331860.jpg   3894222.jpg  931356.jpg
    1600794.jpg  2163079.jpg  2760475.jpg  332232.jpg   3895825.jpg  937133.jpg
    160552.jpg   217250.jpg   2761427.jpg  3322909.jpg  389739.jpg	 945791.jpg
    1606596.jpg  2172600.jpg  2765887.jpg  332557.jpg   3916407.jpg  947877.jpg
    1615395.jpg  2173084.jpg  2768451.jpg  3326734.jpg  393349.jpg	 952407.jpg
    1618011.jpg  217996.jpg   2771149.jpg  3330642.jpg  393494.jpg	 952437.jpg
    1619357.jpg  2193684.jpg  2779040.jpg  3333128.jpg  398288.jpg	 955466.jpg
    1621763.jpg  220341.jpg   2788312.jpg  3333735.jpg  40094.jpg	 9555.jpg
    1623325.jpg  22080.jpg	  2788759.jpg  3334973.jpg  401094.jpg	 961341.jpg
    1624450.jpg  2216146.jpg  2796102.jpg  3335013.jpg  401144.jpg	 97656.jpg
    1624747.jpg  2222018.jpg  280284.jpg   3335267.jpg  401651.jpg	 979110.jpg
    1628861.jpg  2223787.jpg  2807888.jpg  3346787.jpg  405173.jpg	 980247.jpg
    1632774.jpg  2230959.jpg  2815172.jpg  3364420.jpg  405794.jpg	 982988.jpg
    1636831.jpg  2232310.jpg  2818805.jpg  336637.jpg   40762.jpg	 987732.jpg
    1645470.jpg  2233395.jpg  2823872.jpg  3372616.jpg  413325.jpg	 996684.jpg


En sevdiğimiz: Bol bol veri :) Çok görüntü var ama ben yine de bunların kesin bir miktarını istiyorum. Görüntülerin saysısını bulalım.


```python
import os

for dirpath, dirnames, filenames in os.walk("pizza_steak"):
  print(f"'{dirpath}' klasöründe {len(filenames)} veri var.")
```

    'pizza_steak' klasöründe 1 veri var.
    'pizza_steak/test' klasöründe 1 veri var.
    'pizza_steak/test/pizza' klasöründe 250 veri var.
    'pizza_steak/test/steak' klasöründe 250 veri var.
    'pizza_steak/train' klasöründe 1 veri var.
    'pizza_steak/train/pizza' klasöründe 750 veri var.
    'pizza_steak/train/steak' klasöründe 750 veri var.



```python
# Bir dosyada kaç tane resim olduğunu bulmanın başka bir yolu
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
num_steak_images_train
```




    750




```python
# Class adlarını alalım
import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
```

    ['.DS_Store' 'pizza' 'steak']


Pekala, elimizde 750 train görseli ve 250 adet pizza ve biftek görseli içeren bir verisetimiz var.

Bazılarına bakalım.

> 🤔 Not: Verilerle çalışırken, mümkün olduğunca görselleştirmek her zaman iyidir. Bir projenin ilk birkaç adımını verilerle bir bütün olarak ele alın. Görselleştirin, görselleştirin, görselleştirin.


```python
# Bir resmi görüntüleyelim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class

  # random bir görsel path'i
  random_image = random.sample(os.listdir(target_folder), 1)

  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
```


```python
img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="steak")
```

    Image shape: (512, 512, 3)



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_17_1.png)
    


Farklı sınıflardan bir düzine kadar görüntüyü inceledikten sonra, neyle çalıştığımız hakkında bir fikir edinmeye başlayabilirsiniz. Food101 veri setinin tamamı, 101 farklı sınıftan benzer görüntülerden oluşuyor. Görüntü şeklini, çizilen görüntünün yanına yazdırdığımızı fark etmiş olabilirsiniz. Bunun nedeni, bilgisayarımızın görüntüyü görme biçiminin büyük bir dizi (tensör) biçiminde olmasıdır.



```python
img
```




    array([[[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]],
    
           [[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]],
    
           [[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]],
    
           ...,
    
           [[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]],
    
           [[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]],
    
           [[0, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
            ...,
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 2]]], dtype=uint8)




```python
# görüntünün şeklini yazdıralım
img.shape
```




    (512, 512, 3)



Görüntü şekline daha yakından baktığınızda, formda olduğunu göreceksiniz (Genişlik, Yükseklik, Renk Kanalları).

Bizim durumumuzda genişlik ve yükseklik değişkendir ancak renkli görüntülerle uğraştığımız için renk kanalları değeri her zaman 3'tür. Bu, farklı kırmızı, yeşil ve mavi (RGB) piksel değerleri içindir.

img dizisindeki tüm değerlerin 0 ile 255 arasında olduğunu fark edeceksiniz. Bunun nedeni, kırmızı, yeşil ve mavi değerlerin olası aralığının bu olmasıdır.

Örneğin, kırmızı=0, yeşil=0, mavi=255 değerine sahip bir piksel mavi görünecektir.

> 🔑 Not: Daha önce tartıştığımız gibi, sinir ağları dahil birçok makine öğrenimi modeli, birlikte çalıştıkları değerlerin 0 ile 1 arasında olmasını tercih eder. Bunu bilerek, görüntülerle çalışmak için en yaygın ön işleme adımlarından biri ölçeklendirmektir (ayrıca görüntü dizilerini 255'e bölerek piksel değerlerini normalleştirme olarak adlandırılır.


```python
# 0 ve 1 arasındaki tüm piksel değerlerini alın
img/255. 
```




    array([[[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]],
    
           [[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]],
    
           [[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]],
    
           ...,
    
           [[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]],
    
           [[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]],
    
           [[0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            ...,
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314],
            [0.        , 0.        , 0.00784314]]])



## Bir Evrişimsel Sinir Ağının Mimarisi

CNN, birçok farklı şekilde oluşturulabilmeleri nedeniyle diğer derin öğrenme sinir ağlarından farklı değildir. Aşağıda gördükleriniz, geleneksel bir CNN'de bulmayı umduğunuz bazı bileşenlerdir.

Evrişimli bir sinir ağının bileşenleri:
- **Input Image** <br>
Kalıpları keşfetmek istediğiniz görüntüleri hedefleyin
- **Input Layer** <br>
Hedef görüntüleri alır ve daha sonraki katmanlar için önceden işler
- **Convolution Layer** <br>
Hedef görüntülerden en önemli özellikleri çıkarır/öğrenir
- **Hidden Activation** <br>
Öğrenilen özelliklere doğrusal olmayanlara ekler (düz olmayan çizgiler)
- **Pooling layer** <br>
Öğrenilmiş görüntü özelliklerinin boyutsallığını eğitir
- **Fully Connected Layer** <br>
Evrişim katmanlarından öğrenilen özellikleri daha da iyileştirir
- **Output layer** <br>
Öğrenilen özellikleri alır ve bunları hedef etiketler şeklinde verir
- **Output activation** <br>
Çıktı katmanına doğrusal olmayanlar ekler


### Örnek

Verilerimizi inceledik ve sınıf başına 750 train resminin yanı sıra 250 test resmi olduğunu ve hepsinin farklı şekillerde olduğunu gördük.

Doğrudan derinlere atlamanın zamanı geldi.

Orijinal veri seti yazarlarının makalesini okuduğumuzda, bir Random Forest makine öğrenme modeli kullandıklarını ve içlerinde hangi farklı yiyeceklerin farklı görüntülere sahip olduğunu tahmin etmede ortalama %50,76 doğruluk elde ettiklerini görüyoruz.

Şu andan itibaren, bu %50,76 bizim temelimiz olacak.

> 🔑 Not: Temel, denemek ve geçmek istediğiniz bir puan veya değerlendirme metriğidir. Genellikle basit bir modelle başlayacak, bir temel oluşturacak ve modelin karmaşıklığını artırarak onu yenmeye çalışacaksınız. Makine öğrenimini öğrenmenin gerçekten eğlenceli bir yolu, sonuçları yayınlanmış bir tür modeli yenmeye çalışmaktır.

Aşağıdaki hücredeki kod, pizza ve biftek veri setimizi yukarıda listelenen bileşenleri kullanarak bir evrişimsel sinir ağı (CNN) ile modellemek için uçtan uca bir şekilde çoğalır.

Anlamayacağınız yerler olabilir fakat kodu iyice analiz ettiken sonra ne yapmak istediğimi  anlayacağınızdan eminim. 

> 📖 Kaynak: Aşağıda kullandığımız mimari, 2014 ImageNet sınıflandırma yarışmasında 2. olan evrişimli bir sinir ağı olan VGG-16'nın küçültülmüş bir versiyonudur.


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)

# Ön işleme verileri (1 ile 0 arasındaki tüm piksel değerlerini alın, 
# ayrıca ölçekleme/normalleştirme olarak da adlandırılır)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Train ve test dizinlerini kurun
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

# Dizinlerdeki verileri içe aktarın
train_data = train_datagen.flow_from_directory(train_dir,
                                               # bir seferde işlenecek görüntü sayısı
                                               batch_size=32, 
                                               # tüm görüntüleri 224 x 224'e dönüştür
                                               target_size=(224, 224), 
                                               # üzerinde çalıştığımız problemin türü
                                               class_mode="binary", 
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

# CNN modeli yaratma (https://poloclub.github.io/cnn-explainer/)
model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10, 
                         kernel_size=3,
                         activation="relu", 
                         input_shape=(224, 224, 3)), 
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size ayrıca (2, 2) olabilir
                            padding="valid"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), 
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary aktivasyon çıktısı
])

# modeli derleme
model_1.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# modeli fit etme
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
```

    Found 1500 images belonging to 2 classes.
    Found 500 images belonging to 2 classes.
    Epoch 1/5
    47/47 [==============================] - 41s 200ms/step - loss: 0.5977 - accuracy: 0.6667 - val_loss: 0.4103 - val_accuracy: 0.8240
    Epoch 2/5
    47/47 [==============================] - 9s 186ms/step - loss: 0.4253 - accuracy: 0.8147 - val_loss: 0.3645 - val_accuracy: 0.8420
    Epoch 3/5
    47/47 [==============================] - 9s 189ms/step - loss: 0.3926 - accuracy: 0.8327 - val_loss: 0.3142 - val_accuracy: 0.8680
    Epoch 4/5
    47/47 [==============================] - 9s 189ms/step - loss: 0.3635 - accuracy: 0.8473 - val_loss: 0.3409 - val_accuracy: 0.8500
    Epoch 5/5
    47/47 [==============================] - 9s 194ms/step - loss: 0.3058 - accuracy: 0.8740 - val_loss: 0.2852 - val_accuracy: 0.8880


> 🤔 Not: Yukarıdaki hücrenin çalışması epoch başına ~12 saniyeden uzun sürüyorsa GPU hızlandırıcı kullanmıyor olabilirsiniz. Colab dizüstü bilgisayar kullanıyorsanız, Çalışma Zamanı -> Çalışma Zamanı Türünü Değiştir -> Donanım Hızlandırıcı'ya gidip "GPU"yu seçerek bir GPU hızlandırıcıya erişebilirsiniz. Bunu yaptıktan sonra, çalışma zamanı türünü değiştirmek Colab'ın sıfırlanmasına neden olacağından yukarıdaki hücrelerin tümünü yeniden çalıştırmanız gerekebilir.

Güzel! 5 epoch sonra modelimiz %50,76 doğruluk temel puanını geçti (modelimiz train setinde ~%85 doğruluk ve test setinde ~%85 doğruluk elde etti).

Ancak, modelimiz Food101 veri setindeki 101 sınıfın tümü yerine yalnızca ikili sınıflandırma probleminden geçti, bu nedenle bu ölçümleri doğrudan karşılaştıramıyoruz. Bununla birlikte, şu ana kadarki sonuçlar modelimizin bir şeyler öğrendiğini gösteriyor.

> 🛠 Alıştırma: Yukarıdaki hücredeki ana kod bloklarının her birinin üzerinden geçin, her birinin ne yaptığını düşünüyorsunuz? Emin değilseniz sorun değil, bunu yakında halledeceğiz. Bu arada, [CNN açıklayıcı web](https://poloclub.github.io/cnn-explainer/) sitesinde 10 dakika oynayarak vakit geçirin. Web sayfasının üst kısmındaki katman adları hakkında ne fark ediyorsunuz?

Halihazırda bir modeli fit ettiğimize göre, mimarisine bir göz atalım.


```python
model_1.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 222, 222, 10)      280       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 220, 220, 10)      910       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 110, 110, 10)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 108, 108, 10)      910       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 106, 106, 10)      910       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 53, 53, 10)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 28090)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 28091     
    =================================================================
    Total params: 31,101
    Trainable params: 31,101
    Non-trainable params: 0
    _________________________________________________________________


Model_1 katmanlarının adları ve CNN açıklayıcı web sitesinin en üstündeki katman adları hakkında ne fark ediyorsunuz?

Size küçük bir sır vereyim: Model demoları için kullandıkları mimariyi aynen kopyaladık.

Şimdi burada anlatmadığımız birkaç yeni şey var:
- **ImageDataGenerator** sınıfı ve yeniden ölçeklendirme parametresi
- **flow_from_directory()** yöntemi
- **batch_size** parametresi
- **target_size** parametresi
- **Conv2D katmanları** (ve bunlarla birlikte gelen parametreler)
- **MaxPool2D katmanları** (ve parametreleri).
- fit() işlevindeki **step_per_epoch** ve **validation_steps** parametreleri

Bunların her birine derinlemesine işlemeden önce, daha önce üzerinde çalıştığımız bir modeli verilerimizle fit etmeye çalışırsak ne olacağını görelim.

## Daha Önce Olduğu Gibi Aynı Modeli Kullanma

Sinir ağlarının birçok farklı soruna nasıl uyarlanabileceğini örneklemek için, daha önce oluşturduğumuz bir ikili sınıflandırma modelinin verilerimizle nasıl çalışabileceğini görelim.

İki şeyi değiştirmek dışında önceki modelimizde aynı parametrelerin hepsini kullanabiliriz:

- **Veriler** <br>
Artık noktalar yerine resimlerle çalışıyoruz.
- **Input Shape** <br>
Sinir ağımıza üzerinde çalıştığımız görüntülerin şeklini söylemeliyiz. <br>
Yaygın bir uygulama, görüntüleri tek bir boyuta yeniden şekillendirmektir. Bizim durumumuzda, görüntüleri (224, 224, 3) olarak yeniden boyutlandıracağız; bu, kırmızı, yeşil, mavi renk kanalları için 224 piksel yükseklik ve genişlik ve 3 derinlik anlamına gelir.


```python
tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# modeli derleme
model_2.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

#  modeli fit etme
history_2 = model_2.fit(train_data,
                        # yukarıda oluşturulan eğitim verilerinin aynısını kullanıyoruz
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        # yukarıda oluşturulan aynı doğrulama verilerini kullanıyoruz
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
```

    Epoch 1/5
    47/47 [==============================] - 9s 189ms/step - loss: 1.0323 - accuracy: 0.5027 - val_loss: 0.6932 - val_accuracy: 0.5000
    Epoch 2/5
    47/47 [==============================] - 9s 184ms/step - loss: 0.6933 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.5000
    Epoch 3/5
    47/47 [==============================] - 8s 173ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.5000
    Epoch 4/5
    47/47 [==============================] - 8s 172ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.5000
    Epoch 5/5
    47/47 [==============================] - 8s 172ms/step - loss: 0.6933 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.5000


Hmmm... modelimiz çalıştı ama hiçbir şey öğrenmiş gibi görünmüyor.

Mimariyi görelim.



```python
model_2.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 150528)            0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 602116    
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 20        
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 5         
    =================================================================
    Total params: 602,141
    Trainable params: 602,141
    Non-trainable params: 0
    _________________________________________________________________


Vay. Buradaki en dikkat çekici şeylerden biri, model_2'ye kıyasla model_1'deki çok daha fazla parametre sayısıdır.

model_2, 602.141 eğitilebilir parametreye sahipken, model_1 yalnızca 31.101'e sahiptir. Ve bu farklılığa rağmen, model_1 hala model_2'ye göre daha performanslı sonuç üretir.

> 🔑 Not: Eğitilebilir parametreleri, bir modelin verilerden öğrenebileceği kalıplar olarak düşünebilirsiniz. Sezgisel olarak, daha fazlasının daha iyi olduğunu düşünebilirsiniz. Ve bazı durumlarda öyle. Ancak bu durumda, buradaki fark, kullandığımız iki farklı model stilindedir. Bir dizi yoğun katman birbirine bağlı bir dizi farklı öğrenilebilir parametreye ve dolayısıyla daha fazla sayıda olası öğrenilebilir örüntüye sahip olduğunda, evrişimli bir sinir ağı bir görüntüdeki en önemli örüntüleri ayırmaya ve öğrenmeye çalışır. Dolayısıyla, evrişimli sinir ağımızda daha az öğrenilebilir parametreler olsa da, bunlar genellikle bir görüntüdeki farklı özellikler arasında şifre çözmede daha faydalıdır.

Önceki modelimiz çalışmadığına göre, onu nasıl çalıştırabileceğimize dair bir fikriniz var mı? Katman sayısını artırmaya ne dersiniz? Ve belki de her katmandaki nöron sayısını artırabilir mi?

Daha spesifik olarak, her yoğun katmandaki nöron sayısını (gizli birimler olarak da adlandırılır) 4'ten 100'e çıkaracağız ve fazladan bir katman ekleyeceğiz.

🔑 Not: Fazladan katman eklemek veya her katmandaki nöron sayısını artırmak, genellikle modelinizin karmaşıklığını artırmak olarak adlandırılır.


```python
tf.random.set_seed(42)

# model yaratma
model_3 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
  tf.keras.layers.Dense(100, activation='relu'),# nöron sayısını 4'ten 100'e çıkarıyoruz (her katman için)
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'), # ekstradan bir katman ekliyoruz
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# modeli derleme
model_3.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

#  modeli fit etme
history_3 = model_3.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
```

    Epoch 1/5
    47/47 [==============================] - 9s 176ms/step - loss: 4.9315 - accuracy: 0.6160 - val_loss: 0.5567 - val_accuracy: 0.7120
    Epoch 2/5
    47/47 [==============================] - 8s 171ms/step - loss: 0.7969 - accuracy: 0.6973 - val_loss: 0.5401 - val_accuracy: 0.7260
    Epoch 3/5
    47/47 [==============================] - 9s 183ms/step - loss: 0.6629 - accuracy: 0.7220 - val_loss: 0.4935 - val_accuracy: 0.7840
    Epoch 4/5
    47/47 [==============================] - 9s 185ms/step - loss: 0.5487 - accuracy: 0.7660 - val_loss: 0.4388 - val_accuracy: 0.7760
    Epoch 5/5
    47/47 [==============================] - 9s 184ms/step - loss: 0.4576 - accuracy: 0.7947 - val_loss: 0.4313 - val_accuracy: 0.7920


Vay! Görünüşe göre modelimiz yeniden öğreniyor. Eğitim setinde ~%70 doğruluk ve doğrulama setinde ~%70 doğruluk elde etti.

Mimari nasıl görünüyor?


```python
model_3.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 150528)            0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 100)               15052900  
    _________________________________________________________________
    dense_5 (Dense)              (None, 100)               10100     
    _________________________________________________________________
    dense_6 (Dense)              (None, 100)               10100     
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 15,073,201
    Trainable params: 15,073,201
    Non-trainable params: 0
    _________________________________________________________________


Eğitilebilir parametrelerin sayısı model_2'den bile daha fazla arttı. Ve 500 kata yakın (~15.000.000 vs. ~31.000) daha fazla eğitilebilir parametreyle bile, model_3 hala model_1'i geçemiyor.

Bu, evrişimli sinir ağlarının gücünü ve daha az parametre kullanmasına rağmen kalıpları öğrenme yeteneklerini gösteriyor.

## İkili sınıflandırma: Modelde Derinlemesine Çalışalım

1. Verilerle bütünleşin (görselleştirin, görselleştirin, görselleştirin...)
2. Verileri önceden işleyin (bir model için hazırlayın)
3. Bir model oluşturun (bir temel ile başlayın)
4. Modeli fit edin
5. Modeli değerlendirin
6. Farklı parametreleri ayarlayın ve modeli iyileştirin (temel çizginizi geçmeye çalışın)
7. Memnun kalana kadar tekrarlayın

Her birinin üzerinden geçelim.

### 1.Verileri İçe Aktarın ve Verilerle Bütünleşin

Ne tür bir veriyle uğraşırsanız uğraşın, kendi zihinsel veri modelinizi oluşturmaya başlamak için en az 10-100 örneği görselleştirmek iyi bir fikirdir.

Bizim durumumuzda, biftek görüntülerinin daha koyu renklere sahip olma eğiliminde olduğunu, pizza görüntülerinin ise ortada belirgin bir dairesel şekle sahip olma eğiliminde olduğunu fark edebiliriz. Bunlar, sinir ağımızın yakaladığı kalıplar olabilir.

Ayrıca, bazı verilerinizin bozuk olup olmadığını (örneğin, yanlış etikete sahip olup olmadığını) fark eder ve bunları düzeltmek için izleyebileceğiniz yolları düşünmeye başlarsınız.


```python
plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("pizza_steak/train/", "pizza")
```

    Image shape: (384, 512, 3)
    Image shape: (512, 512, 3)



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_40_1.png)
    


### 2.Verileri Önceden İşleyin

Bir makine öğrenimi projesi için en önemli adımlardan biri eğitim ve test seti oluşturmaktır.

Bizim durumumuzda, verilerimiz zaten eğitim ve test setlerine bölünmüştür. Buradaki başka bir seçenek de bir doğrulama seti oluşturmak olabilir, ancak şimdilik bunu bırakacağız.

Bir görüntü sınıflandırma projesi için, verilerinizin her sınıf için her birinde alt klasörler bulunan train ve test dizinlerine ayrılması standarttır.

Başlamak için eğitim ve test dizini yollarını tanımlıyoruz.


```python
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"
```

Bir sonraki adımımız, verilerimizi yığınlara dönüştürmektir.

Toplu iş, bir modelin eğitim sırasında baktığı veri kümesinin küçük bir alt kümesidir. Örneğin, bir seferde 10.000 görüntüye bakmak ve kalıpları anlamaya çalışmak yerine, bir model bir seferde yalnızca 32 görüntüye bakabilir.

Bunu birkaç nedenden dolayı yapar:

- 10.000 görüntü (veya daha fazla) işlemcinizin (GPU) belleğine sığmayabilir.
- 10.000 görüntüdeki kalıpları tek bir vuruşta öğrenmeye çalışmak, modelin çok iyi öğrenememesine neden olabilir.

Neden 32?

32'lik bir epoch büyüklüğü sağlığınız için iyidir.

Hayır, gerçekten, kullanabileceğiniz birçok farklı parti boyutu vardır, ancak 32'nin birçok farklı kullanım durumunda çok etkili olduğu kanıtlanmıştır ve çoğu zaman birçok veri ön işleme işlevi için varsayılandır.

Verilerimizi toplu işlere dönüştürmek için önce veri kümelerimizin her biri için bir ImageDataGenerator örneği oluşturacağız.


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
```

ImageDataGenerator sınıfı, görüntülerimizi yığınlar halinde hazırlamamıza ve modele yüklenirken üzerlerinde dönüşümler gerçekleştirmemize yardımcı olur.

Yeniden ölçeklendirme parametresini fark etmiş olabilirsiniz. Bu, yaptığımız dönüşümlerin bir örneğidir.

Daha önce bir görüntüyü nasıl içe aktardığımızı ve piksel değerlerinin 0 ile 255 arasında olduğunu hatırlıyor musunuz?

1/255 ile birlikte yeniden ölçeklendirme parametresi. "tüm piksel değerlerini 255'e böl" demek gibidir. Bu, tüm görüntünün içe aktarılmasıyla ve piksel değerlerinin normalleştirilmesiyle sonuçlanır (0 ile 1 arasında dönüştürülür).

> 🔑 Not: Veri büyütme ve daha fazla dönüştürme seçeneği için (bunu daha sonra göreceğiz), [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) belgelerine bakın.

Şimdi birkaç ImageDataGenerator örneğimiz var, görüntülerimizi flow_from_directory yöntemini kullanarak ilgili dizinlerinden yükleyebiliriz.


```python
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               batch_size=32)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(224, 224),
                                             class_mode='binary',
                                             batch_size=32)
```

    Found 1500 images belonging to 2 classes.
    Found 500 images belonging to 2 classes.


Olağanüstü! Görünüşe göre eğitim veri setimizde 2 sınıfa (pizza ve steak) ait 1500 görüntü var ve test veri setimizde 2 sınıfa ait 500 görüntü var.

Buraya bazı şeyler:

- Dizinlerimizin nasıl yapılandırıldığına bağlı olarak, sınıflar, `train_dir` ve `test_dir` içindeki alt dizin adlarından anlaşılır.
- `target_size` parametresi, resimlerimizin giriş boyutunu (yükseklik, genişlik) biçiminde tanımlar.
- `'binary'`nin class_mode değeri, sınıflandırma problem türümüzü tanımlar. İkiden fazla sınıfımız olsaydı, `'categorical'` kullanırdık.
- `batch_size`, her toplu işte kaç tane resim olacağını tanımlar, biz varsayılanla aynı olan 32'yi kullandık.

Train_data nesnesini inceleyerek toplu resimlerimize ve etiketlerimize göz atabiliriz.


```python
images, labels = train_data.next()
len(images), len(labels)
```




    (32, 32)



Harika, görünüşe göre resimlerimiz ve etiketlerimiz 32'lik gruplar halinde.

Bakalım resimler nasıl görünüyor.


```python
images[:2], images[0].shape
```




    (array([[[[0.47058827, 0.40784317, 0.34509805],
              [0.4784314 , 0.427451  , 0.3647059 ],
              [0.48627454, 0.43529415, 0.37254903],
              ...,
              [0.8313726 , 0.70980394, 0.48627454],
              [0.8431373 , 0.73333335, 0.5372549 ],
              [0.87843144, 0.7725491 , 0.5882353 ]],
     
             [[0.50980395, 0.427451  , 0.36078432],
              [0.5058824 , 0.42352945, 0.35686275],
              [0.5137255 , 0.4431373 , 0.3647059 ],
              ...,
              [0.82745105, 0.7058824 , 0.48235297],
              [0.82745105, 0.70980394, 0.5058824 ],
              [0.8431373 , 0.73333335, 0.5372549 ]],
     
             [[0.5254902 , 0.427451  , 0.34901962],
              [0.5372549 , 0.43921572, 0.36078432],
              [0.5372549 , 0.45098042, 0.36078432],
              ...,
              [0.82745105, 0.7019608 , 0.4784314 ],
              [0.82745105, 0.7058824 , 0.49411768],
              [0.8352942 , 0.7176471 , 0.5137255 ]],
     
             ...,
     
             [[0.77647066, 0.5647059 , 0.2901961 ],
              [0.7803922 , 0.53333336, 0.22352943],
              [0.79215693, 0.5176471 , 0.18039216],
              ...,
              [0.30588236, 0.2784314 , 0.24705884],
              [0.24705884, 0.23137257, 0.19607845],
              [0.2784314 , 0.27450982, 0.25490198]],
     
             [[0.7843138 , 0.57254905, 0.29803923],
              [0.79215693, 0.54509807, 0.24313727],
              [0.8000001 , 0.5254902 , 0.18823531],
              ...,
              [0.2627451 , 0.23529413, 0.20392159],
              [0.24313727, 0.227451  , 0.19215688],
              [0.26666668, 0.2627451 , 0.24313727]],
     
             [[0.7960785 , 0.59607846, 0.3372549 ],
              [0.7960785 , 0.5647059 , 0.26666668],
              [0.81568635, 0.54901963, 0.22352943],
              ...,
              [0.23529413, 0.19607845, 0.16078432],
              [0.3019608 , 0.26666668, 0.24705884],
              [0.26666668, 0.2509804 , 0.24705884]]],
     
     
            [[[0.38823533, 0.4666667 , 0.36078432],
              [0.3921569 , 0.46274513, 0.36078432],
              [0.38431376, 0.454902  , 0.36078432],
              ...,
              [0.5294118 , 0.627451  , 0.54509807],
              [0.5294118 , 0.627451  , 0.54509807],
              [0.5411765 , 0.6392157 , 0.5568628 ]],
     
             [[0.38431376, 0.454902  , 0.3529412 ],
              [0.3921569 , 0.46274513, 0.36078432],
              [0.39607847, 0.4666667 , 0.37254903],
              ...,
              [0.54509807, 0.6431373 , 0.5686275 ],
              [0.5529412 , 0.6509804 , 0.5764706 ],
              [0.5647059 , 0.6627451 , 0.5882353 ]],
     
             [[0.3921569 , 0.46274513, 0.36078432],
              [0.38431376, 0.454902  , 0.3529412 ],
              [0.4039216 , 0.47450984, 0.3803922 ],
              ...,
              [0.5764706 , 0.67058825, 0.6156863 ],
              [0.5647059 , 0.6666667 , 0.6156863 ],
              [0.5647059 , 0.6666667 , 0.6156863 ]],
     
             ...,
     
             [[0.47058827, 0.5647059 , 0.4784314 ],
              [0.4784314 , 0.5764706 , 0.4901961 ],
              [0.48235297, 0.5803922 , 0.49803925],
              ...,
              [0.39607847, 0.42352945, 0.3019608 ],
              [0.37647063, 0.40000004, 0.2901961 ],
              [0.3803922 , 0.4039216 , 0.3019608 ]],
     
             [[0.45098042, 0.5529412 , 0.454902  ],
              [0.46274513, 0.5647059 , 0.4666667 ],
              [0.47058827, 0.57254905, 0.47450984],
              ...,
              [0.40784317, 0.43529415, 0.3137255 ],
              [0.39607847, 0.41960788, 0.31764707],
              [0.38823533, 0.40784317, 0.31764707]],
     
             [[0.47450984, 0.5764706 , 0.47058827],
              [0.47058827, 0.57254905, 0.4666667 ],
              [0.46274513, 0.5647059 , 0.4666667 ],
              ...,
              [0.4039216 , 0.427451  , 0.31764707],
              [0.3921569 , 0.4156863 , 0.3137255 ],
              [0.4039216 , 0.42352945, 0.3372549 ]]]], dtype=float32),
     (224, 224, 3))



Yeniden ölçeklendirme parametremiz nedeniyle, görüntüler artık 0 ile 1 arasında değerlere sahip (224, 224, 3) şekil tensörlerindedir.

Peki ya etiketler?



```python
labels
```




    array([1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
           1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
          dtype=float32)



`class_mode` parametresinin `'binary'` olması nedeniyle etiketlerimiz 0 (pizza) veya 1 (biftek) şeklindedir.

Artık verilerimiz hazır olduğuna göre, modelimiz görüntü tensörleri ve etiketler arasındaki kalıpları bulmaya çalışacak.

### 3.Bir Model Oluşturun

Varsayılan model mimarinizin ne olması gerektiğini merak ediyor olabilirsiniz.

Ve gerçek şu ki, bu sorunun birçok olası cevabı var.

Bilgisayarlı görme modelleri için basit bir buluşsal yöntem, ImageNet'te en iyi performansı gösteren model mimarisini kullanmaktır (farklı bilgisayarlı görme modellerini kıyaslamak için çeşitli görüntülerden oluşan geniş bir koleksiyon).

Bununla birlikte, başlangıç ​​olarak, geliştirmeye çalıştığınız temel bir sonuç elde etmek için daha küçük bir model oluşturmak iyidir.

> 🔑 Not: Derin öğrenmede daha küçük bir model genellikle son teknolojiden (SOTA) daha az katmana sahip bir modele atıfta bulunur. Örneğin, daha küçük bir model 3-4 katmana sahip olabilirken, ResNet50 gibi son teknoloji bir model 50'den fazla katmana sahip olabilir.

Bizim durumumuzda, [CNN açıklayıcı web sitesinde](https://poloclub.github.io/cnn-explainer/) (yukarıdan model_1) bulunabilecek modelin daha küçük bir versiyonunu alalım ve 3 katmanlı bir evrişimli sinir ağı oluşturalım.


```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
```


```python
# Bir model oluşturma
model_4 = Sequential([
  Conv2D(filters=10, 
         kernel_size=3, 
         strides=1,
         padding='valid',
         activation='relu', 
         input_shape=(224, 224, 3)), # input layer 
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  Flatten(),
  Dense(1, activation='sigmoid') # output layer
])
```

Harika! Kullanıma hazır basit bir evrişimsel sinir ağı mimarimiz var. Tipik model yapısını benzer:
```
# Basic structure of CNN
Input -> Conv + ReLU layers (non-linearities) -> Pooling layer -> Fully connected (dense layer) as Output
```

Conv2D katmanının bazı bileşenlerini tartışalım:

- **"Conv2D"** <br>
Girdilerimizin iki boyutlu (yükseklik ve genişlik) olduğu anlamına gelir, 3 renk kanalı olmasına rağmen, kıvrımlar her kanalda ayrı ayrı çalıştırılır.
- **filters**<br> 
Bunlar, resimlerimiz üzerinde hareket edecek olan "özellik çıkarıcıların" sayısıdır.
- **kernel_size**<br> 
Filtrelerimizin boyutu, örneğin bir kernel_size (3, 3) (veya sadece 3), her filtrenin 3x3 boyutuna sahip olacağı, yani her seferinde 3x3 piksellik bir alana bakacağı anlamına gelir. Çekirdek ne kadar küçükse, o kadar ince taneli özellikler çıkaracaktır.
- **stride**<br> 
Bir filtrenin görüntüyü kaplarken üzerinde hareket edeceği piksel sayısı. 1'lik bir adım, filtrenin her piksel boyunca 1'er 1 hareket ettiği anlamına gelir. 2'lik bir adım, bir seferde 2 piksel hareket ettiği anlamına gelir.
- **padding**<br> 
Bu 'same' veya 'valid' olabilir, 'same' görüntünün dışına sıfırlar ekler, böylece evrişim katmanının sonuçtaki çıktısı girişle aynıdır, burada 'valid' (varsayılan) keser filtrenin sığmadığı fazla piksel (örneğin, 224 piksel genişliğinin 3'lük bir çekirdek boyutuna bölünmesi (224/3 = 74.6)), tek bir pikselin uçtan kesileceği anlamına gelir.

**"feature" nedir?**

Bir özellik, bir görüntünün önemli herhangi bir parçası olarak kabul edilebilir. Örneğin, bizim durumumuzda bir özellik pizzanın dairesel şekli olabilir. Veya bir bifteğin dış tarafındaki pürüzlü kenarlar.

Bu özelliklerin bizim tarafımızdan tanımlanmadığını, bunun yerine modelin görüntü üzerinde farklı filtreler uyguladığı için bunları öğrendiğini belirtmek önemlidir.

Artık modelimiz hazır, derleyelim.


```python
model_4.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
```

İkili bir sınıflandırma problemi (pizza vs. biftek) üzerinde çalıştığımız için, kullandığımız kayıp fonksiyonu 'binary_crossentropy'dir, eğer çok sınıflıysa, 'categorical_crossentropy' gibi bir şey kullanabiliriz.

Tüm varsayılan ayarlarla Adam, optimize edicimizdir ve değerlendirme metriğimiz accuracy'dir.

### 4.Modeli Fit Edin

Modelimiz derlendi, fit etme zamanı. Burada iki yeni parametre fark edeceksiniz:

- **step_per_epoch**<br>
Bu, bir modelin epoch başına geçeceği batch sayısıdır, bizim durumumuzda, modelimizin tüm batchi geçmesini istiyoruz, böylece train_data uzunluğuna eşittir (32'lik gruplar halinde 1500 görüntü = 1500/32 = ~ 47 adım)
- **validation_steps**<br>
Validation_data parametresi dışında yukarıdakiyle aynı (32 = 500/32 = ~16 adımlık gruplar halinde 500 test görüntüsü)


```python
# train ve test verilerinin uzunluklarını görüntüleme
print("train_data len: ", len(train_data))
print("test_data len : ", len(test_data))

# modeli fit etme
history_4 = model_4.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
```

    train_data len:  47
    test_data len :  16
    Epoch 1/5
    47/47 [==============================] - 10s 203ms/step - loss: 1.4260 - accuracy: 0.6487 - val_loss: 0.4715 - val_accuracy: 0.7740
    Epoch 2/5
    47/47 [==============================] - 10s 203ms/step - loss: 0.4645 - accuracy: 0.7880 - val_loss: 0.4084 - val_accuracy: 0.8200
    Epoch 3/5
    47/47 [==============================] - 10s 204ms/step - loss: 0.3711 - accuracy: 0.8480 - val_loss: 0.4260 - val_accuracy: 0.8040
    Epoch 4/5
    47/47 [==============================] - 9s 199ms/step - loss: 0.2413 - accuracy: 0.9180 - val_loss: 0.4248 - val_accuracy: 0.8140
    Epoch 5/5
    47/47 [==============================] - 9s 191ms/step - loss: 0.1162 - accuracy: 0.9660 - val_loss: 0.5601 - val_accuracy: 0.7780


### 5.Modeli Değerlendirin

Ah evet! Görünüşe göre modelimiz bir şeyler öğreniyor. Eğitim eğrilerini kontrol edelim.


```python
import pandas as pd
pd.DataFrame(history_4.history).plot(figsize=(10, 7));
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_63_0.png)
    


Hmm, kayıp eğrilerimize bakılırsa, modelimiz eğitim veri setine fazla uyuyor gibi görünüyor.

> 🔑 Not: Bir modelin doğrulama kaybı artmaya başladığında, büyük olasılıkla eğitim veri kümesine gereğinden fazla uyuyordur (overfitting). Bu, eğitim veri setindeki kalıpları çok iyi öğrendiği ve böylece görünmeyen verilere genelleme yapma yeteneğinin azalacağı anlamına gelir.

Modelimizin eğitim performansını daha fazla incelemek için doğruluk ve kayıp eğrilerini ayıralım.


```python
# Doğrulama ve eğitim verilerini ayrı ayrı çizme
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # loss (kayıp) eğriler
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # accuracy(doğruluk) eğrileri
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

plot_loss_curves(history_4)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_65_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_65_1.png)
    


Bu iki eğri için ideal pozisyon birbirini takip etmektir. Herhangi bir şey varsa, doğrulama eğrisi eğitim eğrisinin biraz altında olmalıdır. Eğitim eğrisi ile doğrulama eğrisi arasında büyük bir boşluk varsa, modeliniz muhtemelen fazla uyuyor demektir.


```python
model_4.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 222, 222, 10)      280       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 220, 220, 10)      910       
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 218, 218, 10)      910       
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 475240)            0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 475241    
    =================================================================
    Total params: 477,341
    Trainable params: 477,341
    Non-trainable params: 0
    _________________________________________________________________


### 6.Model Parametrelerini Ayarlayın

Bir makine öğrenimi modelinin takılması 3 adımda gerçekleşir:

0. Bir temel oluşturun.
1. Daha büyük bir modele overfitting taban çizgisini geçin.
2. overfitting azaltın.

Şimdiye kadar 0 ve 1 adımlarından geçtik.

Ve modelimize daha fazla uydurmaya çalışabileceğimiz birkaç şey daha var:

- Evrişim katmanlarının sayısını artırın.
- Evrişimli filtrelerin sayısını artırın.
- Düzleştirilmiş katmanımızın çıktısına başka bir yoğun katman ekleyin.

Ama bunun yerine yapacağımız şey, modelimizin eğitim eğrilerini birbiriyle daha iyi hizalamaya odaklanmak, başka bir deyişle 2. adımı atacağız.


**Overfitting'i azaltmak neden önemlidir?**

Bir model, eğitim verileri üzerinde çok iyi ve görünmeyen veriler üzerinde zayıf performans gösterdiğinde, onu gerçek dünyada kullanmak istiyorsak, bize pek faydası olmaz.

Diyelim ki bir pizza ve biftek yemek sınıflandırıcı uygulaması oluşturuyorduk ve modelimiz eğitim verilerimiz üzerinde çok iyi performans gösteriyor ancak kullanıcılar bunu denediğinde kendi yemek görüntülerinde çok iyi sonuçlar alamadılar, bu iyi bir deneyim mi?

Tam olarak değil...

Dolayısıyla, inşa edeceğimiz sonraki birkaç model için bir dizi parametreyi ayarlayacağız ve yol boyunca eğitim eğrilerini inceleyeceğiz.

Yani, 2 model daha inşa edeceğiz:

- Maksimum pooling'e sahip bir ConvNet
- Maksimum pooling'e ve veri artırma özelliğine sahip bir ConvNet

İlk model için bu yapıyı takip edeceğiz:
```
Input -> Conv layers + ReLU layers (non-linearities) + Max Pooling layers -> Fully connected (dense layer) as Output
```
Hadi inşa edelim. model_4 ile aynı yapıya sahip olacak, ancak her evrişim katmanından sonra bir MaxPool2D() katmanı olacak.


```python
model_5 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])
```

Vayy, daha önce görmediğimiz başka bir katman tipimiz var.

Evrişimli katmanlar bir görüntünün özelliklerini öğrenirse, bu özelliklerden en önemlilerini bulmak olarak bir Max Pooling katmanını düşünebilirsiniz. Bunun bir örneğini birazdan göreceğiz.



```python
# modeli derleme (model_4 gibi)
model_5.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# modeli fit etme
history_5 = model_5.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
```

    Epoch 1/5
    47/47 [==============================] - 9s 182ms/step - loss: 0.5979 - accuracy: 0.7020 - val_loss: 0.4990 - val_accuracy: 0.7500
    Epoch 2/5
    47/47 [==============================] - 8s 179ms/step - loss: 0.4841 - accuracy: 0.7800 - val_loss: 0.4151 - val_accuracy: 0.8040
    Epoch 3/5
    47/47 [==============================] - 8s 179ms/step - loss: 0.4310 - accuracy: 0.8147 - val_loss: 0.3557 - val_accuracy: 0.8560
    Epoch 4/5
    47/47 [==============================] - 9s 190ms/step - loss: 0.4023 - accuracy: 0.8233 - val_loss: 0.3635 - val_accuracy: 0.8360
    Epoch 5/5
    47/47 [==============================] - 9s 190ms/step - loss: 0.3845 - accuracy: 0.8347 - val_loss: 0.3171 - val_accuracy: 0.8820


Tamam, maxPooling'li modelimiz (model_5) eğitim setinde daha kötü ama doğrulama setinde daha iyi performans gösteriyor gibi görünüyor.

Eğitim eğrilerini kontrol etmeden önce mimarisini kontrol edelim.



```python
model_5.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_7 (Conv2D)            (None, 222, 222, 10)      280       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 10)      0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 109, 109, 10)      910       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 54, 54, 10)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 52, 52, 10)        910       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 26, 26, 10)        0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 6760)              0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 1)                 6761      
    =================================================================
    Total params: 8,861
    Trainable params: 8,861
    Non-trainable params: 0
    _________________________________________________________________


Her MaxPooling2D katmanındaki çıktı şekliyle burada neler olduğunu fark ettiniz mi?

Her seferinde yarı yarıya düşüyor. Bu MaxPooling2D katmanının her Conv2D katmanının çıktılarını alması ve "Ben sadece en önemli özellikleri istiyorum, geri kalanlardan kurtulun" demesidir.

pool_size parametresi ne kadar büyük olursa, maksimum havuzlama katmanı o kadar fazla özellikleri görüntüden çıkarır. Ancak, çok büyük ve model hiçbir şey öğrenemeyebilir.

Bu havuzlamanın sonuçları, toplam eğitilebilir parametrelerde (model_5'te 8.861 ve model_4'te 477.431) büyük bir azalma olarak görülmektedir.

Kayıp eğrilerini kontrol etme zamanı.



```python
plot_loss_curves(history_5)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_75_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_75_1.png)
    


Güzel! Eğrilerin birbirine çok daha yakınlaştığını görebiliriz. Bununla birlikte, doğrulama kaybımız sona doğru artmaya başlıyor ve potansiyel olarak fazla uydurmaya yol açıyor.

Hile çantamıza girme ve overfitting'i önlemenin başka bir yöntemini, veri artırmayı denemenin zamanı geldi.

İlk olarak, kodla nasıl yapıldığını göreceğiz, sonra ne yaptığını tartışacağız.

Veri büyütmeyi uygulamak için ImageDataGenerator örneklerimizi yeniden başlatmamız gerekecek.



```python
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                          rotation_range=0.2,# resmi biraz döndür
                          shear_range=0.2, # görüntüyü kırp
                          zoom_range=0.2,# resmi yakınlaştır
                          width_shift_range=0.2, # görüntü genişliği yollarını kaydır
                          height_shift_range=0.2, # görüntü yüksekliği yollarını kaydır
                          horizontal_flip=True) # görüntüyü yatay eksende çevir

train_datagen = ImageDataGenerator(rescale=1/255.) 
test_datagen = ImageDataGenerator(rescale=1/255.)
```

> 🤔 Soru: Veri büyütme nedir?

**Veri büyütme**, eğitim verilerimizi değiştirme, daha fazla çeşitliliğe sahip olmasına ve dolayısıyla modellerimizin daha genelleştirilebilir kalıpları öğrenmesine izin verme sürecidir. Değiştirmek, bir görüntünün dönüşünü ayarlamak, çevirmek, kırpmak veya benzeri bir şey anlamına gelebilir.

Bunu yapmak, bir modelin gerçek dünyada kullanılabileceği veri türünü simüle eder.

Bir pizza ve biftek uygulaması oluşturuyorsak, kullanıcılarımızın çektiği resimlerin tümü eğitim verilerimize benzer kurulumlarda olmayabilir. Veri büyütmeyi kullanmak, overfitting'i önlemenin ve dolayısıyla modelimizi daha genelleştirilebilir hale getirmenin başka bir yolunu sunar.

> 🔑 Not: Veri büyütme genellikle yalnızca eğitim verileri üzerinde gerçekleştirilir. ImageDataGenerator yerleşik veri büyütme parametrelerini kullanarak, görüntülerimiz dizinlerde olduğu gibi bırakılır, ancak modele yüklendiğinde rastgele manipüle edilir.


```python
print("Augmented training images:")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                      target_size=(224, 224),
                      batch_size=32,
                      class_mode='binary',
                      shuffle=False)

print("Non-augmented training images:")
train_data = train_datagen.flow_from_directory(train_dir,
                      target_size=(224, 224),
                      batch_size=32,
                      class_mode='binary',
                      shuffle=False)

print("Unchanged test images:")
test_data = test_datagen.flow_from_directory(test_dir,
                      target_size=(224, 224),
                      batch_size=32,
                      class_mode='binary')
```

    Augmented training images:
    Found 1500 images belonging to 2 classes.
    Non-augmented training images:
    Found 1500 images belonging to 2 classes.
    Unchanged test images:
    Found 500 images belonging to 2 classes.


Veri büyütme hakkında konuşmaktan daha iyi, onu görmeye ne dersiniz?

(mottomuzu hatırlıyor musun? görselleştir, görselleştir, görselleştir...)



```python
images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next() 

# orjinal ile tahmin edilen görseli karşılaştırma
random_number = random.randint(0, 32)
plt.imshow(images[random_number])
plt.title(f"Original image")
plt.axis(False)
plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image")
plt.axis(False);
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_81_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_81_1.png)
    


Orjinal ve artırılmış görsellerin bir örneğini inceledikten sonra, eğitim görselleri üzerinde bazı örnek dönüşümleri görmeye başlayabilirsiniz.

Bazı artırılmış görüntülerin orijinal görüntünün hafifçe çarpık sürümleri gibi göründüğüne dikkat edin. Bu, modelimizin, gerçek dünya görüntülerini kullanırken genellikle olduğu gibi, mükemmel olmayan görüntülerdeki kalıpları denemek ve öğrenmek zorunda kalacağı anlamına gelir.

> 🤔 Soru: Veri büyütmeyi kullanmalı mıyım? Ve ne kadar arttırmalıyım?

Veri büyütme, bir modelin overfitting olmasını önlemenin bir yoludur. Modeliniz gereğinden fazla overfitting oluyorsa (örneğin, doğrulama kaybı artmaya devam ediyorsa), veri büyütmeyi kullanmayı denemek isteyebilirsiniz.

Ne kadar veri artırılacağına gelince, bunun için belirlenmiş bir uygulama yok. ImageDataGenerator sınıfındaki seçeneklere göz atmak ve kullanım durumunuzdaki bir modelin bazı veri artırmalarından nasıl yararlanabileceğini düşünmek en iyisidir.

Şimdi artırılmış veriye sahibiz, üzerine bir model yerleştirmeye çalışalım ve eğitimi nasıl etkilediğini görelim.

Model_5 ile aynı modeli kullanacağız.


```python
# bir model oluşturma (model_5 gibi)
model_6 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), 
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# modeli derleme
model_6.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# modeli fit etme
history_6 = model_6.fit(train_data_augmented, # artırılmış eğitim verilerine değiştirildi
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))
```

    Epoch 1/5
    47/47 [==============================] - 22s 463ms/step - loss: 0.7099 - accuracy: 0.4740 - val_loss: 0.6818 - val_accuracy: 0.7340
    Epoch 2/5
    47/47 [==============================] - 21s 452ms/step - loss: 0.6873 - accuracy: 0.5573 - val_loss: 0.6559 - val_accuracy: 0.7740
    Epoch 3/5
    47/47 [==============================] - 23s 487ms/step - loss: 0.6991 - accuracy: 0.6120 - val_loss: 0.6396 - val_accuracy: 0.6780
    Epoch 4/5
    47/47 [==============================] - 21s 457ms/step - loss: 0.6720 - accuracy: 0.6027 - val_loss: 0.6064 - val_accuracy: 0.8240
    Epoch 5/5
    47/47 [==============================] - 21s 450ms/step - loss: 0.6666 - accuracy: 0.6153 - val_loss: 0.5807 - val_accuracy: 0.7080


> 🤔 Soru: Modelimiz başlangıçta eğitim setinde neden çok iyi sonuçlar alamadı?

Bunun nedeni, train_data_augmented'i oluşturduğumuzda, shuffle=False kullanarak veri karıştırmayı kapatmış olmamızdır; bu, modelimizin bir seferde yalnızca tek bir tür görüntüden oluşan bir toplu iş gördüğü anlamına gelir.

Örneğin, pizza sınıfı birinci sınıf olduğu için ilk yüklenir. Böylece performansı her iki sınıftan ziyade sadece tek bir sınıfta ölçülür. Doğrulama verileri performansı, karıştırılmış veriler içerdiğinden sürekli olarak iyileşir.

Gösteri amacıyla yalnızca shuffle=False ayarladığımızdan (böylece aynı artırılmış ve büyütülmemiş görüntüyü çizebiliriz), gelecekteki veri oluşturucularda shuffle=True ayarını yaparak bunu düzeltebiliriz.

Ayrıca, artırılmış verilerle eğitim alırken her bir dönemin, artırılmamış verilerle eğitime kıyasla daha uzun sürdüğünü fark etmiş olabilirsiniz (dönem başına ~25sn ve dönem başına ~10sn).

Bunun nedeni, ImageDataGenerator örneğinin, modele yüklenirken verileri büyütmesidir. Bunun yararı, orijinal görüntüleri değiştirmeden bırakmasıdır. Dezavantajı, onları yüklemenin daha uzun sürmesidir.

> 🔑 Not: Veri kümesi manipülasyonunu hızlandırmanın olası bir yöntemi,[ TensorFlow'un paralel okumalarına ve arabelleğe alınmış önceliklendirme](https://www.tensorflow.org/tutorials/images/data_augmentation) seçeneklerine bakmak olabilir.


```python
plot_loss_curves(history_6)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_85_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_85_1.png)
    


Doğrulama kaybı eğrimiz doğru yönde ilerliyor gibi görünüyor, ancak biraz ürkek (en ideal kayıp eğrisi çok keskin değil, yumuşak bir iniş, ancak tamamen pürüzsüz bir kayıp eğrisi bir peri masalına eşdeğerdir).

Arttırılmış verileri karıştırdığımızda ne olacağını görelim.


```python
train_data_augmented_shuffled = train_datagen_augmented.flow_from_directory(train_dir,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        shuffle=True)
```

    Found 1500 images belonging to 2 classes.



```python
model_7 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

model_7.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

history_7 = model_7.fit(train_data_augmented_shuffled,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        verbose=0)

plot_loss_curves(history_7)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_88_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_88_1.png)
    


model_7 ile eğitim veri kümesindeki performansın model_6 ile karşılaştırıldığında neredeyse anında nasıl arttığına dikkat edin. Bunun nedeni, eğitim verilerini, flow_from_directory yönteminde shuffle=True parametresini kullanarak modele geçirirken karıştırmış olmamızdır.

Bu, modelin her partide hem pizza hem de biftek görüntülerinin örneklerini görebildiği ve sırayla tek türden değil, her iki görüntüden öğrendiklerini değerlendirebildiği anlamına gelir.

Ayrıca, kayıp eğrilerimiz karıştırılmış verilerle biraz daha pürüzsüz görünüyor (history_6 ile history_7'yi karşılaştırarak).

### 7.Tatmin Olana Kadar Tekrarlayın

Veri kümemizde zaten birkaç model eğittik ve şu ana kadar oldukça iyi performans gösteriyorlar.

Temel çizgimizi çoktan aştığımız için modelimizi geliştirmeye devam etmek için deneyebileceğimiz birkaç şey var:

- Model katmanlarının sayısını artırın (örneğin, daha fazla evrişim katmanı ekleyin).
- Her evrişim katmanındaki filtre sayısını artırın (örn. 10'dan 32'ye, 64'e veya 128'e, bu sayılar da sabit değildir, genellikle deneme yanılma yoluyla bulunurlar).
- Daha uzun süre training yapın (daha fazla dönem).
- İdeal bir öğrenme oranı (learning_rate) bulma.
- Daha fazla veri alın (modele öğrenmesi için daha fazla fırsat verin).
- Başka bir görüntü modelinin öğrendiklerinden yararlanmak için aktarım öğrenimini kullanın ve bunu kendi kullanım durumumuza göre ayarlayın.

Model geliştirme sırasında bu ayarların her birinin (son ikisi hariç) ayarlanması genellikle hiperparametre ayarı olarak adlandırılır.

Hiperparametre ayarını, en sevdiğiniz yemeği pişirmek için fırınınızdaki ayarları yapmaya benzetebilirsiniz. Fırınınız sizin için pişirmenin çoğunu yapsa da, ısıyı ayarlayarak buna yardımcı olabilirsiniz.

Başladığımız yere geri dönelim ve orijinal modelimizi deneyelim:


```python
model_8 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)), 
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

model_8.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_8 = model_8.fit(train_data_augmented_shuffled,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data=test_data,
                        validation_steps=len(test_data))
```

    Epoch 1/5
    47/47 [==============================] - 23s 473ms/step - loss: 0.6579 - accuracy: 0.6087 - val_loss: 0.5885 - val_accuracy: 0.6440
    Epoch 2/5
    47/47 [==============================] - 24s 502ms/step - loss: 0.5509 - accuracy: 0.7293 - val_loss: 0.4329 - val_accuracy: 0.8140
    Epoch 3/5
    47/47 [==============================] - 22s 473ms/step - loss: 0.5493 - accuracy: 0.7320 - val_loss: 0.5195 - val_accuracy: 0.7300
    Epoch 4/5
    47/47 [==============================] - 22s 478ms/step - loss: 0.5418 - accuracy: 0.7367 - val_loss: 0.4005 - val_accuracy: 0.8380
    Epoch 5/5
    47/47 [==============================] - 24s 501ms/step - loss: 0.5206 - accuracy: 0.7667 - val_loss: 0.3967 - val_accuracy: 0.8340


> 🔑 Not: Model_8'i oluşturmak için model_1 ile karşılaştırıldığında biraz farklı kodlar kullandığımızı fark etmiş olabilirsiniz. Bunun nedeni, daha önce yaptığımız içe aktarmalar, örneğin tensorflow.keras.layers'dan içe aktarma Conv2D, yazmamız gereken kod miktarını azaltır. Kodlar farklı olsa da mimariler aynı.


```python
model_1.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 222, 222, 10)      280       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 220, 220, 10)      910       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 110, 110, 10)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 108, 108, 10)      910       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 106, 106, 10)      910       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 53, 53, 10)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 28090)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 28091     
    =================================================================
    Total params: 31,101
    Trainable params: 31,101
    Non-trainable params: 0
    _________________________________________________________________



```python
model_8.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_20 (Conv2D)           (None, 222, 222, 10)      280       
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 220, 220, 10)      910       
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 110, 110, 10)      0         
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 108, 108, 10)      910       
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 106, 106, 10)      910       
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 53, 53, 10)        0         
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 28090)             0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 1)                 28091     
    =================================================================
    Total params: 31,101
    Trainable params: 31,101
    Non-trainable params: 0
    _________________________________________________________________


Şimdi TinyVGG modelimizin performansını kontrol edelim.


```python
# TinyVGG model performansına göz atın
plot_loss_curves(history_8)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_96_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_96_1.png)
    



```python
# Bu eğitim eğrisi yukarıdakine kıyasla nasıl görünüyor?
plot_loss_curves(history_1)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_97_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_97_1.png)
    


Hmm, eğitim eğrilerimiz iyi görünüyor, ancak modelimizin eğitim ve test setlerindeki performansı önceki modele göre pek gelişmedi.

Eğitim eğrilerine bir kez daha baktığımızda, biraz daha uzun süre (daha fazla dönem) eğitirsek modelimizin performansı artabilir gibi görünüyor.


### Eğitimli Modelimiz ile Tahmin Yapmak

Onunla tahmin yapamıyorsanız, eğitimli bir model ne işe yarar? Gerçekten test etmek için kendi resimlerimizden birkaçını yükleyeceğiz ve modelin nasıl gittiğini göreceğiz. Öncelikle kendimize sınıf isimlerini hatırlatalım ve üzerinde test edeceğimiz görsele bakalım.


```python
# Çalıştığımız Sınıflar
print(class_names)
```

    ['.DS_Store' 'pizza' 'steak']



```python
!wget "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/steak_2.jpg"
steak = mpimg.imread("steak_2.jpg")
plt.imshow(steak)
plt.axis(False);
```

    --2021-07-17 13:06:45--  https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/steak_2.jpg
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 300203 (293K) [image/jpeg]
    Saving to: ‘steak_2.jpg.2’
    
    steak_2.jpg.2       100%[===================>] 293.17K  --.-KB/s    in 0.02s   
    
    2021-07-17 13:06:45 (15.7 MB/s) - ‘steak_2.jpg.2’ saved [300203/300203]
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_101_1.png)
    



```python
# shape değerini kontrol edelim
steak.shape
```




    (1310, 1258, 3)



Modelimiz şekillerin (224, 224, 3) görüntülerini aldığından, kendi modelimiz ile kullanmak için özel imajımızı yeniden şekillendirmemiz gerekiyor.

Bunu yapmak için, tf.io.read_file (dosyaları okumak için) ve tf.image (imajımızı yeniden boyutlandırmak ve bir tensöre dönüştürmek için) kullanarak imajımızı içe aktarabilir ve kodunu çözebiliriz.

> 🔑 Not: Modelinizin, örneğin kendi özel resimleriniz gibi görünmeyen verilerle ilgili tahminlerde bulunabilmesi için, özel görüntünün, modelinizin eğitildiği şekilde olması gerekir. Daha genel bir ifadeyle, özel veriler üzerinde tahminlerde bulunmak için modelinizin eğitildiği formda olması gerekir.




```python
# Bir görüntüyü içe aktarmak ve modelimiz ile kullanılabilecek şekilde 
# yeniden boyutlandırmak için bir işlev oluşturalım
def load_and_prep_image(filename, img_shape=224):
  # resmi okuma
  img = tf.io.read_file(filename)
  # Bir tensöre decode etme
  img = tf.image.decode_jpeg(img)
  # yeniden boyutlandırma
  img = tf.image.resize(img, [img_shape, img_shape])
  # normalize
  img = img/255.
  return img
```

Şimdi özel imajımızı yüklemek için bir fonksiyonumuz var, onu yükleyelim.


```python
steak = load_and_prep_image("steak_2.jpg")
steak
```




    <tf.Tensor: shape=(224, 224, 3), dtype=float32, numpy=
    array([[[0.26381305, 0.24028361, 0.19322479],
            [0.26572865, 0.24219921, 0.19514039],
            [0.2708914 , 0.24736199, 0.20030317],
            ...,
            [0.2650035 , 0.24931723, 0.20225841],
            [0.26321357, 0.2475273 , 0.20046848],
            [0.28401613, 0.26832986, 0.22127101]],
    
           [[0.28641456, 0.26288515, 0.21582633],
            [0.2924    , 0.2688706 , 0.22181177],
            [0.2810408 , 0.25751138, 0.21045254],
            ...,
            [0.24492297, 0.22923669, 0.18609944],
            [0.24658544, 0.23089917, 0.18384035],
            [0.28021708, 0.2645308 , 0.21747199]],
    
           [[0.28692865, 0.2627451 , 0.21601336],
            [0.29481623, 0.2704132 , 0.22379117],
            [0.2961762 , 0.27213612, 0.22533265],
            ...,
            [0.23741287, 0.2217266 , 0.18643248],
            [0.23338516, 0.21421418, 0.1893742 ],
            [0.2734472 , 0.2541664 , 0.22965583]],
    
           ...,
    
           [[0.55565476, 0.5125175 , 0.4310156 ],
            [0.56367356, 0.5205363 , 0.44063556],
            [0.5681545 , 0.52501726, 0.44246843],
            ...,
            [0.6766286 , 0.6413344 , 0.57466775],
            [0.65775603, 0.6224619 , 0.55579525],
            [0.6625994 , 0.62730527, 0.5606386 ]],
    
           [[0.5369749 , 0.50560236, 0.43109256],
            [0.535942  , 0.5045695 , 0.43005973],
            [0.5529945 , 0.52162194, 0.44711217],
            ...,
            [0.6752446 , 0.6438721 , 0.5693623 ],
            [0.67058825, 0.6392157 , 0.5647059 ],
            [0.67270285, 0.6413303 , 0.5668205 ]],
    
           [[0.50416195, 0.47278938, 0.39827958],
            [0.5234028 , 0.49203026, 0.41752046],
            [0.5509676 , 0.519595  , 0.44508523],
            ...,
            [0.6779879 , 0.6466153 , 0.5721055 ],
            [0.6653004 , 0.6339279 , 0.5594181 ],
            [0.6922957 , 0.6609231 , 0.5864133 ]]], dtype=float32)>



Harika, imajımız tensör formatında, modelimiz ile deneme zamanı!


```python
model_8.predict(steak)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-98-f52a71fd4ea7> in <module>()
    ----> 1 model_8.predict(steak)
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py in predict(self, x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
       1725           for step in data_handler.steps():
       1726             callbacks.on_predict_batch_begin(step)
    -> 1727             tmp_batch_outputs = self.predict_function(iterator)
       1728             if data_handler.should_sync:
       1729               context.async_wait()


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        887 
        888       with OptionalXlaContext(self._jit_compile):
    --> 889         result = self._call(*args, **kwds)
        890 
        891       new_tracing_count = self.experimental_get_tracing_count()


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        922       # In this case we have not created variables on the first call. So we can
        923       # run the first trace but we should fail if variables are created.
    --> 924       results = self._stateful_fn(*args, **kwds)
        925       if self._created_variables:
        926         raise ValueError("Creating variables on a non-first call to a function"


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
    
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1569 predict_function  *
            return step_function(self, iterator)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1559 step_function  **
            outputs = model.distribute_strategy.run(run_step, args=(data,))
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:1285 run
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:2833 call_for_each_replica
            return self._call_for_each_replica(fn, args, kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/distribute/distribute_lib.py:3608 _call_for_each_replica
            return fn(*args, **kwargs)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1552 run_step  **
            outputs = model.predict_step(data)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1525 predict_step
            return self(x, training=False)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/base_layer.py:1013 __call__
            input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
        /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/input_spec.py:235 assert_input_compatibility
            str(tuple(shape)))
    
        ValueError: Input 0 of layer sequential_8 is incompatible with the layer: : expected min_ndim=4, found ndim=3. Full shape received: (32, 224, 3)



Bir sorun daha var...

Görselimiz, modelimizin eğitildiği görüntülerle aynı şekilde olmasına rağmen, hala bir boyutu kaçırıyoruz.

Modelimizin gruplar halinde nasıl eğitildiğini hatırlıyor musunuz? batch boyutu ilk boyut olur. Yani gerçekte, modelimiz (batch_size, 224, 224, 3) şeklindeki veriler üzerinde eğitildi.

Bunu, `tf.expand_dims` kullanarak özel görüntü tensörümüze fazladan ekleyerek düzeltebiliriz.


```python
print(f"Shape before new dimension: {steak.shape}")
steak = tf.expand_dims(steak, axis=0)
print(f"Shape after new dimension: {steak.shape}")
steak
```

    Shape before new dimension: (224, 224, 3)
    Shape after new dimension: (1, 224, 224, 3)





    <tf.Tensor: shape=(1, 224, 224, 3), dtype=float32, numpy=
    array([[[[0.26381305, 0.24028361, 0.19322479],
             [0.26572865, 0.24219921, 0.19514039],
             [0.2708914 , 0.24736199, 0.20030317],
             ...,
             [0.2650035 , 0.24931723, 0.20225841],
             [0.26321357, 0.2475273 , 0.20046848],
             [0.28401613, 0.26832986, 0.22127101]],
    
            [[0.28641456, 0.26288515, 0.21582633],
             [0.2924    , 0.2688706 , 0.22181177],
             [0.2810408 , 0.25751138, 0.21045254],
             ...,
             [0.24492297, 0.22923669, 0.18609944],
             [0.24658544, 0.23089917, 0.18384035],
             [0.28021708, 0.2645308 , 0.21747199]],
    
            [[0.28692865, 0.2627451 , 0.21601336],
             [0.29481623, 0.2704132 , 0.22379117],
             [0.2961762 , 0.27213612, 0.22533265],
             ...,
             [0.23741287, 0.2217266 , 0.18643248],
             [0.23338516, 0.21421418, 0.1893742 ],
             [0.2734472 , 0.2541664 , 0.22965583]],
    
            ...,
    
            [[0.55565476, 0.5125175 , 0.4310156 ],
             [0.56367356, 0.5205363 , 0.44063556],
             [0.5681545 , 0.52501726, 0.44246843],
             ...,
             [0.6766286 , 0.6413344 , 0.57466775],
             [0.65775603, 0.6224619 , 0.55579525],
             [0.6625994 , 0.62730527, 0.5606386 ]],
    
            [[0.5369749 , 0.50560236, 0.43109256],
             [0.535942  , 0.5045695 , 0.43005973],
             [0.5529945 , 0.52162194, 0.44711217],
             ...,
             [0.6752446 , 0.6438721 , 0.5693623 ],
             [0.67058825, 0.6392157 , 0.5647059 ],
             [0.67270285, 0.6413303 , 0.5668205 ]],
    
            [[0.50416195, 0.47278938, 0.39827958],
             [0.5234028 , 0.49203026, 0.41752046],
             [0.5509676 , 0.519595  , 0.44508523],
             ...,
             [0.6779879 , 0.6466153 , 0.5721055 ],
             [0.6653004 , 0.6339279 , 0.5594181 ],
             [0.6922957 , 0.6609231 , 0.5864133 ]]]], dtype=float32)>



Özel görselimizin toplu iş boyutu 1'dir! Ona göre bir tahmin yapalım.


```python
pred = model_8.predict(steak)
pred
```




    array([[0.3949119]], dtype=float32)



Ahh, tahminler olasılık şeklinde çıkıyor. Başka bir deyişle, bu, görüntünün bir sınıf veya başka bir sınıf olma olasılığının ne kadar olduğu anlamına gelir.

İkili bir sınıflandırma problemi ile çalıştığımız için, tahmin olasılığı modele göre 0,5'in üzerindeyse, tahmin büyük olasılıkla pozitif sınıf olacaktır (sınıf 1).

Ve tahmin olasılığı 0,5'in altındaysa, modele göre, tahmin edilen sınıf büyük olasılıkla negatif sınıftır (sınıf 0).

> 🔑 Not: 0,5 kesme beğeninize göre ayarlanabilir. Örneğin, pozitif sınıf için limiti 0,8 ve üzeri ve negatif sınıf için 0,2 olarak ayarlayabilirsiniz. Ancak, bunu yapmak neredeyse her zaman modelinizin performans ölçümlerini değiştirecektir, bu nedenle doğru yönde değiştiklerinden emin olun.

Ama pizza 🍕 ve biftek 🥩 ile çalışırken pozitif ve negatif sınıf demek pek mantıklı gelmiyor...

Öyleyse, tahminleri sınıf adlarına dönüştürmek için küçük bir fonksiyon yazalım ve ardından hedef görüntüyü çizelim.


```python
class_names
```




    array(['.DS_Store', 'pizza', 'steak'], dtype='<U9')




```python
class_names = class_names[1:]
class_names
```




    array(['pizza', 'steak'], dtype='<U9')




```python
# Tahmin olasılığını yuvarlayarak tahmin edilen sınıfı indeksleyebiliriz
pred_class = class_names[int(tf.round(pred)[0][0])]
pred_class
```




    'pizza'




```python
def pred_and_plot(model, filename, class_names):
  # Hedef görüntüyü içe aktarma ve önişleme
  img = load_and_prep_image(filename)

  # tahmin yapma
  pred = model.predict(tf.expand_dims(img, axis=0))

  # tahmin yapma (sınıf)
  pred_class = class_names[int(tf.round(pred)[0][0])]

  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);


pred_and_plot(model_8, "steak_2.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_117_0.png)
    


Güzel! Modelimiz tahmini doğru yaptı. Yemekle çalışmanın tek dezavantajı bu beni acıktırıyor. Bir resim daha deneyelim.



```python
# Download another test image and make a prediction on it
!wget "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/pizza_1.jpg" 
pred_and_plot(model_8, "pizza_1.jpg", class_names)
```

    --2021-07-17 13:07:52--  https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/pizza_1.jpg
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.110.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 285436 (279K) [image/jpeg]
    Saving to: ‘pizza_1.jpg’
    
    pizza_1.jpg         100%[===================>] 278.75K  --.-KB/s    in 0.02s   
    
    2021-07-17 13:07:53 (13.2 MB/s) - ‘pizza_1.jpg’ saved [285436/285436]
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_119_1.png)
    


## Çok Sınıflı Sınıflandırma

Bu defter aracılığıyla CNN Açıklayıcı web sitesindeki TinyVGG mimarisine birçok kez atıfta bulunduk, ancak CNN Açıklayıcı web sitesi 10 farklı görüntü sınıfıyla çalışıyor, şu anki modelimiz yalnızca iki sınıfla (pizza ve biftek) çalışıyor.

> 🛠 Alıştırma: Aşağı kaydırmadan önce, aynı türden görüntülerden oluşan 10 sınıfla çalışmak için modelimizi nasıl değiştirebileceğimizi düşünüyorsunuz? Verilerin iki sınıf problemimizle aynı tarzda olduğunu varsayalım.

Pizza 🍕 ve biftek 🥩 sınıflandırıcımızı oluşturmak için daha önce attığımız adımları hatırlıyor musunuz?

O aşamaları bir kez daha gözden geçirmeye ne dersiniz, ama bu sefer 10 farklı yiyecek türüyle çalışacağız.

1. Verilerle bütünleşin (görselleştirin, görselleştirin, görselleştirin...)
2. Verileri önceden işleyin (bir model için hazırlayın)
3. Bir model oluşturun (bir temel ile başlayın)
4. Modeli fit edin
5. Modeli değerlendirin
6. Farklı parametreleri ayarlayın ve modeli iyileştirin (temel çizginizi geçmeye çalışın)
7. Memnun kalana kadar tekrarlayın

<img src="https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2020/05/13/sagemaker-tensorflow.png" />

### 1.Verileri İçe Aktarın ve Verilerle Bütünleşin

Yine, Food101 veri kümesinin bir alt kümesine sahibiz. Pizza ve biftek resimlerine ek olarak, sekiz sınıf daha çıkardık.


```python
import zipfile

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip 

zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()
```

    --2021-07-17 13:07:57--  https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.250.141.128, 142.251.2.128, 2607:f8b0:4023:c06::80, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.250.141.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 519183241 (495M) [application/zip]
    Saving to: ‘10_food_classes_all_data.zip’
    
    10_food_classes_all 100%[===================>] 495.13M   246MB/s    in 2.0s    
    
    2021-07-17 13:07:59 (246 MB/s) - ‘10_food_classes_all_data.zip’ saved [519183241/519183241]
    


Şimdi 10_food_classes dosyasındaki tüm farklı dizinleri ve alt dizinleri kontrol edelim.


```python
import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
  print(f"'{dirpath}' klasöründe {len(filenames)} veri var.")
```

    '10_food_classes_all_data' klasöründe 0 veri var.
    '10_food_classes_all_data/test' klasöründe 0 veri var.
    '10_food_classes_all_data/test/hamburger' klasöründe 250 veri var.
    '10_food_classes_all_data/test/grilled_salmon' klasöründe 250 veri var.
    '10_food_classes_all_data/test/pizza' klasöründe 250 veri var.
    '10_food_classes_all_data/test/chicken_curry' klasöründe 250 veri var.
    '10_food_classes_all_data/test/sushi' klasöründe 250 veri var.
    '10_food_classes_all_data/test/ice_cream' klasöründe 250 veri var.
    '10_food_classes_all_data/test/fried_rice' klasöründe 250 veri var.
    '10_food_classes_all_data/test/chicken_wings' klasöründe 250 veri var.
    '10_food_classes_all_data/test/steak' klasöründe 250 veri var.
    '10_food_classes_all_data/test/ramen' klasöründe 250 veri var.
    '10_food_classes_all_data/train' klasöründe 0 veri var.
    '10_food_classes_all_data/train/hamburger' klasöründe 750 veri var.
    '10_food_classes_all_data/train/grilled_salmon' klasöründe 750 veri var.
    '10_food_classes_all_data/train/pizza' klasöründe 750 veri var.
    '10_food_classes_all_data/train/chicken_curry' klasöründe 750 veri var.
    '10_food_classes_all_data/train/sushi' klasöründe 750 veri var.
    '10_food_classes_all_data/train/ice_cream' klasöründe 750 veri var.
    '10_food_classes_all_data/train/fried_rice' klasöründe 750 veri var.
    '10_food_classes_all_data/train/chicken_wings' klasöründe 750 veri var.
    '10_food_classes_all_data/train/steak' klasöründe 750 veri var.
    '10_food_classes_all_data/train/ramen' klasöründe 750 veri var.


İyi görünüyor! Şimdi eğitim ve test dizini yollarını ayarlayacağız.


```python
# Çok sınıflı veri kümemiz için sınıf adlarını alma
import pathlib
import numpy as np

train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
```

    ['chicken_curry' 'chicken_wings' 'fried_rice' 'grilled_salmon' 'hamburger'
     'ice_cream' 'pizza' 'ramen' 'steak' 'sushi']


Eğitim setinden bir görseli görselleştirmeye ne dersiniz?


```python
import random
img = view_random_image(target_dir=train_dir,
                        target_class=random.choice(class_names)) # get a random class name
```

    Image shape: (512, 384, 3)



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_128_1.png)
    


### 2.Verileri Önceden İşleyin

Birkaç resimden geçtikten sonra (en az 10-100 farklı örneği görselleştirmek iyidir), veri dizinlerimiz doğru şekilde ayarlanmış gibi görünüyor.

Verileri önceden işleme zamanı.


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Verileri yeniden ölçeklendirme
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Dizinlerden veri yükleme ve yığınlara dönüştürme
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical') 

test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')
```

    Found 7500 images belonging to 10 classes.
    Found 2500 images belonging to 10 classes.


İkili sınıflandırmada olduğu gibi, görüntü oluşturucularımız var. Bu seferki ana değişiklik, class_mode parametresini 'categorical' olarak değiştirmemizdir çünkü 10 sınıf yiyecek görüntüsü ile uğraşıyoruz.

Görüntüleri yeniden ölçeklendirmek, toplu iş boyutunu ve hedef görüntü boyutunu oluşturmak gibi diğer her şey aynı kalır.

> 🤔 Soru: Resmin boyutu neden 224x224? Bu aslında istediğimiz herhangi bir boyut olabilir, ancak 224x224, görüntülerin ön işlemesi için çok yaygın bir boyuttur. Sorununuza bağlı olarak daha büyük veya daha küçük resimler kullanmak isteyebilirsiniz.

### 3.Bir Model Oluşturma

İkili sınıflandırma problemi için kullandığımız aynı modeli (TinyVGG) çok sınıflı sınıflandırma problemimiz için birkaç küçük ince ayar ile kullanabiliriz.

Yani:

- Kullanılacak çıktı katmanını değiştirmek, 10 çıktı nöronuna sahiptir (sahip olduğumuz sınıf sayısıyla aynı sayı).
- Çıktı katmanını 'sigmoid' aktivasyonu yerine 'softmax' aktivasyonunu kullanacak şekilde değiştirme.
- Kayıp işlevinin 'binary_crossentropy' yerine 'categorical_crossentropy' olarak değiştirilmesi.


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model_9 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(10, activation='softmax') 
])

model_9.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
```

### 4.Modeli Fit Edin

Şimdi birden fazla sınıfla çalışmaya uygun bir modelimiz var, onu verilerimizle fit edelim.


```python
history_9 = model_9.fit(train_data, # 10 farklı sınıf
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
```

    Epoch 1/5
    235/235 [==============================] - 46s 194ms/step - loss: 2.1381 - accuracy: 0.2271 - val_loss: 2.0048 - val_accuracy: 0.2716
    Epoch 2/5
    235/235 [==============================] - 44s 188ms/step - loss: 1.8894 - accuracy: 0.3475 - val_loss: 1.8485 - val_accuracy: 0.3668
    Epoch 3/5
    235/235 [==============================] - 44s 188ms/step - loss: 1.6009 - accuracy: 0.4596 - val_loss: 1.8771 - val_accuracy: 0.3644
    Epoch 4/5
    235/235 [==============================] - 45s 191ms/step - loss: 1.0496 - accuracy: 0.6556 - val_loss: 2.1192 - val_accuracy: 0.3332
    Epoch 5/5
    235/235 [==============================] - 44s 189ms/step - loss: 0.4512 - accuracy: 0.8572 - val_loss: 2.9371 - val_accuracy: 0.2972


Neden her epoch yalnızca iki görüntü sınıfıyla çalışırken olduğundan daha uzun sürdüğünü düşünüyorsunuz?

Çünkü artık eskisinden daha fazla görüntüyle uğraşıyoruz. 750 eğitim görüntüsü ve her biri toplam 10.000 görüntü olan 250 doğrulama görüntüsü içeren 10 sınıfımız var. İki sınıfımız olduğu zaman, toplam 2000 olmak üzere 1500 eğitim görselimiz ve 500 doğrulama görselimiz vardı.

Buradaki sezgisel akıl yürütme, ne kadar fazla veriye sahip olursanız, bir modelin kalıpları bulması o kadar uzun sürer.

### 5.Modeli Değerlendirin

Woohoo! Az önce bir modeli 10 farklı yiyecek görüntüsü sınıfında eğittik, bakalım nasılmış.


```python
model_9.evaluate(test_data)
```

    79/79 [==============================] - 10s 131ms/step - loss: 2.9371 - accuracy: 0.2972





    [2.9370851516723633, 0.2971999943256378]




```python
plot_loss_curves(history_9)
```

Woah, eğitim ve doğrulama kaybı eğrileri arasındaki boşluk çoook fazla.

Bu bize ne anlatıyor?

Görünüşe göre modelimiz eğitim setine oldukça overfitting olmuş. Başka bir deyişle, eğitim verilerinde harika sonuçlar alıyor ancak görünmeyen verilere genelleme yapamıyor ve test verilerinde düşük performans gösteriyor.


### 6.Model Parametrelini Ayarların (Fine-Tuning)

Eğitim verileri üzerindeki performansı nedeniyle, modelimizin bir şeyler öğrendiği açıktır. Bununla birlikte, eğitim verileri üzerinde iyi performans sergilemek, sınıfta iyi gidiyor ancak becerilerinizi gerçek hayatta kullanamamak gibidir.

İdeal olarak, modelimizin eğitim verilerinde olduğu gibi test verilerinde de performans göstermesini isteriz.

Bu yüzden sonraki adımlarımız, modelimizin fazla takılmasını önlemek olacaktır. Overfitting önlemenin birkaç yolu şunları içerir:

- **Daha fazla veri elde edin**<br> 
Daha fazla veriye sahip olmak, modele yeni örneklere daha genelleştirilebilecek kalıpları, kalıpları öğrenmek için daha fazla fırsat verir.
- **Modeli basitleştirin**<br> 
Mevcut model zaten eğitim verilerine fazla uyuyorsa, bir model için çok karmaşık olabilir. Bu, veri kalıplarını çok iyi öğrendiği ve görünmeyen verilere iyi genelleme yapamadığı anlamına gelir. Bir modeli basitleştirmenin bir yolu, kullandığı katman sayısını azaltmak veya her katmandaki gizli birimlerin sayısını azaltmaktır.
- **Veri büyütmeyi kullan**<br> 
Veri büyütme, eğitim verilerini bir şekilde manipüle eder, böylece verilere yapay olarak daha fazla çeşitlilik eklediğinden modelin öğrenmesi daha zordur. Bir model artırılmış verilerdeki kalıpları öğrenebiliyorsa, model görünmeyen verilere daha iyi genelleme yapabilir.
- **Transfer öğrenimini kullanın**<br> 
Transfer öğrenimi, bir modelin kendi göreviniz için temel olarak kullanmayı öğrendiği kalıplardan (önceden eğitilmiş ağırlıklar olarak da adlandırılır) yararlanmayı içerir. Bizim durumumuzda, çok çeşitli görüntüler üzerinde önceden eğitilmiş bir bilgisayarlı görü modelini kullanabilir ve ardından yiyecek görüntüleri için daha özel olması için biraz ince ayar yapabilirdik.

Halihazırda mevcut bir veri kümeniz varsa, muhtemelen ilk önce yukarıdaki son üç seçenekten birini veya bunların bir kombinasyonunu denemeniz olasıdır.

Daha fazla veri toplamak, elle daha fazla yiyecek görüntüsü almamızı gerektireceğinden, yapabileceklerimizi doğrudan not defterinden deneyelim.

Önce modelimizi sadeleştirmeye ne dersiniz?

Bunu yapmak için, toplam evrişim katmanı sayısını dörtten ikiye alarak iki kat evrişim katmanını kaldıracağız.


```python
model_10 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(10, activation='softmax')
])

model_10.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])

history_10 = model_10.fit(train_data,
                          epochs=5,
                          steps_per_epoch=len(train_data),
                          validation_data=test_data,
                          validation_steps=len(test_data))
```

    Epoch 1/5
    235/235 [==============================] - 44s 185ms/step - loss: 2.2136 - accuracy: 0.2260 - val_loss: 1.9535 - val_accuracy: 0.3224
    Epoch 2/5
    235/235 [==============================] - 42s 177ms/step - loss: 1.7867 - accuracy: 0.3912 - val_loss: 1.9660 - val_accuracy: 0.3044
    Epoch 3/5
    235/235 [==============================] - 43s 183ms/step - loss: 1.4147 - accuracy: 0.5403 - val_loss: 1.9701 - val_accuracy: 0.3280
    Epoch 4/5
    235/235 [==============================] - 43s 184ms/step - loss: 0.9434 - accuracy: 0.7045 - val_loss: 2.2627 - val_accuracy: 0.2972
    Epoch 5/5
    235/235 [==============================] - 42s 178ms/step - loss: 0.5380 - accuracy: 0.8457 - val_loss: 2.6522 - val_accuracy: 0.3004



```python
# model_10'un kayıp eğrilerine göz atalım
plot_loss_curves(history_10)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_143_0.png)
    



    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_143_1.png)
    


Hmm... basitleştirilmiş bir modelle bile, modelimiz hala eğitim verilerine önemli ölçüde uyuyor gibi görünüyor.

Başka ne deneyebiliriz? Veri artırmaya ne dersiniz?

Veri büyütme, modelin eğitim verileri üzerinde öğrenmesini zorlaştırır ve bunun sonucunda öğrendiği kalıpları görünmeyen verilere daha genelleştirilebilir hale getirmeyi umar.

Artırılmış veri oluşturmak için, yeni bir ImageDataGenerator örneğini yeniden oluşturacağız, bu sefer resimlerimizi işlemek için rotation_range ve horizontal_flip gibi bazı parametreler ekleyeceğiz.



```python
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                  target_size=(224, 224),
                                                                  batch_size=32,
                                                                  class_mode='categorical')
```

    Found 7500 images belonging to 10 classes.


Şimdi artırılmış veriye sahibiz, eskisi gibi aynı modelle (model_10) nasıl çalıştığını görelim.

Modeli sıfırdan yeniden yazmak yerine, mevcut bir modeli alıp aynı biçimde yeniden oluşturabilen clon_model adlı TensorFlow'daki kullanışlı bir işlevi kullanarak onu klonlayabiliriz.

Klonlanmış sürüm, orijinal modelin öğrendiği ağırlıkların (kalıpların) hiçbirini içermeyecektir. Yani onu eğittiğimizde, sıfırdan bir modeli eğitmek gibi olacak.

> 🔑 Not: Derin öğrenme ve genel olarak makine öğrenimindeki temel uygulamalardan biri seri deneyci olmaktır. Burada yaptığımız şey bu. Bir şey denemek, işe yarayıp yaramadığını görmek, sonra başka bir şey denemek. İyi bir deneme kurulumu, değiştirdiğiniz şeyleri de takip eder. Bu nedenle öncekiyle aynı modeli ancak farklı verilerle kullanıyoruz. Model aynı kalır, ancak veriler değişir, bu, artırılmış eğitim verilerinin performans üzerinde herhangi bir etkisi olup olmadığını bize bildirir.


```python
model_11 = tf.keras.models.clone_model(model_10)

model_11.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history_11 = model_11.fit(train_data_augmented,
                          epochs=5,
                          steps_per_epoch=len(train_data_augmented),
                          validation_data=test_data,
                          validation_steps=len(test_data))
```

    Epoch 1/5
    235/235 [==============================] - 109s 464ms/step - loss: 2.2901 - accuracy: 0.1651 - val_loss: 2.0693 - val_accuracy: 0.2508
    Epoch 2/5
    235/235 [==============================] - 108s 462ms/step - loss: 2.1009 - accuracy: 0.2405 - val_loss: 1.9395 - val_accuracy: 0.3140
    Epoch 3/5
    235/235 [==============================] - 107s 455ms/step - loss: 2.0259 - accuracy: 0.2909 - val_loss: 1.8651 - val_accuracy: 0.3428
    Epoch 4/5
    235/235 [==============================] - 110s 469ms/step - loss: 1.9746 - accuracy: 0.3056 - val_loss: 1.8180 - val_accuracy: 0.3588
    Epoch 5/5
    235/235 [==============================] - 109s 462ms/step - loss: 1.9305 - accuracy: 0.3273 - val_loss: 1.7820 - val_accuracy: 0.3896


Her epoch bir önceki modelden daha uzun sürdüğünü görebilirsiniz. Bunun nedeni, verilerimizin GPU'ya yüklendikçe CPU'da anında genişletilmesi ve her dönem arasındaki sürenin artmasıdır.

Modelimizin eğitim eğrileri nasıl görünüyor?



```python
plot_loss_curves(history_11)
```

Vay! Bu çok daha iyi görünüyor, kayıp eğrileri birbirine çok daha yakın. Modelimiz artırılmış eğitim setinde iyi performans göstermese de doğrulama veri setinde çok daha iyi performans gösterdi.

Hatta eğitimini daha uzun süre (daha fazla epoch) sürdürürsek, değerlendirme metrikleri gelişmeye devam edebilir gibi görünüyor.


### 7.Memnun kalana kadar tekrarlayın

Burada devam edebilirdik. Modelimizin mimarisini yeniden yapılandırmak, daha fazla katman eklemek, denemek, öğrenme oranını ayarlamak, denemek, farklı veri büyütme yöntemlerini denemek, daha uzun süre eğitim. Ancak, hayal edebileceğiniz gibi, bu oldukça uzun zaman alabilir.

İyi ki henüz denemediğimiz bir numara var ve o da transfer öğrenme.

Bununla birlikte, kendi modellerimizi sıfırdan tasarlamak yerine başka bir modelin öğrendiği kalıplardan kendi görevimiz için nasıl yararlandığımızı göreceğiniz (bir sonraki eğitim yazısında).

Bu arada eğitimli çok sınıflı modelimiz ile bir tahminde bulunalım.

### Eğitim Modelimiz ile Tahmin Yapmak

Onunla tahmin yapamayacaksanız bir model ne işe yarar?

Önce kendimize çok sınıflı modelimizin üzerinde eğitim aldığı sınıfları hatırlatalım ve sonra çalışmak için kendi özel görüntülerinden bazılarını indireceğiz.


```python
class_names
```




    array(['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon',
           'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi'],
          dtype='<U14')



Güzel, şimdi özel resimlerimizden bazılarını alalım.

Google Colab kullanıyorsanız, dosyalar sekmesi aracılığıyla kendi resimlerinizden bazılarını da yükleyebilirsiniz.


```python
# görütüleri colab'e indirme

!wget -q "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/pizza_1.jpg"
!wget -q "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/pizza_2.jpg"
!wget -q "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/steak_1.jpeg"
!wget -q "https://raw.githubusercontent.com/Furkan-Gulsen/TensorFlow-ile-Yapay-Zeka-Gelistirme/main/3-Evri%C5%9Fimsel%20Sinir%20A%C4%9Flar%C4%B1%20(CNN)/images/steak_2.jpg"
```

Tamam, deneyecek bazı özel resimlerimiz var, hadi resimlerden birinde model_11 ile bir tahmin yapmak için pred_and_plot işlevini kullanalım ve onu çizelim.


```python
pred_and_plot(model=model_11, 
              filename="steak_1.jpeg", 
              class_names=class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_157_0.png)
    



```python
pred_and_plot(model_11, "pizza_1.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_158_0.png)
    



```python
pred_and_plot(model_11, "pizza_2.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_159_0.png)
    



```python
pred_and_plot(model_11, "steak_2.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_160_0.png)
    


Hespi yanlış çıktı. Pred_and_plot işlevimizle ilgili olabileceğini düşünüyorum.


```python
img = load_and_prep_image("steak_1.jpeg")

pred = model_11.predict(tf.expand_dims(img, axis=0))

pred_class = class_names[pred.argmax()]
plt.imshow(img)
plt.title(pred_class)
plt.axis(False);
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_162_0.png)
    


Çok daha iyi! pred_and_plot işlevimizde bir şeyler olmalı.

Ve sanırım ne olduğunu biliyorum.

pred_and_plot işlevi, mevcut modelimizin çok sınıflı bir sınıflandırma modeli olduğu ikili sınıflandırma modelleriyle kullanılmak üzere tasarlanmıştır.

Ana fark, tahmin fonksiyonunun çıktısında yatmaktadır.



```python
# Tahmin fonksiyonunun çıktısını kontrol edelim
pred = model_11.predict(tf.expand_dims(img, axis=0))
pred
```




    array([[0.03432237, 0.02044422, 0.05089506, 0.16825534, 0.06048798,
            0.04497578, 0.00705055, 0.10628431, 0.44707397, 0.06021047]],
          dtype=float32)



Modelimiz bir 'softmax' aktivasyon fonksiyonuna ve 10 çıkış nöronuna sahip olduğundan, modelimizde her sınıf için bir tahmin olasılığı verir.

En yüksek olasılığa sahip sınıf, modelin görüntünün içerdiğine inandığı sınıftır.

argmax kullanarak maksimum değer indeksini bulabilir ve ardından bunu, tahmin edilen sınıfın çıktısını almak için class_names listemizi indekslemek için kullanabiliriz.



```python
class_names[pred.argmax()]
```




    'steak'



pred_and_plot işlevimizi ikili sınıfların yanı sıra birden çok sınıfla çalışacak şekilde yeniden ayarlayabiliriz.


```python
def pred_and_plot(model, filename, class_names):
  img = load_and_prep_image(filename)
  
  pred = model.predict(tf.expand_dims(img, axis=0))

  if len(pred[0]) > 1: # ikili - çoklu kontrol
    pred_class = class_names[pred.argmax()] 
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])]

  plt.imshow(img)
  plt.title(f"Tahmin: {pred_class}")
  plt.axis(False);
```

Deneyelim. Doğru yaptıysak, farklı görüntüler kullanmak farklı çıktılara yol açmalıdır (her seferinde Chicken_curry yerine).


```python
pred_and_plot(model_11, "steak_2.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_170_0.png)
    



```python
pred_and_plot(model_11, "pizza_1.jpg", class_names)
```


    
![png](TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_files/TensorFlow_ile_Evri%C5%9Fimli_Sinir_A%C4%9Flar_%28CNN%29_171_0.png)
    


Modelimizin tahminleri çok iyi değil, bunun nedeni test veri setinde yalnızca ~%35 doğrulukta performans göstermesidir.

## Modelimizi Kaydetme ve Yükleme

Bir modeli eğittikten sonra, muhtemelen onu kaydedip başka bir yere yüklemek istersiniz.

Bunun için save ve load_model fonksiyonlarını kullanabiliriz.


```python
# bir modeli kaydetmek
model_11.save("saved_trained_model")
```

    INFO:tensorflow:Assets written to: saved_trained_model/assets



```python
# modeli yükleme ve değerlendirme
loaded_model_11 = tf.keras.models.load_model("saved_trained_model")
loaded_model_11.evaluate(test_data)
```

    79/79 [==============================] - 10s 130ms/step - loss: 1.7820 - accuracy: 0.3896





    [1.7820457220077515, 0.38960000872612]


