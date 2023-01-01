
#########################################################################################################################
#########################################################################################################################

# Gerekli Kütüphaneler ve Ayarlar
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 800)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

############################################################################################################
############################################################################################################

# DEĞİŞKENLER
# master_id : Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

############################################################################################################
############################################################################################################

#### GÖREV 1: VERİYİ HAZIRLAMA ####
# Adım1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("crm_analytics-220908-141436/FLO_CLTV_Tahmini-221020-121219/FLO_CLTV_Tahmini/flo_data_20k.csv")
df = df_.copy()

# Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

df.describe().T
df.quantile([0.01, 0.1, 0.2, 0.5, 0.7, 0.95, 0.96,0.97,0.98, 0.99, 1]).T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range)
    low_limit = (quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


# Adım3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayınız.
outlier = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for i in outlier:
    replace_with_thresholds(df, i)

# Adım4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
for i in df.columns:
    if 'date' in i:
        df[i] = pd.to_datetime(df[i])

#### GÖREV 2: CLTV VERİ YAPISININ OLUŞTURULMASI ####
# Adım1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df.last_order_date.max() # Timestamp('2021-05-30 00:00:00')
analysis_date = df.last_order_date.max() + dt.timedelta(days=2)

# Adım2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv = pd.DataFrame()
cltv["customer_id"] = df.master_id
cltv["recency_cltv_weekly"] = ((df.last_order_date - df.first_order_date).astype('timedelta64[D]')) / 7
cltv["T_weekly"] = ((analysis_date - df.first_order_date).astype('timedelta64[D]')) / 7
cltv["frequency"] = df.order_num_total_ever
cltv["monetary_cltv_avg"] = df.customer_value_total_ever / df.order_num_total_ever
cltv.head()

#### GÖREV 3: BG/NBD, GAMMA-GAMMA MODELİNİN KURULMASI VE CLTV'NİN HESAPLANMASI ####
# Adım1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

#• 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
#dataframe'ine ekleyiniz.
cltv["exp_sales_3_month"] = bgf.predict(4*3,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])

#• 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
#dataframe'ine ekleyiniz.
cltv["exp_sales_6_month"] = bgf.predict(4*6,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])


# Adım2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
#dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"],
                                                                    cltv["monetary_cltv_avg"])


# Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#• Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                            cltv["frequency"],
                                            cltv["recency_cltv_weekly"],
                                            cltv["T_weekly"],
                                            cltv["monetary_cltv_avg"],
                                            time = 6, # 6 aylık
                                            freq = "W", # T'nin frekansı
                                            discount_rate = 0.01)

cltv.sort_values(by="cltv", ascending=False).head(20)



#### GÖREV 4: CLTV DEĞERİNE GÖRE SEGMENTLERİN OLUŞTURULMASI ####

# Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.


cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
cltv.head()

cltv.groupby("segment").agg({"count", "mean", "std", "sum"})
cltv.columns

# Adım2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv.groupby("segment").agg({"exp_sales_6_month": ["count", "mean", "std", "sum"]})

# A grubuna ait müşterilerin 6 aylık CLTV değeri en yüksek. Bu müşterilerin 6 aylık satın alma sayısı ortalama 1.5, standart sapması 0.5.
# D grubuna ait müşterilerin 6 aylık CLTV değeri ise en düşük. Bu müşterilerin 6 aylık satın alma sayısı ortalama 0.5, standart sapması 0.2.

# A,B,C segmentlerindeki ortalama satın alma sayıları birbirini çok yakın olduğu için müşteriler birleştirilerek yeni bir segment oluşturulabilir.
