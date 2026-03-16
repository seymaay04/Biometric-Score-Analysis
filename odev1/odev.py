import numpy as np
import matplotlib.pyplot as plt

# Dosyayı yüklüyor ve dosyanın içeriğindeki ilk array'i alıyoruz.
data = np.load('Features.npz')

# İçindeki dosyaların listesinden ilkini alıyoruz
features = data[data.files[0]] 

# Sadece ilk 100 kişiye ait verileri almamız yani son 10 kişiyi dahil etmememiz istenmişti
features_100 = features[:, :100, :]

# Tüm değerleri [0, 1] aralığında normalize etmemiz gerekiyor bunun için formülümüz (Değer - Min) / (Max - Min)
min_val = np.min(features_100)
max_val = np.max(features_100)

features_norm = (features_100 - min_val) / (max_val - min_val)

# Skor Hesaplama

genuine_scores = []  # Aynı kişinin olan skorlar
imposter_scores = [] # Farklı kişiler olan skorlar


# features_norm verimiz 3 boyutlu bir yapıya sahip: (10 zaman, 100 kişi, 6 özellik)
# shape komutu bu boyutların listesini verir. Biz de bu listeden ihtiyacımız olanları alıyoruz

num_samples = features_norm.shape[0] # shape[0] listenin ilk elemanıdır yani her kişiden kaç farklı zamanda örnek alındığını (10) verir
num_persons = features_norm.shape[1] # shape[1] listenin ikinci elemanıdır yani veri setindeki toplam kişi sayısını (100) verir

# Döngü ile mesafeleri ve skorları hesaplıyoruz
for p1 in range(num_persons):
    for s1 in range(num_samples):
        vec1 = features_norm[s1, p1]
        
        # Aynı kişinin farklı zamanlardaki örnekleri karşılaştırılır
        # Çiftleri tekrar hesaplamamak için s1+1'den başlıyoruz
        for s2 in range(s1 + 1, num_samples):
            vec2 = features_norm[s2, p1]
            
            # Öklid mesafesi hesaplama
            euclidean_dist = np.linalg.norm(vec1 - vec2)

            # Skor formülü ödev belgesinde verilmişti: 1 / (1 + Öklid Mesafesi)
            score = 1 / (1 + euclidean_dist)
            genuine_scores.append(score)
            
        # Farklı kişilerin örnekleri karşılaştırılır
        # Çiftleri tekrar hesaplamamak için p1+1'den başlıyoruz
        for p2 in range(p1 + 1, num_persons):
            for s2 in range(num_samples):
                vec2 = features_norm[s2, p2]
                
                euclidean_dist = np.linalg.norm(vec1 - vec2)
                score = 1 / (1 + euclidean_dist)
                imposter_scores.append(score)

genuine_scores = np.array(genuine_scores)
imposter_scores = np.array(imposter_scores)


# FAR, FRR ve EER Hesaplama

# Skorlarımız 0 ile 1 arasında olduğu için 0'dan 1'e kadar 1000 adet eşik değeri oluşturuyoruz
thresholds = np.linspace(0, 1, 1000)
far = []
frr = []

for t in thresholds:
    # Yanlış Kabul (False Accept), sahteci skorun eşik değerinden büyük veya eşit olması
    false_accepts = np.sum(imposter_scores >= t)
    # Yanlış Ret (False Reject), gerçek skorun eşik değerinden küçük olması
    false_rejects = np.sum(genuine_scores < t)
    
    # Oranları bulup listelere ekliyoruz
    far.append(false_accepts / len(imposter_scores))
    frr.append(false_rejects / len(genuine_scores))

far = np.array(far)
frr = np.array(frr)

# Eşit Hata Oranı (EER), FAR ve FRR'nin birbirine eşit olduğu veya en yakın olduğu noktadır. İki dizi arasındaki mutlak farkın en küçük olduğu indeksi buluyoruz
diff = np.abs(far - frr)
eer_index = np.argmin(diff)

eer_threshold = thresholds[eer_index]
# FAR ve FRR tam kesişmeyebilir bu yüzden o noktadaki değerlerin ortalamasını almamız gerekiyor
eer_value = (far[eer_index] + frr[eer_index]) / 2 

print(f"Hesaplanan EER Değeri: {eer_value:.4f} (Eşik: {eer_threshold:.4f})")


# Grafikler

# 3 satır, 1 sütundan oluşan figür oluşturuyoruz
fig, axs = plt.subplots(3, 1, figsize=(8, 14))


# Grafik 1: Gerçek ve sahteci skor dağılımları
axs[0].hist(imposter_scores, bins=50, alpha=0.6, color='red', label='Sahteci Skor (Imposter)', density=True)
axs[0].hist(genuine_scores, bins=50, alpha=0.6, color='blue', label='Gerçek Skor (Genuine)', density=True)
axs[0].set_title('Gerçek ve Sahteci Skor Dağılımları')
axs[0].set_xlabel('Eşleşme Skoru')
axs[0].set_ylabel('Yoğunluk')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)


# Grafik 2: Eşik değerlerine karşı FAR ile FRR değerlerinin değişimi ve ERR değeri
axs[1].plot(thresholds, far, color='red', label='FAR (Yanlış Kabul Oranı)', linewidth=2)
axs[1].plot(thresholds, frr, color='blue', label='FRR (Yanlış Ret Oranı)', linewidth=2)
# EER noktasını grafikte işaretliyoruz
axs[1].plot(eer_threshold, eer_value, 'ko', markersize=8, label=f'EER Noktası ({eer_value:.4f})')
axs[1].set_title('Eşik Değerlerine Göre FAR ve FRR Değişimi')
axs[1].set_xlabel('Eşik Değeri')
axs[1].set_ylabel('Hata Oranı')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

# Grafik 3: FAR değerlerine karşı FRR değerlerinin değişimi
axs[2].plot(far, frr, color='purple', linewidth=2, label='FRR vs FAR')
# EER noktasını, FAR ve FRR'nin eşit olduğu y=x doğrusu veya o noktadaki değeri ile gösterebiliriz
axs[2].plot(eer_value, eer_value, 'ko', markersize=8, label=f'EER Noktası ({eer_value:.4f})')
axs[2].set_title('FAR Değerlerine Karşı FRR Değişimi')
axs[2].set_xlabel('FAR (Yanlış Kabul Oranı)')
axs[2].set_ylabel('FRR (Yanlış Ret Oranı)')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.5)

# Grafikleri ekranda gösterip birbirine girmemesi için boşlukları ayarlıyoruz
plt.tight_layout(h_pad=2.0)
plt.show()