# Biyometrik Sistem Performans Analizi

Bu proje, Python kullanılarak biyometrik sistemlerin performansını değerlendirmek amacıyla geliştirilmiş bir analiz ve görselleştirme uygulamasıdır. Proje kapsamında, numpy dizisi (array) formatındaki biyometrik özellik vektörleri üzerinde işlemler gerçekleştirilmiştir.

## Proje İçeriği

*   **Veri İşleme ve Normalizasyon:** `Features.npz` dosyasından okunan 100 kişiye ait, her biri 6 elemanlı özellik vektörleri işlenmiş ve değerleri [0, 1] aralığına normalize edilmiştir.
*   **Skor Hesaplama:** Özellik vektörleri arasındaki Öklid mesafesi baz alınarak skor hesaplamaları yapılmıştır.
*   **Dağılım Analizi:** Hesaplanabilen tüm gerçek (Genuine) ve sahteci (Imposter) skorlar bulunarak dağılımları elde edilmiş ve aynı grafik üzerinde görselleştirilmiştir.
*   **Performans Metrikleri:** Sistem performansını değerlendirmek için farklı eşik (threshold) değerlerine karşılık gelen Yanlış Kabul Oranı (FAR - False Acceptance Rate) ve Yanlış Ret Oranı (FRR - False Reject Rate) hesaplanmıştır.
*   **EER ve Görselleştirme:** Eşit Hata Oranı (EER - Equal Error Rate) bulunmuş, FAR-FRR değişimi ve EER değeri tek bir grafikte gösterilmiş, ayrıca FAR değerlerine karşılık FRR değişimini gösteren çizimler oluşturulmuştur.

## Kullanılan Teknolojiler
*   **Programlama Dili:** Python
*   **Kütüphaneler:** NumPy (vektörel işlemler, veri normalizasyonu ve Öklid mesafesi hesaplamaları için), Matplotlib (skor dağılımları ile FAR/FRR/EER grafiklerinin çizimi için)
