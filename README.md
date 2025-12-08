# Logistic Regression

Bu proje, Python kullanılarak geliştirilmiş kapsamlı bir **Lojistik Regresyon** (Logistic Regression) uygulamasını içerir. Model eğitimi, test işlemleri ve sonuçların görselleştirilmesi için modüler bir yapıda tasarlanmıştır.

## 📂 Proje Yapısı

Proje dosyaları ve görevleri aşağıdaki gibidir:

* **`main.py`**: Projenin ana çalışma dosyasıdır. Modeli başlatır, veri setini yükler ve eğitimi tetikler.
* **`logistic_regression.py`**: Lojistik Regresyon algoritmasının matematiksel hesaplamalarını ve sınıf yapısını (Model Class) barındırır.
* **`options.py`**: Modelin hiperparametrelerini (learning rate, iterasyon sayısı vb.) ve komut satırı argümanlarını yönetir.
* **`plot.py`**: Eğitim kaybı (loss), doğruluk (accuracy) grafikleri ve karar sınırlarını (decision boundary) çizdirmek için kullanılır.
* **`datasets/`**: Eğitim ve test için kullanılan veri seti dosyalarını içerir.
* **`results/`**: Modelin eğitim sonrası çıktıları ve kaydedilen grafiklerin tutulduğu dizindir.

## ⚙️ Gereksinimler (Requirements)

Projeyi çalıştırmadan önce aşağıdaki Python kütüphanelerinin yüklü olduğundan emin olun:

* Python 3.x
* NumPy
* Matplotlib
* Pandas (Veri işleme için gerekliyse)

Gerekli paketleri yüklemek için:

```bash
pip install numpy matplotlib pandas
