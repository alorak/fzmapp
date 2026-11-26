# FZM Akademik Uzmanlık Arama

Ankara Üniversitesi Fizik Mühendisliği Bölümünde araştırma konunuzla ilgili yayın veya tez çalışması olan akademisyenleri bulun. Uygulama AVESİS’ten derlenen bölüm verisini ilk açılışta yükler. Depodaki veri seti 23 akademisyene ait 1.815 yayın ve 168 tez kaydı içerir. [Uygulamayı aç](https://alorak.github.io/fzmapp/#arama)

## Kullanım

- **Arama.** Türkçe veya İngilizce bir araştırma konusu yazın. Akademisyenler, çalışmalarının sorguya uygunluğuna göre sıralanır. Her sonuç kartı en yakın üç çalışmayı gösterir. Arama ekranındaki **anlam parmak izi**, sorgunuzun ve en yakın kaydın gerçek model vektörlerini yan yana gösterir. [Arama](https://alorak.github.io/fzmapp/#arama)
- **Akademisyen sayfası.** Bir akademisyenin adına tıklayarak kayıtlarını açın. Yayın ve tezleri ayrı ayrı görüntüleyebilir, başlıklarda arayabilir, kayıtları tarihe veya son aramanıza uygunluğa göre sıralayabilirsiniz.
- **Makaleyi sor / Tezi sor.** Çalışmanın özetini ve kaynak bağlantılarını açar. Özet bulunduğunda, WebGPU destekli bir tarayıcıda sohbet modelini yükleyip özet hakkında soru sorabilirsiniz.
- **Veri kaynağı.** Akademisyen listesini ve kayıt sayılarını gösterir. Kendi JSON dosyanızı yükleyebilir veya **Bölüm verisini yükle** ile depodaki verilere dönebilirsiniz. [Veri kaynağı](https://alorak.github.io/fzmapp/#veri)

## Modeller ve sıralama

- **Arama modeli.** Multilingual E5 Small, sorguyu ve kayıtların başlıklarını, kayıt içinde varsa özetleriyle birlikte, 384 boyutlu vektörlere dönüştürür. Modelin 8 bitlik ONNX ağırlıkları yaklaşık **118 MB** ve ilk aramada indirilir. Arama ekranı modelin durumunu, indirme ve dizin hazırlama ilerlemesini gösterir. `search-worker.js` modeli WebAssembly ile arka planda çalıştırır. [Model kartı](https://huggingface.co/intfloat/multilingual-e5-small) · [ONNX ağırlıkları](https://huggingface.co/Xenova/multilingual-e5-small/blob/main/onnx/model_quantized.onnx)
- **Uygunluk.** Kayıtlar kosinüs benzerliğine göre karşılaştırılır. Akademisyen skoru, en yakın kaydın benzerliğinin %60’ı ile en yakın en fazla beş kaydın ortalama benzerliğinin %40’ından oluşur. Ekrandaki uygunluk yüzdesi, sonucun aynı sorgudaki kayıtların medyan benzerliğine göre ne kadar öne çıktığını gösterir. [Nasıl çalışır?](https://alorak.github.io/fzmapp/#nasil-calisir)
- **Sohbet modeli.** `chat-worker.js`, Qwen3 0.6B’nin yaklaşık **570 MB** boyutundaki ONNX sürümünü WebGPU ile çalıştırır. İlk indirme **Modeli yükle ve başla** düğmesiyle başlar. Model, yanıtı çalışmanın başlığına, mevcut anahtar kelimelerine ve özetine dayandırması için yönlendirilir. [Sohbet modelinin ONNX dosyaları](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/tree/main/onnx)

- **Eğik atış modeli.** [`models/egik-atis.onnx`](models/egik-atis.onnx), hava dirençli eğik atış için eğitilmiş küçük bir fizik bilgili sinir ağıdır (PINN, 4.994 parametre, yaklaşık 21 KB). Girdileri boyutsuz zaman `t·g/v0`, atış açısı ve boyutsuz direnç `k·v0²/g`; çıktısı `v0²/g` biriminde konumdur. Bu sayede tek model her ilk hız için çalışır. Eğitimde örnek yörünge kullanılmaz, yalnızca hareket denklemlerinin hatası küçültülür. `egik-atis.js` modeli [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) ile tarayıcıda çalıştırır ve sonucu havasız formül ve Runge–Kutta çözümüyle karşılaştırır. Doğruluk ölçümleri ve eğitim kaydı (kayıp eğrisi, eğitim sırasındaki ara yörüngeler, açı ve dirence göre hata haritası) [`models/egik-atis.json`](models/egik-atis.json) dosyasındadır; [Nasıl çalışır? › Eğik atış modeli](https://alorak.github.io/fzmapp/#nasil-calisir/egik-atis) sekmesi bu kayıttan çizilir. Yeniden eğitmek için: `pip install -r scripts/projectile/requirements.txt` ve `python -m scripts.projectile.train_pinn` (CPU’da yaklaşık 6 dakika).

İki model de sorguları ve sohbet mesajlarını cihazınızda işler. Model dosyaları internetten yüklenir. Tarayıcı, içe aktarılan kayıtları ve arama dizinini IndexedDB’de, model ağırlıklarını kendi önbelleğinde saklar. Çalışma ayrıntısında bir yayın özeti aranırken DOI önce OpenAlex’e, gerekirse Crossref’e gönderilir.

## Veri dosyaları

| Dosya | İçerik |
| --- | --- |
| [`data/academics.json`](data/academics.json) | Uygulamayla gelen yayın ve tez kayıtları |
| [`data/profiles.json`](data/profiles.json) | Toplayıcının kullandığı AVESİS profil kimlikleri |
| [`data/faculty.json`](data/faculty.json) | Akademisyen adları, AVESİS profilleri ve fotoğraf adresleri |
| [`data/abstracts.json`](data/abstracts.json) | AVESİS’ten derlenen tez özetleri |
| `data/years.json` | Yıl toplama iş akışının ürettiği isteğe bağlı yayın ve tez yılları. Kaydın kendi `year` alanı önceliklidir. |
| [`photos/`](photos/) | Yerel profil fotoğrafları. Uygulama önce `photos/<profil>.jpg`, sonra `data/faculty.json` içindeki fotoğraf adresini dener. Son seçenek akademisyenin baş harfleridir. |

İçe aktarma dosyası en fazla 20.000 kayıt içeren bir JSON dizisidir. Her kayıtta `faculty` ve `title` zorunludur. İsteğe bağlı alanlar `url`, `kind` (`publication` veya `thesis`), `abstract`, `doi`, `keywords` ve tam sayı `year` değeridir. Çalışma ayrıntısı önce kaydın kendi özetini, sonra mevcutsa yerel tez özetini kullanır. DOI bulunan kayıtlarda açık kaynaklardan özet arayabilir.

## Yerel çalıştırma

Depo kökünde aşağıdaki komutu çalıştırın ve [http://localhost:8000/#arama](http://localhost:8000/#arama) adresini açın. Statik arayüz `index.html` içinden çalışır. Python bağımlılıkları veri toplama ve testler için gerekir.

```sh
python3 -m http.server 8000
```

Veri toplama betiklerini ve testleri Python 3.12 ile çalıştırabilirsiniz.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Tek profille canlı bir toplama denemesi için aşağıdaki örnek çıktıları geçici bir klasöre yazar.

```sh
python scripts/collect.py --profile mkabak --output-dir /tmp/fzmapp-pilot
```

## Veri güncelleme

**Collect AVESIS metadata** yalnızca elle çalıştırılır. Tam kadro için `profile` alanını boş bırakın. Varsayılan kaynak `data/profiles.json` listesidir. Bölüm sitesinden profil keşfi, `discover_department` seçeneğiyle istenebilir ve sitenin tarama kurallarına tabidir. Toplama betiği AVESİS’in `robots.txt` kurallarını ve istek aralığını uygular. [GitHub Actions](https://github.com/alorak/fzmapp/actions)

İş akışı `academics.json` ile `faculty.json` dosyalarını **academics** çıktısına, profil bazındaki kayıt sayılarını ve hataları **collection-report** çıktısına koyar. Yayındaki veri setini güncellemek için kapsam raporunu kontrol edin ve doğrulanmış tam kadro dosyalarını bir PR ile `data/` dizinine ekleyin.

| Yardımcı iş akışı | Çıktı | Tetikleyici |
| --- | --- | --- |
| [Fetch AVESIS profile photos](.github/workflows/photos.yml) | `photos/` | Elle veya `photos/**` dallarına fotoğraf betiği, akademisyen verisi ya da ilgili iş akışı dosyası gönderildiğinde |
| [Collect AVESIS thesis abstracts](.github/workflows/abstracts.yml) | `data/abstracts.json` | Elle veya `abstracts/**` dallarına ilgili betik ya da iş akışı dosyası gönderildiğinde |
| [Collect AVESIS publication years](.github/workflows/years.yml) | `data/years.json` | Elle veya `abstracts/**` dallarına ilgili betik ya da iş akışı dosyası gönderildiğinde |

Yardımcı iş akışları ürettikleri değişiklikleri çalıştıkları dala kaydeder.
