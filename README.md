# fzmapp

Ankara Üniversitesi Fizik Mühendisliği akademisyenlerini araştırma konusuna göre bulan arama sayfası.

## Akademisyen fotoğrafları

Sayfa her akademisyen için önce depodaki `photos/<profil>.jpg` dosyasını, bulamazsa `data/faculty.json` içindeki AVESİS adresini dener. İkisi de yüklenemezse baş harfler gösterilir.

AVESİS fotoğrafları başka sitelerde doğrudan gösterilemeyebilir. Fotoğrafları kesin göstermek için her birini `photos/` klasörüne profil adıyla kaydedin. Profil adları `data/faculty.json` dosyasındaki `profile` alanıdır; örneğin `photos/mkabak.jpg` veya `photos/101269.jpg`.
