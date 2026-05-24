# Automatska klasifikacija IT support tiketa pomoću neuronskih mreža


Tema projekta je automatska klasifikacija tekstualnih IT support tiketa pomoću neuronskih mreža. Sistem na osnovu opisa tiketa predviđa kojoj kategoriji tiket pripada, na primer:

Ulaz: I cannot access my account after password reset.

Izlaz: Access

Projekat je implementiran u jednom Jupyter Notebook fajlu:

it_support_ticket_class.ipynb

Notebook obuhvata kompletan tok rada: preuzimanje podataka preko Kaggle API-ja, analizu i pripremu podataka, treniranje baseline modela, treniranje neuronskih modela, eksperimente sa hiperparametrima, evaluaciju, čuvanje najboljeg modela i predikciju novih tiketa.

---

## 1. Opis problema

Klasifikacija IT support tiketa predstavlja problem višeklasne klasifikacije teksta.

U realnom IT support sistemu korisnici šalju zahteve u slobodnoj tekstualnoj formi. Ti zahtevi se zatim ručno pregledaju i prosleđuju odgovarajućem timu, na primer timu za pristup nalogu, hardver, skladište, administrativna prava ili HR podršku.

Cilj ovog projekta je da se napravi model koji automatski određuje kategoriju tiketa na osnovu njegovog tekstualnog opisa.

Primer:

Tekst tiketa:

Printer is not working after update.

Predviđena kategorija:

Hardware

Ovakav sistem može da pomogne u:

* automatskom usmeravanju tiketa ka odgovarajućem timu,
* smanjenju ručnog rada,
* bržoj obradi korisničkih zahteva,
* boljoj organizaciji help desk sistema.

Model kao ulaz dobija tekst tiketa, a kao izlaz vraća jednu od unapred poznatih kategorija.

---

## 2. Podaci

### Izvor podataka

Za projekat je korišćen Kaggle dataset:

IT Service Ticket Classification Dataset

Dataset se preuzima direktno u Google Colab runtime pomoću Kaggle API-ja.

Korišćeni Kaggle dataset:

adisongoh/it-service-ticket-classification-dataset

Nakon preuzimanja koristi se CSV fajl:

all_tickets_processed_improved_v3.csv

Podaci se učitavaju iz foldera:

data/

Napomena: pošto se projekat izvršava u Google Colab okruženju, dataset se ne čuva lokalno trajno, već se svaki put preuzima u aktivni runtime.

### Struktura podataka

Dataset sadrži ukupno:

47.837 primera

2 kolone

Korišćene kolone su:

| Kolona      | Opis                           |
| ----------- | ------------------------------ |
| Document    | tekst IT support tiketa        |
| Topic_group | kategorija kojoj tiket pripada |

U notebook-u su ove kolone preimenovane u:

| Nova kolona | Značenje                    |
| ----------- | --------------------------- |
| text        | originalni tekst tiketa     |
| category    | ciljna kategorija           |
| clean_text  | očišćen tekst tiketa        |
| label       | numerička oznaka kategorije |

### Kategorije

Dataset sadrži 8 kategorija:

| ID | Kategorija            |
| -: | --------------------- |
|  0 | Access                |
|  1 | Administrative rights |
|  2 | HR Support            |
|  3 | Hardware              |
|  4 | Internal Project      |
|  5 | Miscellaneous         |
|  6 | Purchase              |
|  7 | Storage               |

Raspodela klasa u celom skupu:

| Kategorija            | Broj primera | Procenat |
| --------------------- | -----------: | -------: |
| Hardware              |       13.617 |   28.47% |
| HR Support            |       10.915 |   22.82% |
| Access                |        7.125 |   14.89% |
| Miscellaneous         |        7.060 |   14.76% |
| Storage               |        2.777 |    5.81% |
| Purchase              |        2.464 |    5.15% |
| Internal Project      |        2.119 |    4.43% |
| Administrative rights |        1.760 |    3.68% |

Iz raspodele se vidi da dataset nije potpuno balansiran. Najzastupljenije klase su Hardware i HR Support, dok su Administrative rights i Internal Project znatno manje zastupljene. Zbog toga je pored accuracy metrike posebno važna i metrika F1 macro, jer ona ravnopravnije posmatra sve klase.

### Analiza tekstova

Urađena je osnovna analiza dužine tekstova:

Prosečna dužina teksta: 43.60 reči

Minimalna dužina teksta: 2 reči

Maksimalna dužina teksta: 981 reči

U notebook-u su prikazani i grafikoni:

plots/class_distribution.png

plots/text_lengths.png

Prvi grafikon prikazuje raspodelu kategorija, a drugi histogram dužine tekstova.

### Preprocesiranje podataka

Pre treniranja modela urađeni su sledeći koraci:

1. učitavanje CSV fajla,
2. izbor kolona Document i Topic_group,
3. uklanjanje praznih vrednosti,
4. čišćenje teksta,
5. uklanjanje duplikata,
6. pretvaranje kategorija u numeričke labele,
7. stratifikovana podela na train, validation i test skup.

Funkcija za čišćenje teksta radi sledeće:

* pretvara tekst u mala slova,
* uklanja URL adrese,
* uklanja email adrese,
* uklanja specijalne karaktere,
* zadržava slova, brojeve i razmake,
* uklanja višestruke razmake.

Primer:

Originalni tekst:

Printer is not working after update.

Očišćen tekst:

printer is not working after update

Kategorije su pretvorene u numerički oblik pomoću LabelEncoder klase. Mapiranje između naziva kategorija i numeričkih labela čuva se u fajlu:

models/label_mapping.json

### Podela podataka

Podaci su podeljeni stratifikovano na:

| Skup             | Procenat | Broj primera |
| ---------------- | -------: | -----------: |
| Trening skup     |      70% |       33.485 |
| Validacioni skup |      15% |        7.176 |
| Test skup        |      15% |        7.176 |

Korišćena je stratifikovana podela kako bi raspodela kategorija ostala približno ista u svim skupovima.

---

## 3. Arhitektura modela

U projektu su implementirana i upoređena tri pristupa:

1. baseline model: TF-IDF + Logistic Regression,
2. osnovni neuronski model,
3. CNN model za tekstualnu klasifikaciju.

### Baseline model

Baseline model služi kao referentna tačka za poređenje sa neuronskim mrežama.

Korišćeni baseline pristup:

Tekst tiketa → TF-IDF vektorizacija → Logistic Regression → Predikcija kategorije

Parametri baseline modela:

TfidfVectorizer:

* max_features = 10.000
* ngram_range = (1, 2)

LogisticRegression:

* max_iter = 1000
* class_weight = "balanced"

Baseline model je važan zato što pokazuje koliko dobar rezultat može da se dobije jednostavnijim klasičnim pristupom. Ako neuronska mreža ne daje bolje ili barem uporedive rezultate, to znači da dodatna složenost modela nije opravdana.

### Osnovni neuronski model

Osnovni neuronski model koristi sledeću arhitekturu:

Sirov tekst tiketa → TextVectorization → Embedding → GlobalAveragePooling1D → Dense → Dropout → Dense softmax izlaz → Predviđena kategorija

Korišćeni slojevi:

| Sloj                   | Uloga                                      |
| ---------------------- | ------------------------------------------ |
| TextVectorization      | pretvara tekst u niz tokena                |
| Embedding              | pretvara tokene u vektorske reprezentacije |
| GlobalAveragePooling1D | pravi jednu reprezentaciju celog teksta    |
| Dense                  | uči nelinearne obrasce                     |
| Dropout                | smanjuje overfitting                       |
| Dense softmax          | vraća verovatnoće po klasama               |

Osnovna konfiguracija modela:

MAX_TOKENS = 10.000

SEQUENCE_LENGTH = 100

embedding_dim = 64

dense_units = 64

dropout_rate = 0.3

optimizer = Adam

loss = sparse_categorical_crossentropy

metrics = accuracy

Model direktno prima sirov tekst kao ulaz jer je TextVectorization sloj deo same Keras arhitekture.

### CNN model

Kao napredniji neuronski model implementiran je i CNN model za tekstualnu klasifikaciju.

Arhitektura CNN modela:

Sirov tekst tiketa → TextVectorization → Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Dense softmax izlaz → Predviđena kategorija

CNN model koristi konvolucioni sloj kako bi naučio lokalne obrasce u tekstu, na primer kombinacije reči koje se često pojavljuju kod određenih kategorija tiketa.

Korišćena konfiguracija CNN modela:

embedding_dim = 64

filters = 64

kernel_size = 5

dense_units = 64

dropout_rate = 0.4

sequence_length = 120

CNN model je na kraju ostvario najbolji F1 macro rezultat među neuronskim modelima.

---

## 4. Trening

Za treniranje neuronskih modela korišćen je TensorFlow/Keras.

Modeli su trenirani nad trening skupom, dok je validacioni skup korišćen za praćenje ponašanja modela tokom treniranja.

Korišćeni elementi treninga:

optimizer = Adam

loss = sparse_categorical_crossentropy

metrics = accuracy

batch_size = 32

Kod osnovnog neuronskog modela korišćeno je do 20 epoha:

epochs = 20

Kod eksperimenata i CNN modela korišćeno je do 10 epoha:

epochs = 10

### EarlyStopping

Korišćen je EarlyStopping callback kako bi se trening automatski zaustavio kada se validacioni gubitak više ne poboljšava.

Primer:

monitor = val_loss

patience = 2 ili 3

restore_best_weights = True

Ovo smanjuje rizik od overfitting-a i sprečava nepotrebno treniranje nakon što model prestane da se poboljšava.

### ModelCheckpoint

Korišćen je i ModelCheckpoint callback kako bi se sačuvala najbolja verzija modela prema validacionom gubitku.

Sačuvani modeli nalaze se u folderu:

models/

Najbolji model je sačuvan kao:

models/best_ticket_classifier.keras

### Grafikoni treninga

Za svaki neuronski model prikazuju se grafikoni:

* trening i validaciona tačnost,
* trening i validacioni gubitak.

Grafikoni se čuvaju u folderu:

plots/


---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

U projektu je urađena analiza osetljivosti kroz više eksperimenata sa različitim hiperparametrima neuronske mreže.

Menjani su sledeći hiperparametri:

* dimenzija Embedding sloja,
* broj neurona u Dense sloju,
* vrednost Dropout regularizacije,
* maksimalna dužina ulazne sekvence.

Eksperimentalne konfiguracije:

| Model                               | Embedding dim | Dense units | Dropout | Sequence length |
| ----------------------------------- | ------------: | ----------: | ------: | --------------: |
| NN_emb32_dense64_dropout03_len100   |            32 |          64 |     0.3 |             100 |
| NN_emb64_dense64_dropout03_len100   |            64 |          64 |     0.3 |             100 |
| NN_emb128_dense128_dropout05_len100 |           128 |         128 |     0.5 |             100 |
| NN_emb64_dense128_dropout05_len150  |            64 |         128 |     0.5 |             150 |
| CNN_Text_Model                      |            64 |          64 |     0.4 |             120 |

Za svaki eksperiment pravljen je poseban TextVectorization sloj, posebno adaptiran nad trening tekstovima. Ovo je važno zato što se promenom sequence_length menja i oblik ulaza u model.

Cilj eksperimenata bio je da se proveri kako promene arhitekture utiču na kvalitet klasifikacije.

Rezultati pokazuju da povećanje složenosti modela ne mora automatski da dovede do boljeg rezultata. Na primer, modeli sa većim embedding slojem i većim brojem neurona nisu nužno bili bolji od jednostavnijih modela. Najbolji rezultat po F1 macro metrici ostvario je CNN model.

---

## 6. Rezultati evaluacije

Za evaluaciju modela korišćen je test skup koji nije korišćen tokom treniranja.

Korišćene metrike:

| Metrika          | Objašnjenje                                       |
| ---------------- | ------------------------------------------------- |
| accuracy         | procenat ukupno tačno klasifikovanih tiketa       |
| precision macro  | prosečna preciznost po klasama                    |
| recall macro     | prosečan odziv po klasama                         |
| F1 macro         | balans između precision i recall po klasama       |
| confusion matrix | prikaz najčešćih tačnih i pogrešnih klasifikacija |

Glavna metrika za poređenje modela je F1 macro, zato što dataset nije potpuno balansiran.

### Poređenje svih modela

| Model                               | Accuracy | Precision macro | Recall macro | F1 macro |
| ----------------------------------- | -------: | --------------: | -----------: | -------: |
| CNN_Text_Model                      |   0.8477 |          0.8633 |       0.8444 |   0.8535 |
| TF-IDF + Logistic Regression        |   0.8519 |          0.8342 |       0.8736 |   0.8506 |
| NN_emb32_dense64_dropout03_len100   |   0.8432 |          0.8799 |       0.8110 |   0.8405 |
| Osnovni neuronski model             |   0.8400 |          0.8803 |       0.8082 |   0.8369 |
| NN_emb128_dense128_dropout05_len100 |   0.8303 |          0.8813 |       0.7836 |   0.8198 |
| NN_emb64_dense64_dropout03_len100   |   0.8305 |          0.8817 |       0.7766 |   0.8166 |
| NN_emb64_dense128_dropout05_len150  |   0.8158 |          0.8749 |       0.7649 |   0.8048 |

Najbolji model po F1 macro metrici je:

CNN_Text_Model

Njegovi rezultati na test skupu:

accuracy = 0.8477

precision macro = 0.8633

recall macro = 0.8444

F1 macro = 0.8535

Baseline model TF-IDF + Logistic Regression ostvario je:

accuracy = 0.8519

F1 macro = 0.8506

To znači da baseline model ima neznatno viši accuracy, dok CNN model ima blago bolji F1 macro.

### Rezultati najboljeg sačuvanog modela po klasama

Najbolji sačuvani model je CNN_Text_Model.

Klasifikacioni izveštaj na test skupu:

| Kategorija            | Precision | Recall | F1-score | Support |
| --------------------- | --------: | -----: | -------: | ------: |
| Access                |      0.89 |   0.86 |     0.88 |    1069 |
| Administrative rights |      0.81 |   0.75 |     0.78 |     264 |
| HR Support            |      0.86 |   0.85 |     0.86 |    1638 |
| Hardware              |      0.81 |   0.86 |     0.83 |    2042 |
| Internal Project      |      0.91 |   0.86 |     0.88 |     318 |
| Miscellaneous         |      0.82 |   0.80 |     0.81 |    1059 |
| Purchase              |      0.91 |   0.89 |     0.90 |     370 |
| Storage               |      0.90 |   0.88 |     0.89 |     416 |

Ukupni rezultat:

accuracy = 0.85

macro avg F1-score = 0.85

weighted avg F1-score = 0.85

### Confusion matrix

Za svaki model prikazana je confusion matrix. Ona pokazuje koje klase model najčešće dobro prepoznaje, a koje klase najčešće meša.

Najčešća greška najboljeg modela javlja se između kategorija:

HR Support

Hardware

Ovo je očekivano zato što pojedini tiketi mogu sadržati slične tehničke izraze, pa model ponekad meša zahteve koji se odnose na hardver sa opštim IT/HR support zahtevima.

### Predikcija novih tiketa

Na kraju projekta implementirana je funkcija:

predict_ticket_category(text, model, label_mapping, top_k=3)

Funkcija vraća:

* originalni tekst,
* očišćen tekst,
* predviđenu kategoriju,
* sigurnost modela,
* top-k najverovatnijih kategorija.

Primer rezultata:

Input: I cannot enter my profile after password reset.

Očišćen tekst: i cannot enter my profile after password reset

Predicted category: Access

Confidence: 98.56%

Top 3 predikcije:

1. Access - 98.56%
2. Hardware - 1.33%
3. Administrative rights - 0.07%

Još jedan primer:

Input: Printer is not working after update.

Očišćen tekst: printer is not working after update

Predicted category: Hardware

Confidence: 48.17%

Top 3 predikcije:

1. Hardware - 48.17%
2. Administrative rights - 38.16%
3. Access - 7.26%

---

## 7. Diskusija

Rezultati pokazuju da se klasifikacija IT support tiketa može uspešno rešiti i klasičnim ML pristupom i neuronskim mrežama.

Baseline model TF-IDF + Logistic Regression ostvario je veoma dobar rezultat. To pokazuje da je za ovaj dataset reprezentacija teksta pomoću TF-IDF vrednosti veoma informativna, jer se kategorije često mogu prepoznati na osnovu karakterističnih reči i fraza.

Sa druge strane, CNN model je ostvario najbolji F1 macro, što znači da je u proseku dao nešto uravnoteženiji rezultat po klasama. CNN model može da nauči lokalne obrasce u tekstu, odnosno kraće kombinacije reči koje su važne za određenu kategoriju.

Važno zapažanje je da složeniji neuronski modeli nisu automatski dali bolje rezultate. Eksperimenti sa većim embedding slojem, većim brojem neurona i dužom sekvencom nisu nadmašili CNN model. To pokazuje da kod neuronskih mreža nije dovoljno samo povećati broj parametara, već je potrebno pažljivo podešavati arhitekturu i pratiti validacione rezultate.

Takođe postoji blag signal overfitting-a. Razlika između trening i validacione tačnosti kod najboljeg modela pokazuje da model delimično bolje pamti trening podatke nego što generalizuje na validacione podatke. Zbog toga su korišćeni Dropout, EarlyStopping i ModelCheckpoint.

Još jedan izazov je nebalansiranost klasa. Klase kao što su Hardware i HR Support imaju mnogo više primera od klasa Administrative rights i Internal Project. Zbog toga accuracy nije dovoljna metrika, jer model može imati dobru ukupnu tačnost, a slabije rezultate na manjim klasama. Zato je F1 macro korišćen kao glavna metrika za poređenje.

Moguća unapređenja projekta:

* dodatno balansiranje klasa,
* bolja obrada domenskih IT termina,
* korišćenje naprednije tokenizacije,
* dodavanje lematizacije ili stemovanja,
* testiranje LSTM ili BiLSTM arhitekture,
* korišćenje pretreniranih jezičkih modela kao što su BERT ili DistilBERT,
* detaljnija analiza pogrešno klasifikovanih primera,
* dodatna regularizacija modela.

---

## 8. Zaključak

U ovom projektu je implementiran kompletan sistem za automatsku klasifikaciju IT support tiketa pomoću neuronskih mreža.

Korišćen je Kaggle dataset IT Service Ticket Classification Dataset, odnosno CSV fajl all_tickets_processed_improved_v3.csv. Dataset sadrži 47.837 primera i 8 kategorija.

Tok projekta obuhvata:

1. preuzimanje dataseta preko Kaggle API-ja,
2. učitavanje i analizu podataka,
3. čišćenje tekstualnih podataka,
4. kodiranje kategorija,
5. podelu podataka na trening, validacioni i test skup,
6. implementaciju baseline modela,
7. implementaciju osnovnog neuronskog modela,
8. implementaciju CNN modela,
9. eksperimente sa hiperparametrima,
10. evaluaciju modela,
11. čuvanje najboljeg modela,
12. predikciju kategorije za nove tikete.

Najbolji neuronski model je:

CNN_Text_Model

Ostvareni rezultati najboljeg modela:

accuracy = 0.8477

precision macro = 0.8633

recall macro = 0.8444

F1 macro = 0.8535

Baseline model TF-IDF + Logistic Regression ostvario je veoma slične rezultate, čak i nešto bolji accuracy, ali CNN model ima blago bolji F1 macro.

Na osnovu rezultata može se zaključiti da neuronske mreže mogu uspešno da se koriste za klasifikaciju IT support tiketa, ali i da jednostavniji baseline modeli mogu biti veoma konkurentni kod tekstualnih problema gde su ključne reči snažno povezane sa klasama.

Krajnji rezultat projekta je sačuvan model koji može direktno da primi novi tekst tiketa i da vrati predviđenu kategoriju, procenat sigurnosti i nekoliko najverovatnijih kategorija.

---

## Pokretanje projekta

Projekat je namenjen za pokretanje u Google Colab okruženju.

### 1. Otvoriti notebook

Otvoriti fajl:

it_support_ticket_class.ipynb

### 2. Podesiti Kaggle API

Za preuzimanje dataseta potreban je Kaggle API nalog.

Potrebno je podesiti sledeće paremetre unutar notebook fajla:

KAGGLE_USERNAME

KAGGLE_KEY

Postupak za dobijanje API ključa:
1. Nakon logovanja na kaggle.com potrebno je kliknuti na profilnu sliku gore desno
2. Klik na "Your API tokens"
3. Klik na "Generate New Token"
4. Uneti ime tokena
5. Klik na Generate


### 3. Pokrenuti sve ćelije

U Google Colab-u izabrati:

Runtime → Run all

Notebook će zatim:

* instalirati/proveriti Kaggle biblioteku,
* preuzeti dataset,
* učitati podatke,
* trenirati modele,
* prikazati rezultate,
* sačuvati grafikone,
* sačuvati najbolji model.

### 4. Izlazni folderi

Tokom izvršavanja kreiraju se sledeći folderi:

data/

models/

outputs/

plots/

Njihova uloga:

| Folder   | Opis                                 |
| -------- | ------------------------------------ |
| data/    | preuzeti dataset                     |
| models/  | sačuvani modeli i mapiranje labela   |
| outputs/ | tabele rezultata                     |
| plots/   | grafikoni i confusion matrix prikazi |

Najvažniji izlazni fajlovi:

models/best_ticket_classifier.keras

models/label_mapping.json

models/best_model_config.json

outputs/model_results.csv

plots/model_comparison_accuracy.png

plots/model_comparison_f1.png

plots/best_saved_model_confusion_matrix.png

---

## Korišćene biblioteke

Glavne korišćene biblioteke:

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Kaggle API

Verzije korišćene u notebook-u:

Python: 3.12.13

TensorFlow: 2.20.0

pandas: 2.2.2

scikit-learn: 1.6.1
