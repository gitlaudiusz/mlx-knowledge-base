# MLX Fine-tuning Kompendium

## Spis treści

### Podstawy Fine-Tuningu
- [Wprowadzenie do dostrajania modeli językowych](#wprowadzenie-do-dostrajania-modeli-językowych-lm)
- [Metody dostrajania parametrów modeli AI](#metody-dostrajania-parametrów-modeli-ai-z-otwartymi-wagami)
- [Kluczowe pojęcia w ML i fine-tuning](#opis-i-omówienie-kluczowych-pojęć)
- [Dobór metod dostrajania](#dobór-metod-dostrajania-modeli-do-indywidualnych-potrzeb)
- [Formatowanie i przygotowanie datasetów](#datasety--formatowanie-dobór-tworzenie-własnych)
- [Najważniejsze zalecenia i przeciwwskazania](#najważniejsze-zalecenia-dos-i-przeciwwskazania-donts-oraz-wstępne-porady-praktyczne)

### Apple Silicon i MLX
- [Ewolucja procesorów Apple M1-M4](#ewolucja-procesorów-od-m1-do-m4max)
- [Zunifikowana pamięć i ML](#zunifikowana-pamięć-unified-memory-i-jej-wpływ-na-ml)
- [Wydajność CPU, GPU i Neural Engine](#wydajność-cpu-gpu-i-neural-engine-w-kontekście-mlai)
- [Integracja MLX z Apple Silicon](#integracja-mlx-ml-z-architekturą-apple-silicon)
- [Wydajność MLX w różnych zadaniach](#wydajność-mlx-ml-na-zadaniach-nlp-cv-i-generatywnych)
- [Porównanie z GPU/TPU](#porównanie-applesilicon--mlx-do-gpu-nvidiaamd-oraz-tpu)
- [Analiza kosztów i efektywności](#analiza-kosztów-obliczeniowych-i-efektywności-energetycznej)

### Praktyczne zastosowania MLX
- [Optymalizacja pamięci](#optymalizacja-pamięci-i-wykorzystanie-unified-memory)
- [Techniki mieszanej precyzji](#techniki-mieszanej-precyzji-i-ich-wpływ-na-wydajność)
- [Efektywne metody dostrajania](#efektywne-metody-dostrajania-modeli-fine-tuning-w-mlx-ml)
- [Integracja z ekosystemem Apple](#integracja-mlx-ml-z-ekosystemem-apple-mps-coreml-metal)
- [Instalacja i konfiguracja MLX-LM](#instalacja-mlx-lm-oraz-konfiguracja-środowiska)
- [Podstawowe parametry treningu](#podstawowe-parametry-treningu-w-mlx-lm)

### Scenariusze Fine-Tuningu
- [Dostrajanie językowe (polski, branżowy)](#fine-tuning-językowy--nauka-języka-polskiego-model-3b)
- [Dostrajanie technologiczne (kod)](#fine-tuning-technologiczny--nauka-języka-programowania-model-7b)
- [Computer Vision i Vision-Language](#fine-tuning-cv--model-detekcji-obrazów-model-7b)
- [Modele generatywne](#fine-tuning-generatywny--stable-diffusion-model-3b)
- [Speech-to-Text i Text-to-Speech](#fine-tuning-stttts--model-rozpoznawania-i-generowania-mowy)
- [Sieci konwolucyjne](#fine-tuning-konwolucyjnych-sieci-neuronowych--u-netresnetgan)
- [Inne scenariusze zastosowań](#inne-scenariusze-fine-tuningu)

### Porównania i optymalizacja
- [MLX vs inne narzędzia](#porównanie-mlx-lm-z-innymi-narzędziami-na-apple-silicon)
- [Techniki dostrajania](#techniki-dostrajania-i-optymalizacji)
- [Hiperparametry vs sprzęt](#hiperparametry-vs-rozmiar-modelu-i-sprzęt)

# Wprowadzenie do dostrajania modeli językowych (LM)
Dostrajanie (*fine-tuning*) modeli językowych to proces dalszego trenowania uprzednio wytrenowanych modeli AI (np. dużych modeli językowych) na nowych danych w celu adaptacji do specyficznych zadań lub dziedzin. Poniżej przedstawiono przegląd metod dostrajania parametrów modeli o otwartym dostępie do wag, objaśnienie kluczowych pojęć, wskazówki doboru odpowiedniej metody, aspekty związane z przygotowaniem danych (datasetów) oraz najważniejsze zalecenia i ostrzeżenia praktyczne.

## Metody dostrajania parametrów modeli AI z otwartymi wagami
Istnieje kilka podejść do dostrajania dużych, uprzednio wytrenowanych modeli AI. Poniżej omówiono najpopularniejsze metody wraz z ich zaletami i wadami, zarówno w kontekście czysto tekstowych modeli językowych, jak i modeli multimodalnych:

- **Pełne dostrajanie modelu (full fine-tuning)** – Tradycyjna metoda polegająca na modyfikowaniu **wszystkich** wag modelu podczas trenowania na nowym zbiorze danych. Zapewnia to maksymalną elastyczność i potencjalnie najwyższą dokładność dostosowania do nowych danych, ale ma wysokie koszty obliczeniowe. W przypadku bardzo dużych modeli (miliardy parametrów) pełne dostrajanie wymaga ogromnej mocy obliczeniowej (GPU/TPU) i pamięci. Dla przykładu, pełne dostrojenie modelu GPT-3 (175 mld parametrów) dla każdej nowej instancji jest praktycznie niewykonalne na typowym sprzęcie użytkownika​

[arxiv.org](https://arxiv.org/abs/2106.09685# :~:text=,Compared%20to)

Wadą jest również większe ryzyko *katastrofalnego zapominania* (utraty ogólnych umiejętności modelu na rzecz danych treningowych) oraz możliwość nadmiernego dopasowania (*overfitting*) do niewielkiego zbioru danych. Pełne fine-tuning stosuje się głównie, gdy dysponujemy stosunkowo niewielkim modelem lub bardzo dużym zasobem danych i mocy obliczeniowej, a zależy nam na maksymalnej jakości modelu w wąskiej dziedzinie.

- **LoRA (Low-Rank Adaptation)** – Metoda efektywnego dostrajania parametrów zaproponowana przez Hu et al. (2021)​

[arxiv.org](https://arxiv.org/abs/2106.09685# :~:text=example%20,trainable%20parameters%2C%20a%20higher%20training)

. W podejściu LoRA zamrażamy oryginalne wagi modelu i dodajemy do każdej warstwy niewielkie macierze *adaptacyjne* o niskiej randze, które są trenowane na nowych danych. Dzięki temu liczba trenowanych parametrów jest dramatycznie mniejsza niż w pełnym fine-tuningu (LoRA potrafi zredukować liczbę trenowanych parametrów nawet ~10000-krotnie) oraz obniża się zużycie pamięci GPU nawet kilkukrotnie​

[arxiv.org](https://arxiv.org/abs/2106.09685# :~:text=example%20,trainable%20parameters%2C%20a%20higher%20training)

. Co istotne, jakość modelu dostrojonego LoRA bywa porównywalna z pełnym dostrojeniem, a sama metoda nie powoduje spowolnienia wnioskowania – dodatkowe macierze są wkomponowane w strukturę modelu i nie zwiększają złożoności obliczeń podczas inferencji. Zalety LoRA to więc: **niski koszt pamięci**, **szybsze trenowanie** (mniej parametrów do optymalizacji) oraz **brak wpływu na oryginalne wagi** (można zachować ogólną wiedzę modelu i nakładać różne „adaptery” LoRA dla różnych zadań). Wadą LoRA może być ograniczona pojemność adaptacji – dodając relatywnie mało nowych parametrów, model może nie uchwycić bardzo dalekiego przesunięcia domeny, jeśli nasze nowe zadanie jest skrajnie odmienne od danych, na których model był wstępnie trenowany. W praktyce jednak LoRA sprawdza się znakomicie dla wielu zadań i jest obecnie standardem przy dostrajaniu bardzo dużych modeli językowych na mniejszych zbiorach danych. W kontekście modeli multimodalnych, koncepcja LoRA również bywa stosowana – można np. zamrozić część odpowiedzialną za obraz i dostroić za pomocą LoRA tylko komponent językowy modelu, lub odwrotnie.

- **QLoRA (Quantized LoRA)** – najnowsze usprawnienie metody LoRA, zaproponowane przez Dettmersa i in. (2023)​

[arxiv.org](https://arxiv.org/abs/2305.14314# :~:text=,new%20data%20type%20that%20is)

. QLoRA dodatkowo redukuje zużycie pamięci przez **kwantyzację** modelu bazowego do 4-bitowej precyzji przed dostrajaniem. Model w trakcie trenowania jest przechowywany w formie 4-bit, a gradienty są propagowane przez zamrożone, skwantyzowane wagi do adapterów LoRA. Pozwala to trenować ekstremalnie duże modele (np. 33–65 mld parametrów) na pojedynczej karcie GPU 48 GB, zachowując jakość porównywalną z pełnym dostrajaniem 16-bitowym​

[arxiv.org](https://arxiv.org/abs/2305.14314# :~:text=,new%20data%20type%20that%20is)

. Autorzy QLoRA zaprezentowali, że przy użyciu tej metody udało im się dostroić model 65B (LLaMA 65B) w 24 godziny na pojedynczym GPU, osiągając ~99% jakości ChatGPT na benchmarach typu Vicuna​

[arxiv.org](https://arxiv.org/abs/2305.14314# :~:text=preserving%20full%2016,memory%20footprint%20by%20quantizing%20the)

	Zalety QLoRA to **skrajnie niski wymagany budżet pamięci** (dzięki 4-bitowej kwantyzacji) połączony z korzyściami LoRA. W praktyce pewnym kosztem jest nieco bardziej złożona implementacja (trzeba obsłużyć kwantyzację i dekwantyzację podczas obliczeń) oraz potencjalnie minimalnie wolniejsze mnożenia na 4-bitach (choć specjalne kernele CUDA/Metal to niwelują). Dla użytkowników z ograniczonym sprzętem (np. laptop z Apple Silicon) QLoRA bywa jedyną opcją dostrojenia modeli powyżej 7–13B parametrów. W modelach multimodalnych QLoRA również może znaleźć zastosowanie analogicznie, jeśli model bazowy da się spójnie skwantyzować.

- **Destylacja modeli (knowledge distillation)** – Metoda istotnie różniąca się od powyższych, ponieważ zamiast dalszego trenowania *tego samego* modelu, uczymy zupełnie nowy, mniejszy model (*uczeń*) na podstawie wyjść modelu dużego (*nauczyciela*). Destylacja została spopularyzowana przez Hinton et al. (2015)​

[arxiv.org](https://arxiv.org/abs/1503.02531# :~:text=are%20large%20neural%20nets,a%20mixture%20of%20experts%2C%20these)

jako technika kompresji wiedzy modelu: zamiast trenować studenta tylko na oryginalnych danych z twardymi etykietami, wykorzystuje się miękkie prawdopodobieństwa predykcji nauczyciela jako dodatkowe cele treningowe. Uczeń „uczy się” imitować odpowiedzi nauczyciela, zachowując jego zdolności w skompresowanej formie. Przykładem sukcesu destylacji jest model **DistilBERT** (Sanh et al. 2019), który zredukował rozmiar BERT o 40% przy zachowaniu 97% jego zdolności językowych i uzyskaniu 60% większej szybkości działania​

[arxiv.org](https://arxiv.org/abs/1910.01108# :~:text=for%20building%20task,device%20study)

	Zaletą destylacji jest możliwość **uzyskania mniejszego i szybszego modelu** do wdrożenia na urządzenia edge (np. smartfony) lub serwery o ograniczonych zasobach, bez ogromnej utraty jakości. Ponadto destylacja pozwala czasem poprawić ogólne uogólnienie – nauczyciel przekazuje uczniowi bardziej „wiedzę” niż pojedyncze przykłady (np. uśredniając sygnały z wielu modeli, jeśli używamy ensembles). Wadą jest konieczność przeprowadzenia pełnego procesu trenowania nowego modelu – więc oszczędzamy zasoby przy inferencji, ale niekoniecznie podczas trenowania (trening ucznia może być równie kosztowny co trenowanie modelu od zera, zwłaszcza gdy uczeń nadal jest dość duży). Dodatkowo, jakość destylacji zależy od architektury ucznia – zbyt mały model ucznia może nie być w stanie odzwierciedlić złożonych umiejętności nauczyciela. Destylację stosuje się często, gdy chcemy **udostępnić model podobny do dużego oryginału, ale lżejszy**, np. destylowanie 13B modelu do 6B, albo by **połączyć wiedzę wielu modeli w jednym**. W kontekście modeli multimodalnych destylacja bywa rzadziej stosowana, ale są próby np. destylowania modelu tekst+obraz do czysto tekstowego modelu (lub odwrotnie) w celu przekazania części informacji.

- **Inne techniki** – Warto wspomnieć, że istnieją również inne metody dostrajania i adaptacji modeli, takie jak np. *adaptery* (niewielkie warstwy dodawane do modelu, trenowane podobnie jak LoRA, lecz często wprowadzające niewielkie opóźnienie w inferencji) czy *prompt tuning* / *prefix tuning* (gdzie optymalizuje się tylko dodatkowe tokeny lub wektory wejściowe, pozostawiając model nietknięty). W praktyce jednak podejścia te są mniej popularne w 2024/2025 roku niż LoRA/QLoRA ze względu na nieco gorszą efektywność lub mniejszą uniwersalność.

**Porównanie technik w kontekście modeli językowych vs. multimodalnych:** Powyższe metody zostały pierwotnie opracowane dla dużych modeli językowych (LLM), ale wiele z nich można zastosować również do modeli multimodalnych z pewnymi modyfikacjami. Pełne dostrajanie jest uniwersalne (można nim dostrajać zarówno modele tekstowe, jak i np. modele przetwarzające obrazy czy dźwięk, o ile mamy wystarczający sprzęt). Metody typu LoRA/QLoRA są szczególnie cenne dla LLM, lecz ich odpowiedniki stosuje się np. w modelach generowania obrazów (adaptacja Stable Diffusion do konkretnego stylu za pomocą DreamBooth/LoRA) czy w modelach kombinowanych obraz-tekst (dodając niskorangowe adaptery do modułów przetwarzania obrazu lub tekstu). W modelach multimodalnych często stosuje się podejście hybrydowe: np. zamraża się część wizualną (np. encoder CNN/Vision Transformer) i dostraja tylko część językową, lub odwrotnie, w zależności od zadania i dostępnych danych. Destylacja modeli multimodalnych może służyć np. do stworzenia lżejszej wersji modelu rozpoznającego obraz i opisującego go tekstem – poprzez trenowanie mniejszego modelu na wyjściach dużego. Generalnie jednak dostrajanie modeli multimodalnych bywa bardziej złożone, bo wymaga odpowiednio zestrojonych danych wielu modalności i często większej ostrożności, aby nie pogorszyć zdolności w jednej modalności podczas dostrajania pod inną.

## Opis i omówienie kluczowych pojęć

Aby w pełni zrozumieć proces dostrajania modeli, warto przypomnieć kilka podstawowych pojęć z zakresu uczenia maszynowego oraz wprowadzić terminologię związaną z różnymi technikami trenowania modeli:

**Trening modelu ML (backpropagation, loss function, gradient descent)** – Trenowanie sieci neuronowej polega na iteracyjnym modyfikowaniu jej wag w celu minimalizacji pewnej funkcji kosztu (funkcji straty). Standardowym algorytmem jest **wsteczna propagacja błędu** (*backpropagation*), zaproponowana już w latach 80., która umożliwia obliczenie gradientu funkcji straty względem wszystkich wag sieci poprzez propagację błędu od warstwy wyjściowej do wejściowej. **Funkcja straty** (*loss function*) stanowi matematyczną miarę różnicy między przewidywaniami modelu a oczekiwanym wyjściem (np. *cross-entropy* dla klasyfikacji lub *mean squared error* dla regresji). Podczas trenowania, dla każdej porcji danych (batcha) obliczany jest loss, a następnie metoda optymalizacji, taka jak **spadek gradientu** (*gradient descent*) lub jego odmiany (np. Adam, RMSprop), wykorzystuje gradienty do aktualizacji wag w kierunku przeciwnym do gradientu (aby obniżyć stratę). W praktyce stosuje się *stochastyczny spadek gradientu* (SGD), gdzie optymalizacja odbywa się na podstawie losowych podzbiorów danych (batchy) w każdej iteracji, co uśrednia się w dłuższym horyzoncie na całym zbiorze. Proces ten trwa przez wiele epok, aż model osiągnie zadowalającą minimalną wartość straty (lub przestanie się poprawiać na zbiorze walidacyjnym). W kontekście dostrajania (*fine-tuning*), wykorzystujemy wstępnie wytrenowany model jako punkt startowy, co zazwyczaj oznacza, że zaczynamy już z wagami leżącymi w pobliżu dobrego minimum dla zadania ogólnego, a trening fine-tuning ma za zadanie *lekko dostosować* te wagi do nowego zadania, zwykle przy użyciu mniejszej liczby kroków i mniejszego learning rate niż w treningu od zera.

**Destylacja modeli (metoda nauczyciel–uczeń)** – Destylacja wiedzy to technika, w której **model nauczyciel** (zwykle duży i dokładny) przekazuje informacje **modelowi uczniowi** (mniejszemu, lżejszemu) poprzez swoje przewidywania. Zamiast klasycznego trenowania ucznia tylko na podstawie danych z etykietami, w destylacji ucznia trenuje się tak, by jego rozkłady wyjściowe były jak najbardziej zbliżone do rozkładów wyjściowych nauczyciela. Implementacyjnie często wykorzystuje się do tego *miękkie* prawdopodobieństwa z warstwy softmax nauczyciela jako cel dla ucznia, minimalizując np. dywergencję Kullbacka-Leiblera lub skrośną entropię między rozkładami wyjściowymi obu modeli. Często stosuje się **temperaturę** w softmaxie nauczyciela, aby uwypuklić więcej informacji – np. nie tylko która klasa jest poprawna, ale i które klasy suboptymalne model uznaje za prawdopodobne. Uczeń tym samym uczy się nie tylko docelowych etykiet, ale i „myślenia” nauczyciela. Jak wspomniano wyżej, destylacja pozwoliła stworzyć DistilBERT, który zachował prawie pełnię możliwości BERTa przy znacznie mniejszym rozmiarze​

[arxiv.org](https://arxiv.org/abs/1910.01108# :~:text=for%20building%20task,device%20study)

## W praktyce proces destylacji wygląda następująco:

```python
teacher.eval() # model nauczyciel (zamrożony) student.train() # model uczeń (będziemy trenować)

for batch in dataloader: # iterujemy po minibatchach danych inputs, labels = batch
``` 

```python
with torch.no_grad():                    
teacher_logits = teacher(inputs)               # predykcje nauczyciela student_logits = student(inputs)                   # predykcje ucznia  # Obliczenie straty destylacji między rozkładami nauczyciela i ucznia: loss_distill = KLDivLoss(softmax(student_logits, T),                          softmax(teacher_logits, T))   # z odpowiednią temperaturą T  # Opcjonalnie: dodajemy klasyczną stratę na etykietach, aby uczeń  # równocześnie dobrze wykonywał zadanie (np. klasyfikację): loss_task = CrossEntropyLoss(student_logits, labels)    loss = loss_distill + alpha * loss_task    # łączna strata (alpha - waga komponentu zadaniowego)  loss.backward()                # oblicz gradienty (propagacja wsteczna) optimizer.step()               # aktualizacja wag ucznia optimizer.zero_grad()          # wyzerowanie gradientów przed następną iteracją`

```

Pseudokod powyżej ilustruje uproszczony schemat destylacji: na każdym minibatchu obliczamy predykcje nauczyciela (bezgradiencie) i ucznia, następnie wyliczamy stratę destylacji (np. KL-divergence) pomiędzy rozkładami prawdopodobieństw obu modeli. Często dodaje się też zwykłą stratę względem prawidłowych etykiet (tzw. *hard loss*), aby uczeń nie odbiegł od faktycznego zadania. Optymalizator następnie aktualizuje tylko wagi ucznia. Nauczyciel pozostaje niezmieniany. Po wystarczającej liczbie iteracji otrzymujemy dostatecznie dobrego ucznia – znacznie mniejszego niż nauczyciel – który imituje jego odpowiedzi.

**Fine-tuning modeli** – *Fine-tuning* to właśnie proces, na którym się skupiamy: dalsze trenowanie modelu uprzednio wytrenowanego (*pre-trained*) na dużym zbiorze danych ogólnych, przy użyciu danych bardziej specyficznych dla zadania lub domeny. Celem fine-tuningu jest zwiększenie **dopasowania modelu do niszowego zastosowania** – np. mamy model językowy wytrenowany na ogromnym korpusie internetu, a chcemy, by sprawdzał się w dialogach medycznych; lub model rozpoznający obrazy ogólnie, który chcemy dostosować do wykrywania wad produkcyjnych na taśmie. Fine-tuning zwykle przebiega szybciej niż trenowanie od zera, bo model już posiada ogólną wiedzę, którą musimy tylko wyspecjalizować. W praktyce podczas dostrajania często stosuje się mniejsze tempo uczenia, mniejsze batch size oraz krótszy trening (np. kilka epok) w porównaniu do pretrenowania. Ważne jest też, by zapobiegać wspomnianemu **katastrofalnemu zapominaniu** – model nie powinien utracić swojej wcześniejszej wiedzy (chyba że nam na tym nie zależy). W tym celu czasami zamraża się część warstw i trenuje tylko wyższe poziomy sieci lub stosuje techniki regularyzacji nakładające karę za odejście wag od wartości pierwotnych. Fine-tuning może znacząco poprawić wyniki w wąskiej dziedzinie – np. dostrojenie GPT-2 do pisania poezji sprawi, że model będzie lepiej naśladował styl poetycki kosztem zdolności do generowania innych form tekstu. Fine-tuning bywa też używany, by model lepiej przestrzegał instrukcji lub określonych *reguł formatowania* – słynne *instruction tuning* (jak w modelach typu GPT-3.5/GPT-4) to nic innego jak fine-tuning na rozmowach człowiek-asystent, by model nauczył się odpowiedniego stylu dialogu.

**Modele hybrydowe (Dense + Reasoning, MoE + Reasoning, multimodalność)** – Wraz ze wzrostem złożoności zadań i potrzeb, powstały architektury hybrydowe łączące różne podejścia:

- _Modele gęste + moduły rozumowania (Dense + Reasoning)_: Klasyczne „gęste” modele językowe (tzn. sieć neuronowa, w której wszystkie parametry są wykorzystywane do każdego zapytania) można wzbogacić o mechanizmy rozumowania. Przykładem może być dodanie **jawnego wnioskowania krok po kroku** – np. poprzez technikę *Chain-of-Thought*, gdzie model generuje ciąg pośrednich kroków przed odpowiedzią. Sam model pozostaje gęsty (np. GPT-3), ale jest uczony, by wewnętrznie przeprowadzać wieloetapowe rozumowanie. Można to osiągnąć przez fine-tuning na danych z pokazanymi „rozwiązaniami krokowymi” problemów (np. zadania matematyczne rozpisane na kroki). Model gęsty można też połączyć z zewnętrznym modułem rozumującym – np. wykorzystać narzędzia (kalkulator, wyszukiwarkę) w trakcie generacji. Wówczas mówimy o architekturze rozszerzonej, ale wciąż rdzeń modelu (generatywny) jest gęsty.

- _Mixture of Experts + Reasoning (MoE + Reasoning)_: **Mixture-of-Experts (MoE)** to architektura, w której mamy wiele niezależnych podmodeli („ekspertów”) i mechanizm routingu, który wybiera, który ekspert obsłuży dany przykład. Model MoE jest więc **rzadko aktywowany** – dla każdego zapytania aktywna jest tylko część parametrów (np. 2 spośród 16 ekspertów), co pozwala efektywnie skalować liczbę parametrów bez proporcjonalnego wzrostu kosztu obliczeń​

    [arxiv.org](https://arxiv.org/abs/2101.03961# :~:text=,training%20techniques%20help%20wrangle%20the)

    . Switch Transformer (Fedus et al. 2021) pokazał, że można trenować modele z bilionem parametrów właśnie dzięki Mixture-of-Experts​

    [arxiv.org](https://arxiv.org/abs/2101.03961# :~:text=instabilities%20and%20we%20show%20large,XXL%20model)

    . W kontekście *reasoning*, idee MoE są używane np. do specjalizowania ekspertów w różnych rodzajach zadań rozumowania. Najnowsze prace (np. Mixture-of-Reasoning-Experts, MORE) sugerują, że można mieć osobnych „ekspertów” od rozumowania arytmetycznego, od rozumowania faktograficznego, od zdroworozsądkowego etc., a następnie mieć moduł decydujący, który ekspert najlepiej poradzi sobie z danym pytaniem​

    [openreview.net](https://openreview.net/forum?id=UMywlqrW3n# :~:text=answering%20,select%20the%20best%20answer%20for)

    ​

    [openreview.net](https://openreview.net/forum?id=UMywlqrW3n# :~:text=Mixture,results%20compared%20to%20baselines%20without)

    . Takie podejście poprawia uogólnienie – pojedynczy model rzadko radzi sobie świetnie ze wszystkimi typami zadań naraz, podczas gdy ensemble ekspertów może pokryć szerszy zakres umiejętności. Wadą jest złożoność: modele MoE są trudniejsze w trenowaniu (problem zbalansowania użycia ekspertów, komunikacji między nimi itd.); choć prace takie jak Switch Transformer upraszczają routing i pokazują sposoby stabilizacji treningu​

    [arxiv.org](https://arxiv.org/abs/2101.03961# :~:text=,training%20speed%20with%20the)

    . Dostrajanie modelu MoE może oznaczać dostrajanie wszystkich ekspertów lub tylko wybranych – np. możemy dodać nowego eksperta wyspecjalizowanego w nowym rodzaju zadania, zamiast modyfikować istniejących.

- _Multimodalność_: Model multimodalny to taki, który przetwarza więcej niż jedną modalność danych, np. tekst + obraz, tekst + audio, obraz + opis tekstowy, itp. Przykładem jest CLIP (łączenie obrazów z tekstowymi opisami) czy bardziej zaawansowane: Flamingo, GPT-4 (wizualny), które potrafią przyjmować obraz jako kontekst do generowania tekstu. Fine-tuning modelu multimodalnego wymaga datasetów zawierających pary różnych modalności (np. obraz z podpisem) i często specjalnych architektur: np. sieć wizualna (CNN lub ViT) połączona z siecią językową (transformer). **Dense + multimodalność**: niektóre modele są „gęste” multimodalnie – np. pojedynczy Transformer przyjmuje wymieszane tokeny tekstowe i wizualne (po odpowiedniej ewaluacji obrazu). Inne są modularyzowane. W kontekście dostrajania, jeśli np. mamy model, który umie czytać obraz i odpowiadać na pytania, a chcemy go dostroić do bardzo specyficznego rodzaju obrazów (np. zdjęcia rentgenowskie) i pytań (diagnostyka medyczna), to często zamrażamy część, która rozumie język ogólnie, a dostrajamy tylko część wizualną na nowej dziedzinie obrazów – lub odwrotnie, jeśli język jest specjalistyczny medyczny, a obrazy nadal ogólne. Multimodalne dostrajanie bywa wrażliwe na ilość danych – dane multimodalne są trudniejsze do zebrania, więc często pracuje się na niewielkich zbiorach, co wymaga ostrożnego fine-tuningu (np. użycia metod typu LoRA albo treningu tylko ostatnich warstw, by nie rozstroić całego modelu).

Podsumowując, modele hybrydowe łączą tradycyjne podejścia (gęste modele, eksperckie modele, modele multimodalne) z celowym ukierunkowaniem na rozumowanie lub łączenie wiedzy z różnych źródeł. Dostrajanie takich modeli wymaga zrozumienia zarówno ich architektury, jak i dostępnych danych do treningu – np. czy mamy oznaczone kroki rozumowania, czy mamy pary obraz-tekst, czy potrzebujemy nauczyć gating między ekspertami. W kolejnych częściach przewodnika skupimy się jednak głównie na praktycznym dostrajaniu LLM (głównie tekstowych), co stanowi podstawę, od której można wyjść do bardziej złożonych scenariuszy.

## Dobór metod dostrajania modeli do indywidualnych potrzeb
Nie każda metoda dostrajania jest optymalna w każdej sytuacji. Wybór podejścia zależy od wielu czynników: rodzaju modelu, jakim dysponujemy (jego wielkości i architektury), dostępności danych do fine-tuningu, zasobów sprzętowych (GPU / Apple Silicon) oraz celu, jaki chcemy osiągnąć (czy model ma działać na urządzeniu mobilnym, czy wystarczy on-premise na mocnej maszynie). Poniżej przedstawiono kilka wskazówek, które pomogą dobrać właściwą metodę dostrajania do potrzeb:

**Kiedy użyć LoRA zamiast pełnego fine-tuningu?** Gdy mamy **duży model** (setki milionów lub miliardy parametrów) i chcemy go dostroić na stosunkowo niewielkim zbiorze danych lub przy ograniczonej pamięci GPU/RAM. LoRA jest idealna, jeśli zależy nam na szybkim eksperymentowaniu – dzięki mniejszej liczbie trenowanych wag możemy przeprowadzać wiele eksperymentów taniej. Użycie LoRA zamiast pełnego fine-tune jest wskazane również wtedy, gdy chcemy **zachować oryginalny model nietknięty** – np. mamy model podstawowy i chcemy przygotować jego różne warianty dla różnych zadań, zachowując możliwość przełączania się między nimi. LoRA daje możliwość dystrybucji tylko małych plików z adapterami (rzędu megabajtów zamiast gigabajtów oryginalnych wag). Wadą pełnego fine-tuningu, oprócz wymagań sprzętowych, jest to, że model staje się *jednowariantowy* – trudno scalić potem dwie wersje dostrojone do różnych zadań, podczas gdy w podejściu LoRA możemy trzymać osobno bazowe wagi i osobno kilka zestawów adapterów do różnych zastosowań. Reasumując: **użyj LoRA**, jeśli Twój model jest zbyt duży na pełne trenowanie na posiadanym sprzęcie lub gdy chcesz minimalnie zmodyfikować model (przy zachowaniu jego pierwotnej wszechstronności), albo potrzebujesz wielu wersji modelu dla różnych zastosowań. Przykładowo, zamiast fine-tunować 7B parametrów modelu LLaMA na dane dialogowe, można wytrenować LoRA, który zajmuje <100 MB i osiąga podobną jakość dialogową​

[arxiv.org](https://arxiv.org/abs/2106.09685# :~:text=example%20,trainable%20parameters%2C%20a%20higher%20training)

.

**Kiedy preferować pełny fine-tuning?** Pełne dostrajanie wciąż bywa uzasadnione w niektórych scenariuszach. Jeśli model bazowy nie jest ogromny (powiedzmy do 300–500 mln parametrów) i mamy dość pamięci, można pokusić się o pełny fine-tune – uzyskamy wtedy maksymalne dostosowanie wag. Również jeśli posiadamy **bardzo duży zbiór nowych danych**, porównywalny z pretrainingiem, i chcemy gruntownie przetrenować model (np. adaptacja modelu językowego do innego języka niż większość danych pretreningowych), pełny fine-tuning może wydobyć więcej z modelu niż dodatkowe adaptery. Trzeba jednak uważać – w pełnym fine-tuningu łatwo *przeuczyć* model, jeśli nasz nowy dataset nie jest wystarczająco zróżnicowany. Co więcej, pełne dostrajanie unieważnia niektóre zalety modelu bazowego – np. jeżeli był wszechstronny, to po dostrojeniu do jednej domeny może utracić część możliwości spoza domeny (chyba że dataset jest naprawdę duży i różnorodny). Zatem pełny fine-tuning zostawiamy sytuacjom, gdy: (a) mamy sporo zasobów obliczeniowych, (b) zależy nam wyłącznie na wydajności w nowym zadaniu i ewentualna utrata innych umiejętności modelu nas nie martwi, (c) model wielkościowo pozwala na to (np. chcemy dostroić niewielki model 60M parametrów do naszych danych – to wykonalne na GPU 16 GB w rozsądnym czasie).

**Jakie modele nadają się do których technik?**

- **LoRA/QLoRA** najbardziej zyskujemy przy większych modelach (powyżej 1B parametrów), bo tam oszczędność trenowanych wag i pamięci jest największa. Dla bardzo małych modeli (np. 50M) stosowanie LoRA nie jest konieczne – pełny fine-tune nie jest wtedy problemem, choć można i tak użyć LoRA, by np. porównywać efekty lub jeśli chcemy zachować oryginalne wagi. Modele typu GPT-2, GPT-3, LLaMA, T5, BERT i ich warianty – wszystkie one były testowane z powodzeniem z LoRA. Z kolei QLoRA jest szczególnie przydatna dla modeli 7B+ które chcemy trenować na sprzęcie z ograniczoną pamięcią (jak pojedyncza karta 24 GB, czy nawet 16 GB). Przykładowo, autorzy QLoRA pokazali, że model 65B parametrów można dostroić na jednej GPU 48 GB​

    [arxiv.org](https://arxiv.org/abs/2305.14314# :~:text=,new%20data%20type%20that%20is)

    – to samo byłoby absolutnie niemożliwe w 16-bitowej arytmetyce. Zatem do największych modeli wybierz QLoRA; do średnich LoRA; do małych ewentualnie pełny fine-tune.

- **Destylacja** ma sens przede wszystkim, gdy celem końcowym jest **zmniejszenie modelu**. Jeśli dysponujemy bardzo silnym modelem (np. 30B parametrów) działającym na serwerze i chcielibyśmy mieć jego okrojoną wersję na smartfonie, to destylacja do modelu 1B lub 500M może pozwolić uzyskać akceptowalne wyniki w dużo lżejszej formie. Modele, które najczęściej poddaje się destylacji, to te stosunkowo duże, z których chcemy uzyskać „młodsze rodzeństwo”. Przykład: DistilBERT (destylowany z BERT-base), TinyBERT, DistilGPT-2, itp. Jeśli nasz model bazowy jest już niewielki, destylacja nie ma sensu (nie uzyskamy dużo mniejszego). Warto też pamiętać, że destylacja wymaga przygotowania predykcji nauczyciela na odpowiednim zbiorze „transferowym” – co samo w sobie bywa czasochłonne (trzeba przepuścić dużo danych przez duży model). Gdy jednak planujemy **wiele wdrożeń modelu** i chcemy zmniejszyć koszty inferencji, destylacja jednorazowo jest warta zachodu. W kontekście technik: destylację można łączyć z fine-tuningiem – np. najpierw dostroić duży model do zadania (jeśli bazowy nie radził sobie dostatecznie), a następnie zdestylować tę wiedzę do mniejszego modelu ucznia.

- **Mixture-of-Experts / architektury specjalne**: Jeśli Twój model bazowy jest architekturą MoE, to dostrajanie może wymagać decyzji, czy dostrajać wszystkich ekspertów, czy np. tylko bramkę routingu. W literaturze spotyka się np. podejście *sparsity fine-tuning*, gdzie tylko niewielki podzbiór ekspertów otrzymuje gradient (np. te najczęściej wykorzystywane dla danych trenowanych). MoE to jednak niszowa sytuacja dla większości użytkowników (raczej nie posiadasz lokalnie modelu MoE, gdyż takie modele zajmują ogromne rozmiary i są trudne w obsłudze). Jeśli jednak korzystasz z usług chmurowych z modelami MoE, być może mechanizmy dostrajania są ukryte w ofercie dostawcy. Ogólnie, modele MoE nadają się do zadań, gdzie różnorodność danych jest duża – można wtedy mieć ekspertów od różnych podzadań. Dla porównania, modele gęste są bardziej uniwersalne w użyciu lokalnym.

**Porównanie metod w zależności od sprzętu użytkownika:** Sprzęt, którym dysponujemy, często narzuca wybór techniki dostrajania. Przykładowo, posiadacze MacBooków z Apple Silicon (M1, M2, M3) mają do dyspozycji zunifikowaną pamięć RAM zamiast tradycyjnej karty GPU z VRAM, co ma plusy i minusy. Omówmy kilka typowych konfiguracji:

- _Laptop/desktop z Apple Silicon, 16 GB RAM (np. MacBook Air/Pro M1/M2)_ – 16 GB to stosunkowo niewiele, jeśli chodzi o trenowanie dużych modeli. Na takim sprzęcie realne jest dostrajanie modeli do ~7 miliardów parametrów w precyzji 16-bit, jednak z małym batch size (nawet batch = 1) i ewentualnie przy skróceniu sekwencji. Świetnym rozwiązaniem jest użycie **4-bitowej kwantyzacji i LoRA (QLoRA)**, co pozwoli zmieścić większy model. Na przykład, 7B model w 4-bit zajmie ok. 4 GB pamięci, a dodatkowe struktury LoRA to ułamek tego – można więc próbować trenować nawet model 13B na 16 GB przy batch=1, korzystając z QLoRA. Jeżeli model 7B w pełnej precyzji mieści się w pamięci, można użyć zwykłej LoRA, co uprości pipeline. Pełny fine-tuning 7B na 16 GB może być problematyczny, bo podczas backpropagacji pamięć zajmują też gradienty i optymalizator – ale z LoRA (gdzie gradienty liczymy tylko dla małych macierzy) jest to wykonalne. **Rekomendacja**: dla 16 GB wybierz LoRA/QLoRA koniecznie, ogranicz batch size do 1–2, rozważ skrócenie sekwencji treningowych lub użycie *gradient checkpointing* (który zamienia pamięć na dodatkowe obliczenia). Unikaj pełnego fine-tune dużego modelu – chyba że jest to model rzędu 1–2B, wtedy być może się uda. Jeśli model jest mniejszy (np. DistilBERT 66M), oczywiście możesz go trenować normalnie.

- _Mac z 32 GB RAM (np. wyższa konfiguracja MacBook Pro, Mac Studio)_ – 32 GB umożliwia wygodniejsze dostrajanie. Przykładowo, eksperymenty Apple pokazują, że model 7B z LoRA (4 warstwy LoRA) osiąga ~250 tokenów/s podczas treningu na M1 Max 32 GB​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=mlx_lm.lora%20%5C%20,data%20wikisql)

    . To całkiem przyzwoita szybkość. Z 32 GB można pokusić się o trenowanie 13B w 4-bit LoRA albo 7B w pełnej precyzji z LoRA i nieco większym batch size. Wciąż jednak 32 GB to za mało na pełny fine-tune np. 13B (wymagałby >2x tyle). **Rekomendacja**: LoRA/QLoRA pozostaje najlepszym wyborem, ewentualnie częściowe dostrajanie pełne małych modeli. Warto wykorzystać to, że Unified Memory w Apple Silicon pozwala wykorzystać maksymalnie dostępne zasoby – przy 32 GB można ustawić nieco większy `--batch-size` lub więcej warstw LoRA (np. 8 zamiast 4), co poprawi jakość kosztem użycia pamięci​

    [apeatling.com](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/# :~:text=You%E2%80%99ll%20want%20to%20experiment%20with,some%20good%20tips%20for%20this)

    ​

    [apeatling.com](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/# :~:text=With%20my%20M1%20Max%20and,to%20120%20tokens%2Fsec%20on%20average)

    .

- _Mac Studio / Mac Pro z czipem M2 Ultra, 64–128 GB RAM_ – Ta klasa sprzętu otwiera nowe możliwości. Mając 64 GB, można pokusić się o dostrajanie modeli 30B parametrów przy użyciu 4-bit QLoRA (30B 4-bit to ok. 15 GB pamięci, reszta zostaje na gradienty i overhead). 128 GB RAM potencjalnie umożliwi nawet eksperymenty z modelami 65B w 4-bit quant, choć prędkość może być ograniczona. Przy takiej pamięci można też spróbować pełnego fine-tune średnich modeli (13B, a może nawet 30B z gradient checkpointing). Należy pamiętać, że szybkość treningu na Apple Silicon wciąż jest niższa niż na najnowszych GPU NVIDIA, ale dla wielu zastosowań jest wystarczająca. **Rekomendacja**: w dalszym ciągu QLoRA będzie najbezpieczniejszą opcją dla największych modeli. Dla średnich (6B–13B) można rozważyć pełne dostrojenie jeśli konieczne, ale często LoRA da porównywalny efekt szybciej. Sprzęt 64+ GB umożliwia także trenowanie z większymi batchami, co może przyspieszyć konwergencję – warto wtedy zwiększać stopniowo batch i obserwować zużycie pamięci.

- _GPU klasy desktop (dla porównania)_ – Jeśli ktoś dysponuje np. kartą NVIDIA RTX 3090 (24 GB) lub RTX 4090 (24 GB), sytuacja jest podobna do 16–32 GB Apple – 24 GB VRAM pozwoli na 7B full fine-tune (na styk) lub wygodny LoRA, a z 4-bit pewnie da się 13B. Przewagą GPU NVIDIA jest obecnie ekosystem – wiele narzędzi jest pisanych najpierw pod CUDA. Z kolei Apple nadrabia własnym frameworkiem **MLX**, który jest zoptymalizowany pod Apple Silicon, dając imponujące wyniki jak na dostępny hardware.

Podsumowując, dobór techniki: **Jeżeli masz mocno ograniczony sprzęt – korzystaj z metod parametrycznie efektywnych (LoRA, QLoRA) i małych batchy.** Jeżeli masz dostęp do dużej pamięci i mocy – możesz pozwolić sobie na pełne dostrajanie lub większy komfort (większy batch, dłuższe sekwencje) przy LoRA. Zawsze warto monitorować zużycie pamięci i w razie potrzeby skorzystać z opcji takich jak gradient checkpointing czy redukcja liczby warstw do dostrojenia​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=1,Setup%20section%20for%20more%20details)

​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=3,with%20a%20lot%20of%20data)

. Poniżej przykład uruchomienia dostrajania LoRA modelu 7B za pomocą Apple MLX na Macu (komentarze wyjaśniają parametry):

```

# Uruchomienie dostrajania LoRA z wykorzystaniem narzędzia mlx_lm.lora:

python3 -m mlx_lm.lora
--train \ # uruchom tryb treningu (fine-tuning) --model mistralai/Mistral-7B-Instruct-v0.2 \ # model bazowy z HF Hub (7B instruktażowy) --data ~/datasets/moj_zbior_danych \ # folder z train.jsonl i valid.jsonl --batch-size 2 \ # rozmiar batcha (dostosuj do pamięci; tu 2) --lora-layers 8 \ # liczba warstw z adapterami LoRA (domyślnie 16 dla 7B) --iters 1000 # liczba iteracji treningowych (tu: 1000) ```

Powyższa komenda (wykorzystująca framework Apple MLX) rozpocznie dostrajanie modelu Mistral-7B na danych użytkownika. Każdy parametr:

- `--model` wskazuje nazwę modelu (może to być ścieżka lokalna lub model z HuggingFace Hub). W powyższym przykładzie używamy otwartego modelu Mistral 7B z instruktażem.
- `--data` to ścieżka do katalogu z danymi treningowymi; plik `train.jsonl` oraz `valid.jsonl` powinny tam być obecne.
- `--batch-size` określa ile przykładów na raz przetwarzamy. Większy batch zwiększa zużycie pamięci; na M1/M2 16GB często trzeba ustawić 1, na 32GB można 2–4.
- `--lora-layers` definiuje, w ilu warstwach modelu umieszczone zostaną adaptery LoRA. Domyślnie MLX stosuje LoRA do 16 warstw. Zmniejszenie tej liczby do 8 lub 4 zmniejsza zużycie pamięci (mniej parametrów do uczenia, mniej gradientów), ale może nieco pogorszyć wynik modelu jeśli nasze dane wymagają dostrojenia głębszych warstw. Apple zaleca dla 32 GB RAM użycie `--batch-size 1` i `--num-layers 4`, co właśnie skutkuje ok. 250 tokenami/s trenowania​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=For%20example%2C%20for%20a%20machine,following%20should%20run%20reasonably%20fast)

    .
- `--iters` to liczba iteracji (update’ów wag) do wykonania. W zależności od rozmiaru danych i batcha, 1000 iteracji może odpowiadać np. kilku epokom lub ułamkowi epoki – należy dobrać tak, by model zdążył się nauczyć (często ustala się liczbę epok zamiast iteracji, ale MLX pozwala też sterować iteracjami).

Wynikiem treningu LoRA będzie plik `adapters.npz` (w folderze roboczym), zawierający wyuczone „nakładki” LoRA. Ten plik można załadować wraz z oryginalnym modelem bazowym, aby uzyskać model dostrojony – mający zdolności oryginalnego modelu + dostosowanie do naszego zadania.

## Datasety – formatowanie, dobór, tworzenie własnych

Jakość i odpowiednie przygotowanie zbioru danych treningowych (*datasetu*) do fine-tuningu jest często czynnikiem decydującym o sukcesie dostrajania. W tej sekcji omówimy, jak formatować dane wejściowe dla narzędzi takich jak MLX-LM i inne, jak dobrać dane do zadania (zwłaszcza dla zadań wymagających *reasoning*), a także pułapki związane z ilością danych.

**Formatowanie datasetów dla narzędzi Apple (MLX-LM) i innych**: Apple udostępnia framework **MLX** i narzędzie `mlx_lm` do trenowania i dostrajania modeli na Apple Silicon. Wykorzystuje on zunifikowaną pamięć i jest zoptymalizowany pod Metal, co umożliwia sprawne trenowanie na Macach. MLX-LM przyjmuje dane w formacie plików `*.jsonl` (JSON Lines), gdzie każdy wiersz to osobny przykład w formacie JSON​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Currently%2C%20%60,are%20examples%20of%20these%20formats)

. Obsługiwane są różne typy danych:

- **Format `chat`** – dla danych konwersacyjnych (wielozwrotkowych). Każdy przykład to obiekt JSON z kluczem `"messages"`, którego wartością jest lista wiadomości. Każda wiadomość ma pole `"role"` (np. user, assistant, system) i `"content"` z treścią.
- **Format `completions`** – dla par instrukcja/wezwanie – odpowiedź. Użyteczny do zadań typu pytanie-odpowiedź, uzupełnianie tekstu itp. Każdy przykład ma np. `"prompt": "<tekst polecenia>"` oraz `"completion": "<tekst odpowiedzi>"`.
- **Format `text`** – dla surowych ciągów tekstowych, np. kontynuacja tekstu. Każdy przykład ma klucz `"text"` z wartością będącą całym tekstem (model dostraja się wtedy do przewidywania kolejnych tokenów na podstawie tej sekwencji).
- **Format `tools`** – specjalny format gdy uczymy model korzystania z narzędzi (funkcji). Oprócz `"messages"`zawiera klucz `"tools"` opisujący dostępne funkcje. Ten format jest przydatny do dostrajania modeli do tzw. *MRKL* (model + wywołania narzędzi), ale jest rzadziej używany w klasycznym fine-tuningu LLM.

MLX automatycznie rozpoznaje format na podstawie kluczy obecnych w JSON. Użytkownik musi jedynie zapewnić, że w folderze z danymi są pliki `train.jsonl` (dane treningowe) i `valid.jsonl` (walidacja). Ewentualnie można też podać `test.jsonl` do oceny końcowej​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=For%20fine,in%20the%20data%20directory)

. Dla innych narzędzi (np. Hugging Face Trainer) format może być inny – często jednak sprowadza się do posiadania dwóch kolumn: *wejście* i *wyjście*. Można wtedy użyć `datasets` z HuggingFace do wczytania pliku JSON i zmapowania pól. W MLX można również korzystać bezpośrednio z datasetów HuggingFace – w configu można wskazać nazwę datasetu i kolumny, które odpowiadają promptowi i odpowiedzi​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Otherwise%2C%20provide%20a%20mapping%20of,For%20example)

​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=You%20can%20specify%20a%20list,For%20example)

.

Przykład formatu `completions` w pliku JSONL dla prostego zadania pytanie-odpowiedź w języku polskim:

``` {"prompt": "Pytanie: Jaka jest stolica Francji?", "completion": "Odpowiedź: Stolicą Francji jest Paryż."} ```

W powyższym przykładzie model będzie trenowany, by na prompt _"Pytanie: Jaka jest stolica Francji?"_ wygenerować _"Odpowiedź: Stolicą Francji jest Paryż."_. Warto zauważyć, że utrzymano styl pytania i odpowiedzi – konsekwencja formatowania jest istotna. W datasetach instruktażowych (np. Alpaca, Dolly) często stosuje się podobny schemat z oznaczaniem roli pytania i odpowiedzi.

Dla formatu `chat`, jeśli chcemy przedstawić dialog z jedną wymianą użytkownik-model:

``` {"messages": [ {"role": "user", "content": "Witaj, jak się dziś czujesz?"}, {"role": "assistant", "content": "Dziękuję, czuję się świetnie i jestem gotów do pomocy."} ]} ```

Taki format przydaje się, gdy trenujemy model do bycia konwersacyjnym asystentem – posiada on historię wiadomości. Można też uwzględnić rolę `"system"` na początku z instrukcją (np. `"system": "You are a helpful assistant."`).

**Dobór danych do zadania (np. reasoning) – przykłady Stanford S1, s1k, DeepSeek R1**: Gdy celem jest wyspecjalizowanie modelu w **rozumowaniu** (np. rozwiązywaniu zadań matematycznych, logicznych, wieloetapowych), kluczowe jest posiadanie odpowiedniego datasetu. Dwa podejścia ilustrują projekty **Stanford S1** i **DeepSeek R1**.

- Zespół Stanforda zaprezentował w 2025 roku model **s1** wraz z niewielkim zbiorem danych **s1k** (1000 przykładów) do nauki *reasoningu*. Co ważne, te 1000 przykładów nie było losowych – zostały **starannie wyselekcjonowane i przygotowane** spośród ~59k kandydatów​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=Initial%20Collection)

    . Źródła obejmowały zadania matematyczne (NuminaMATH, AIME), pytania z egzaminów standaryzowanych (AGIEval), problemy z nauk ścisłych (OlympicArena) i inne, a do każdego pytania dołączono rozwiązanie/rozumowanie (tzw. *chain-of-thought*) oraz poprawną odpowiedź. Proces tworzenia s1k składał się z kilku etapów: (1) **filtrowanie jakościowe** – odrzucono błędne lub źle sformatowane dane, (2) **filtrowanie po trudności** – usunięto zbyt łatwe pytania. Konkretnie, przepuszczono każde pytanie przez dwa modele (mały 7B i większy 32B), a jeśli choć jeden model odpowiedział poprawnie, to pytanie uznano za zbyt łatwe i odrzucono. Pozostawiono tylko te przykłady, które **obie** wersje modelu rozwiązały źle​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=In%20this%20stage%2C%20the%20researchers,stage%20keeps%20approximately%2025%2C000%20samples)

    . To gwarantowało, że w s1k zostają problemy, które nawet dość zaawansowany model bazowy uznaje za trudne – zmuszając model do nauki nowych schematów rozumowania. (3) **Zapewnienie różnorodności** – pogrupowano pozostałe ~25k trudnych zadań na kategorie (matematyka, fizyka, biologia, logika itp.) i wybrano z nich ograniczoną liczbę tak, by dataset końcowy pokrywał szeroki wachlarz typów problemów, preferując te z dłuższymi rozwiązaniami (co sugeruje większą złożoność)​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=This%20stage%20has%20two%20steps,of%20samples%20from%20all%20domains)

    . Po tych etapach otrzymano finalne **s1k = 1000 przykładów wysokiej jakości**. Co zaskakujące, okazało się, że model **s1-32B** wytrenowany tylko na tych 1000 przykładach osiągnął wynik niemal dorównujący znacznie większym modelom (np. OpenAI o1) na zadaniach wymagających rozumowania, udowadniając hipotezę, że **jakość danych jest ważniejsza niż ilość】. Autorzy nawiązali tu do pracy Meta AI **LIMA: Less is More for Alignment**, która pokazała, że starannie dobrany 1000 przykładów instruktażowych wystarczył, by dostroić 65B model do zadań dialogowych na poziomie porównywalnym z dużo większymi zbiorami​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=This%20approach%20of%20using%20a,to%20activate%20and%20refine%20it)

    ​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=for%20alignment%20,to%20activate%20and%20refine%20it)

    . Innymi słowy, zarówno LIMA, jak i Stanford S1 dowodzą zasady _„mniej danych, ale najwyższej jakości, może dać lepszy efekt niż masy danych słabej jakości”_.

- Z kolei projekt **DeepSeek R1** obrał inną strategię dla modeli rozumujących. Model R1 bazował na ogromnym modelu (671B) trenowanym z użyciem wzmocnienia (RL) i self-play, a następnie twórcy opracowali jego warianty *distilled* i *refined*. W kontekście datasetów, DeepSeek R1 wygenerował olbrzymie ilości danych treningowych poprzez symulacje i samodoskonalenie. Wspomniany został model **r1-distill**, który osiągnął bardzo wysoką skuteczność, ale wymagał aż 800 razy więcej przykładów niż model s1​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=o1,appear%20on%20the%20far%20right)

    ​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=,requires%20800%20times%20more%20samples)

    . Oznacza to, że podczas gdy s1 wykorzystał 1k przykładów, r1-distill użył rzędu 800k (prawdopodobnie wygenerowanych lub zebranych syntetycznie) do destylacji zdolności R1 w mniejszy model (Qwen 14B). DeepSeek R1 pokazał, że mając ogromne zasoby, można brute-force’owo wytrenować potężny model rozumujący, ale koszt w danych i obliczeniach jest astronomiczny. Z praktycznego punktu widzenia, dla indywidualnych twórców modeli, podejście stanfordzkie (kuracja niewielkiego, ale treściwego datasetu) jest znacznie bardziej realne niż próba odtworzenia R1.

**Tworzenie własnego skutecznego datasetu**: Bazując na powyższych przykładach, oto rady jak przygotować dobry zbiór danych do fine-tuningu:

- **Zdefiniuj jasno zadanie i format**: Najpierw ustal, co dokładnie model ma robić po dostrojeniu. Czy ma odpowiadać na pytania faktograficzne? Rozwiązywać równania? Prowadzić uprzejmą rozmowę? Od tego zależy, jakie dane zbierzesz. Określ format wejścia/wyjścia (zgodny z wymaganiami modelu/trenera) – np. dla chatbota: format rozmowy z rolami; dla klasyfikatora: prosta para {tekst, etykieta**; dla modelu tłumaczącego: zdanie w jęz. A i jego tłumaczenie w jęz. B.

- **Zbierz dane z różnych źródeł, ale trzymaj jakość**: Jeśli tworzysz dataset samemu, często łączy się dane z wielu miejsc (open-source’owe zbiory, wygenerowane syntetycznie, własnoręcznie zlabelowane). Pamiętaj jednak, by **przeglądać dane i filtrować**. Lepiej mieć 1000 starannie dobranych przykładów niż 100k byle jakich. Automatyczne modele (np. GPT-4) mogą pomóc generować dane – np. tworzyć pytania i odpowiedzi – ale trzeba sprawdzić ich poprawność. W zadaniach obliczeniowych warto generować zarówno poprawne jak i typowe błędne rozumowania, by model nauczył się odróżniać.

- **Upewnij się, że dane są trudne na tyle, by model się czegoś nauczył**: Jeśli używasz modelu do wygenerowania datasetu, uważaj, by nie karmić go potem tymi samymi danymi – bo model nauczy się tylko powtarzać, nie rozumiejąc. W przypadku reasoning, tak jak w S1, można wykorzystać istniejące modele do oceny trudności problemów i odrzucić te, które model już potrafi rozwiązać. Zostaw takie, które obecny model bazowy rozwiązuje źle – w ten sposób fine-tuning faktycznie wniesie nową wiedzę.

- **Diversity (różnorodność)**: Nawet jeśli twoje zadanie jest wąskie, postaraj się, by przykłady nie były kalką jednego schematu. Np. jeśli trenujesz model do dialogów customer support, zawrzyj różne warianty zapytań klientów, różne temperamenty wypowiedzi, różne problemy – tak, aby model nie tylko zapamiętał jedną formułkę odpowiedzi. Różnorodność pomaga modelowi uogólniać. Widać to w S1 – zadbano by s1k pokryło wiele dziedzin, by model nauczył się ogólniejszego rozumowania, a nie tylko jednego typu zagadki​

    [aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=This%20stage%20has%20two%20steps,of%20samples%20from%20all%20domains)

    .

- **Format i zgodność z modelem**: Jeśli model bazowy jest np. modelem dialogowym (takim jak LLaMA-2-chat), to format twojego datasetu powinien być zbliżony do formatu czatu (role user/assistant). Jeśli model bazowy jest zwykłym LLM (nie chat), a chcesz go dostroić do dialogu, można nadal użyć formatu z rolami, bo model się tego nauczy – warto jednak wtedy poprzedzić każdą rozmowę specjalnym tokenem systemowym, jeśli model tego oczekuje. Ogólnie trzeba znać wymagania modelu – niektóre mają specjalne tokeny (np. `<|im_start|>` w modelach OpenAI, lub wymóg określonych nagłówków). MLX upraszcza to, bo integruje się z *templatkami* HuggingFace – np. automatycznie doda sekwencje `<s>` czy specjalne znaczniki rozmów jeśli model ich używa​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=datasets)

    .

**Zasada "The more data isn’t always better"**: Wielokrotnie przewijała się już myśl, że **liczy się jakość, nie ilość**. Dodawanie danych spoza założonego zakresu może wręcz zaszkodzić – model może zacząć gorzej się uczyć (szum w danych utrudnia konwergencję). Np. dokładając do datasetu reasoning wiele „śmieciowych” przykładów, ryzykujemy, że model nauczy się złych nawyków lub zmarnuje kapacytet na zapamiętywanie błędnych informacji. Stanford S1 kontra DeepSeek R1 to dobry kontrast: s1 osiągnął *imponującą skuteczność będąc najbardziej efektywnym pod względem liczby próbek modelem* w zestawieniu​

[aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=o1,appear%20on%20the%20far%20right)

. R1-distill co prawda przekroczył jego accuracy, ale za cenę nieporównywalnie większych zasobów. Dla większości użytkowników zwiększanie datasetu ponad pewną rozsądną ilość zwraca *coraz mniejsze korzyści* – często lepiej poświęcić czas na poprawienie/oczyszczenie istniejących np. 5k przykładów niż dozbierać kolejne 50k o wątpliwej jakości. Oczywiście, jeśli możemy mieć i dużo, i jakościowo, to najlepiej mieć dużo *dobrych* danych – ale to rzadkie. W praktyce: znajdź balans. Monitoruj wskaźniki podczas treningu: jeśli dokładasz dane, a walidacja się pogarsza lub stoi w miejscu, może dodatkowe dane nie wnoszą informacji lub wprowadzają chaos. Czasem mieszanie danych z różnych źródeł wymaga **ważenia lossów** lub kolejności – np. najpierw trenować na bardziej ogólnych, potem dopiero dokładać najtrudniejsze, albo odwrotnie. Te niuanse jednak wychodzą poza wstęp – wymagają eksperymentów.

## Najważniejsze zalecenia (do’s) i przeciwwskazania (don’ts) oraz wstępne porady praktyczne

Na koniec tej części przedstawiamy zbiór praktycznych wskazówek – rzeczy, które warto robić i takich, których należy unikać – podczas dostrajania modeli językowych:

**Zalecenia (Do’s):**

- **Zadbaj o odpowiedni dobór danych treningowych** – Upewnij się, że dane użyte do dostrajania są **reprezentatywne** dla zadania, które model ma wykonywać po treningu. Sprawdź ręcznie przykładowe wpisy, oczyść je z błędów, standaryzuj format. Jeśli tworzysz model do dialogu, trenuj na dialogach; jeśli do tłumaczenia – na zdaniach i ich tłumaczeniach itd. Unikaj danych, które mogłyby sprzecznie wpływać na model (np. mieszania stylów odpowiedzi bez jasnego rozróżnienia).

- **Używaj zestawu walidacyjnego** – Zawsze wydziel część danych jako **walidację** (validation set) i **nie trenuj** na nich modelu. Monitoruj metryki na walidacji w trakcie treningu. Pozwoli Ci to wychwycić nadmierne dopasowanie (gdy strata treningowa spada, a walidacyjna rośnie – oznaka overfittingu). Dzięki walidacji możesz też porównać różne konfiguracje (np. różne wartości learning rate, różne liczby epok) obiektywnie.

- **Stosuj wczesne zatrzymanie (early stopping)** – Jeśli zauważysz, że po pewnej liczbie epok metryki walidacyjne przestają się poprawiać lub zaczynają się pogarszać, przerwij trening. Unikniesz tym samym przeuczenia modelu. Czasem warto zachować najlepszy model z walidacji (tzw. checkpoint) i na nim zakończyć dostrajanie.

- **Eksperymentuj z hiperparametrami w bezpiecznych granicach** – Fine-tuning nie wymaga zwykle ogromnych learning rate. Zacznij od dość niskiego (np. $2\mathrm{e}{-5}$ dla transformers) i obserwuj. Zbyt duży learning rate może szybko zniszczyć pretrenowaną wiedzę modelu (*catastrophic forgetting*). Batch size – próbuj taki, jaki wejdzie w pamięć, ale nie jest krytyczne by był bardzo duży (w przeciwieństwie do pre-treningu, gdzie ogromne batch size były normą, w fine-tuningu często batch 8–32 wystarcza, a nawet 1–2 w przypadku bardzo małych datasetów). Jeśli używasz LoRA, możesz też eksperymentować z liczbą adapterów (np. rank=4 vs rank=16) – wyższy rank pozwoli modelowi więcej się nauczyć, ale wymaga więcej danych by nie przeuczyć.

- **Regularnie sprawdzaj jakość modelu na przykładach testowych** – Po treningu (a nawet w trakcie, jeśli to możliwe) testuj model na kilku przykładowych zadaniach, które dobrze znasz. Jako twórca datasetu zapewne wiesz, jak powinna wyglądać poprawna odpowiedź. Taka ręczna kontrola pomoże wychwycić np. dziwne zachowania modelu, błędy w formacie odpowiedzi czy stronniczość. Walidacja liczbowa (np. accuracy, BLEU, loss) nie zawsze pokaże Ci wszystko – manualne testy na kilku promptach mogą ujawnić np. że model zawsze zaczyna od przeprosin albo że generuje bardzo krótkie odpowiedzi.

- **Zapisuj wyniki i logi** – Prowadź notatki z eksperymentów: jakie dane użyłeś, jakie hiperparametry, jaki wynik walidacji. Ułatwi Ci to udoskonalanie procesu. Dobrą praktyką jest też zapisywanie checkpointów modelu co pewien czas (np. co epokę) – w razie gdyby ostatnia epoka pogorszyła wyniki, masz model z wcześniejszej.

- **Zabezpiecz się przed utratą ogólnych zdolności modelu** – Jeśli chcesz, by model po fine-tuningu nadal był w miarę uniwersalny (nie tylko świetny w wąskim zadaniu), rozważ strategie takie jak: _mixing_ – domieszanie do treningu od czasu do czasu przykładów z oryginalnego pretrainingu (zapobiega zapominaniu), lub _regularizacja_ – np. dodanie do loss funkcji karzącej duże zmiany wag (L2 regularization względem wag bazowych). Przykładowo, niektórzy badacze fine-tunując modele do bycia asystentami rozmowy dodawali niewielką porcję danych z oryginalnego korpusu, by model nie zapomniał jak pisać w innych stylach.

**Przestrogi (Don’ts):**

- **Nie używaj danych niskiej jakości / nieoczyszczonych** – Jednym z największych błędów jest założenie, że „model sam się domyśli”. Jeśli karmimy go chaotycznym, błędnym zbiorem, model oczywiście nauczy się tego chaosu. Unikaj zdublowanych przykładów, losowych ciągów, danych z błędami językowymi (chyba że to zamierzony efekt). W szczególności w zadaniach wymagających precyzji (np. obliczenia) upewnij się, że rozwiązania w dataset są poprawne – inaczej model nauczy się rozwiązywać *źle*. Sprawdź też format: np. brak zamykającego cudzysłowu w JSON może popsuć wczytanie danych lub zbić model z tropu co do formatu odpowiedzi.

- **Nie przeuczaj modelu (overfitting)** – Overfitting objawia się tym, że model działa świetnie na danych treningowych, ale słabo poza nimi. Aby temu zapobiec, nie trenuj zbyt długo na małym zbiorze. Jeśli masz tylko 100 przykładów, to przepuszczenie ich 100 razy przez model prawie na pewno spowoduje, że model je zapamięta dosłownie. Lepiej wówczas użyć technik augmentacji (o ile możliwe) lub przyjąć, że drobny błąd generalizacji jest ceną za mały dataset. Overfitting często widać, gdy krzywa loss dla treningu idzie w dół, a loss dla walidacji zaczyna rosnąć – to sygnał, by przerwać. Innym sygnałem jest, gdy model generuje niemal identyczne odpowiedzi dla różnych zapytań – może to znaczyć, że „nauczył” się jednej odpowiedzi ze zbioru treningowego i ją powtarza.

- **Nie mieszaj sprzecznych celów bez odpowiedniego oznaczenia** – Jeśli chcesz dostroić model do wielu zadań na raz (tzw. multi-task fine-tuning), upewnij się, że model dostaje wyraźny sygnał, co ma zrobić. Np. nie karm go jednym ciągiem: zdanie do przetłumaczenia, a innym razem: pytanie do odpowiedzenia, bez żadnego rozróżnienia. Model może zgłupieć, czy ma tłumaczyć, czy odpowiadać. Dodaj np. instrukcję w prompt typu "<Task: Translation>`<`zdanie`>`" vs "<Task: QA>`<`pytanie`>` – i dopiero na tej podstawie model wygeneruje odpowiedź. Bez tego model spróbuje uśrednić zachowania dla obu zadań i wyjdzie coś niespójnego. Innymi słowy, **unikaj wieloznaczności w danych treningowych**.

- **Nie ignoruj praw autorskich i etyki danych** – To punkt często pomijany, ale istotny: upewnij się, że masz prawo używać danych, na których trenujesz model, zwłaszcza jeśli planujesz udostępnić model publicznie. Np. nie fine-tunuj na dużym zbiorze tekstów zastrzeżonych prawnie (bez licencji), bo efekt modelu może naruszać te prawa. Podobnie, nie używaj danych wrażliwych (np. informacji osobistych) – model może je zapamiętać i potencjalnie ujawnić. W kontekście Apple – jeśli korzystasz z narzędzi Apple, one ci nie zabronią pewnych danych, ale dobra praktyka inżynierska nakazuje dbałość o zgodność z licencjami i etykę AI.

- **Nie oczekuj cudów od zbyt małego modelu lub datasetu** – Choć fine-tuning potrafi zdziałać wiele, to np. próbując nauczyć 60-milionowy model GPT-2 pełnienia roli ChatGPT możesz się rozczarować. Model ma pewną pojemność i jeśli zadanie mocno przekracza jego możliwości, fine-tuning nie pomoże (a czasem wręcz pogorszy, bo model zacznie bredzić starając się dopasować do niemożliwego). Podobnie, jeśli dasz tylko 5 przykładów treningowych, model raczej nie nauczy się nowej umiejętności – co najwyżej lekko przekierujesz jego zachowanie. Dlatego stosuj fine-tuning tam, gdzie ma to sens: model bazowy już coś umie w danym obszarze, a my go chcemy udoskonalić lub wyspecjalizować, lub gdy model jest na tyle duży, że ma potencjał nauczyć się naszego zadania. Dla ekstremalnie niskich zasobów danych rozważ *prompt tuning* zamiast pełnego trenowania (czyli wyspecjalizowanie kilku parametrów wejściowych) – to jednak osobna technika.

**Wstępne porady praktyczne:**

- Jeśli dopiero zaczynasz przygodę z fine-tuningiem LLM na Macu, rozważ skorzystanie z gotowych przykładów Apple MLX. Repozytorium `mlx-examples` na GitHubie zawiera skrypty i konfiguracje (np. do LoRA, QLoRA) przystosowane do różnych modeli​

    [apeatling.com](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/# :~:text=Machine%20learning%20research%20folks%20at,thus%20bringing%20significant%20performance%20improvements)

    ​

    [apeatling.com](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/# :~:text=For%20this%20guide%20we%E2%80%99re%20going,tuning%20adapters)

    . Możesz tam znaleźć np. przykłady dostrajania Mistral 7B do SQL (jak w blogu Niklasa Heidloffa). Warto zacząć od tych skryptów, modyfikując je pod swoje dane, zamiast pisać wszystko od zera.

- **Sprawdź wymagania modelu bazowego**: Przed treningiem zobacz, czy model wymaga specjalnego tokenizatora, czy ma jakieś nietypowe tokeny (np. <|endoftext|>). Zwykle modele z Hugging Face są dość standaryzowane, ale np. modele instruktażowe mogą oczekiwać sekwencji startowej (jak ### Human: ... ### Assistant: ...). Jeśli tego nie podasz, model też się nauczy, ale może osiągnąć gorszą płynność.

- **Używaj narzędzi monitorujących**: Na macOS możesz monitorować zużycie pamięci i GPU w narzędziu *Activity Monitor*. Istnieją też rozwiązania jak `tensorflow-metal` i `pytorch-mps` które podają logi wykorzystania. W MLX możesz zobaczyć w konsoli, ile parametrow jest trenowanych i jak szybko idzie iteracja​

    [apeatling.com](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/# :~:text=Loading%20pretrained%20model%20Total%20parameters,586)

    – wykorzystaj te informacje. Jeśli widzisz, że iteracje lecą np. 1 it/s, a tokenów/s bardzo mało, może sekwencja jest za długa (przytnij inputy) albo batch za duży (zmniejsz).

- **Zadbaj o powtarzalność (seed)**: Ustawiając *seed* losowy dla inicjalizacji i shufflowania danych możesz sprawić, że eksperyment będzie powtarzalny (przydatne do debugowania). W MLX można ustawić seed w konfiguracji. To pomoże porównać wpływ zmian – eliminując czynnik losowy.

- **Stopniowo zwiększaj trudność**: Czasem stosuje się technikę curriculum learning – najpierw trenuj na łatwiejszych przykładach, potem trudniejsze. Np. jeśli masz zarówno bardzo podstawowe jak i skomplikowane zadania w zbiorze, można początkowo wytrenować model tylko na tych łatwiejszych (aby złapał podstawy), a następnie w kolejnym etapie dodać trudne. To może zapewnić stabilniejszy trening, szczególnie dla modeli, które na początku mogą mieć problem z bardzo złożonymi przykładami (model może utknąć w lokalnym minimium).

Podsumowując: Fine-tuning to potężne narzędzie do wyciśnięcia z dużego modelu tego, czego potrzebujemy, ale wymaga pewnej ostrożności i doświadczenia. Powyższe do’s and don’ts pomogą uniknąć najbardziej powszechnych błędów. W kolejnych częściach przewodnika zajmiemy się m.in. szczegółowymi przykładami dostrajania z użyciem Apple MLX na konkretnych modelach i zadaniach, analizą wyników oraz optymalizacją inferencji dostrojonych modeli na Apple Silicon.

% Bibliografia
\begin{thebibliography}{9}
\bibitem{Hu2021LoRA} Hu, Edward J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 

[arxiv.org](https://arxiv.org/abs/2106.09685# :~:text=example%20,trainable%20parameters%2C%20a%20higher%20training)

.
\bibitem{Dettmers2023QLoRA} Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314 

[arxiv.org](https://arxiv.org/abs/2305.14314# :~:text=,new%20data%20type%20that%20is)

.
\bibitem{Sanh2019DistilBERT} Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108 

[arxiv.org](https://arxiv.org/abs/1910.01108# :~:text=for%20building%20task,device%20study)

.
\bibitem{Hinton2015Distillation} Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531 

[arxiv.org](https://arxiv.org/abs/1503.02531# :~:text=are%20large%20neural%20nets,a%20mixture%20of%20experts%2C%20these)

.
\bibitem{Fedus2021Switch} Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv:2101.03961 

[arxiv.org](https://arxiv.org/abs/2101.03961# :~:text=,training%20techniques%20help%20wrangle%20the)

.
\bibitem{Raschka2025Reasoning} Raschka, S. (2025). *Understanding Reasoning LLMs*. Blog article (Feb 2025) 

[openreview.net](https://openreview.net/forum?id=UMywlqrW3n# :~:text=answering%20,select%20the%20best%20answer%20for)

​

[openreview.net](https://openreview.net/forum?id=UMywlqrW3n# :~:text=Mixture,results%20compared%20to%20baselines%20without)

.
\bibitem{Stanford2025S1} Qiu, X., et al. (2025). *s1: Simple Test-Time Scaling*. Stanford University. arXiv:2501.19393 (v2) 

[aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=In%20this%20stage%2C%20the%20researchers,stage%20keeps%20approximately%2025%2C000%20samples)

​

[aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=This%20approach%20of%20using%20a,to%20activate%20and%20refine%20it)

.
\bibitem{DeepSeek2025R1} Team DeepSeek.AI (2025). *DeepSeek R1 Technical Report*. arXiv:2501.12948 

[aipapersacademy.com](https://aipapersacademy.com/s1/# :~:text=o1,appear%20on%20the%20far%20right)

.
\bibitem{MLX2024docs} Apple MLX Team (2024). *MLX: An array framework for machine learning on Apple silicon – Documentation and Examples* 

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=1,Setup%20section%20for%20more%20details)

​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=For%20example%2C%20for%20a%20machine,following%20should%20run%20reasonably%20fast)

.
\end{thebibliography}

#####
######
#####
######
# Wstęp\label{sec:wstep}
Historia rozwoju sprzętu GPU oraz algorytmów AI/ML jest naznaczona przełomami, które ukształtowały obecną erę sztucznej inteligencji. Jeszcze pod koniec XX wieku układy graficzne służyły głównie przyspieszaniu renderowania gier, jednak w 2007 r. firma NVIDIA udostępniła platformę CUDA, otwierając GPU na obliczenia ogólnego przeznaczenia. Już w 2009 r. badacze (m.in. z grupy Andrew Ng) pokazali, że na GPU można trenować duże sieci neuronowe znacznie szybciej niż na CPU​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=2009)

. Momentem zwrotnym był rok 2012 – zespół pod kierunkiem Geoffreya Hintona wygrał konkurs ImageNet dzięki głębokiej sieci konwolucyjnej (AlexNet) trenowanej na układach GPU​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=2012)

. Był to sygnał początku rewolucji deep learning, po którym nastąpił lawinowy wzrost zainteresowania sieciami neuronowymi. Już około 2015 r. modele oparte o głębokie uczenie osiągały poziom lepszy od człowieka w zadaniach rozpoznawania obrazów czy mowy​

[blogs.nvidia.com](https://blogs.nvidia.com/blog/first-gpu-gaming-ai/# :~:text=By%202015%2C%20AI%20had%20reached,neural%20networks%20running%20on%20GPUs)

, napędzane masywną równoległą mocą obliczeniową układów GPU.

Kolejne przełomy przyszły w drugiej połowie lat 2010: w 2014 r. powstały generatywne sieci przeciwstawne (GAN), dając początek nowym metodom generowania obrazów, a w 2016 r. system AlphaGo pokonał mistrza go, pokazując potęgę połączenia sieci neuronowych i uczenia ze wzmocnieniem. W 2017 r. badacze z Google zaprezentowali architekturę transformera​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=Google%20researchers%20developed%20the%20concept,LLMs)

, która zrewolucjonizowała przetwarzanie języka naturalnego i otworzyła drogę do budowy wielkich modeli językowych (LLM). Od tego czasu rozmiary modeli eksplodowały – dla przykładu w 2020 r. model GPT-3 miał aż 175 miliardów parametrów​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=Open%20AI%20released%20the%20GPT,to%20generate%20humanlike%20text%20models)

, co wymagało ogromnych zasobów obliczeniowych do treningu. Jednocześnie społeczność AI zaczęła korzystać z coraz większych klastrów GPU i specjalizowanych akceleratorów, aby trenować tak złożone sieci. Punktem kulminacyjnym popularyzacji AI stał się końcówka 2022 r., gdy udostępniono publicznie interfejs ChatGPT oparty o model GPT-3.5​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=Intel%20claimed%20its%20FakeCatcher%20real,accurate)

. W ciągu zaledwie kilkudni od premiery z usługi skorzystały miliony użytkowników, co podkreśliło, że AI wkroczyła do mainstreamu.

Dynamiczny rozwój AI nie byłby możliwy bez równoległego postępu w sprzęcie – zwłaszcza układach GPU. Obecnie (2025) układy graficzne są podstawą infrastruktury AI; NVIDIA dominuje rynek z ~90% udziałem i podaje, że ponad 40 tysięcy firm na świecie wykorzystuje jej GPU do projektów z obszaru sztucznej inteligencji​

[mobidev.biz](https://mobidev.biz/blog/gpu-machine-learning-on-premises-vs-cloud# :~:text=NVIDIA%2C%20which%20holds%20around%2090,machine%20learning%20and%20artificial%20intelligence)

. Rosnąca moc obliczeniowa pozwala trenować coraz głębsze modele w rozsądnym czasie, podczas gdy ulepszone algorytmy i architektury (transformery, modele uśredniające diffuzję itp.) zapewniają skoki jakościowe. Ta synergia sprzętu i algorytmów doprowadziła nas do obecnego momentu, w którym systemy AI osiągają zdumiewające rezultaty – od generowania realistycznych obrazów, przez konwersacyjne AI pokroju ChatGPT, po modele przewyższające ludzi w zadaniach specjalistycznych. Niniejszy rozdział przedstawia przegląd aktualnych technologii ML i AI na luty 2025 r., omawiając oprogramowanie (frameworki), dostępny sprzęt, kwestie wydajności i kosztów oraz praktyczne strategie doboru rozwiązań.

# Biblioteki iframeworkiML\label{sec:frameworki}
Sukces projektów machine learning w dużej mierze zależy od wyboru odpowiedniego frameworka – biblioteki programistycznej, która dostarcza narzędzia do budowy i trenowania modeli. W 2025 r. dojrzały ekosystem oferuje wiele opcji, z których każda ma swoje zalety i zastosowania. Poniżej przeglądamy najważniejsze z nich.

**PyTorch** – obecnie jeden z dwóch najpopularniejszych frameworków deep learning. Otwarty kod rozwijany oryginalnie przez Facebook (Meta), ceniony za prostotę i “pythoniczność”. PyTorch wykorzystuje dynamiczne budowanie sieci (ang. *define-by-run*), co ułatwia eksperymentowanie – struktura modelu może zmieniać się w trakcie wykonania. Dzięki przyjaznej składni i bogatej społeczności (wiele przykładów, tutoriali, gotowych modeli) stał się ulubionym narzędziem badaczy i inżynierów – w ankietach ok. 71% deweloperów wskazuje PyTorch jako łatwiejszy w użyciu niż TensorFlow​

[f22labs.com](https://www.f22labs.com/blogs/pytorch-vs-tensorflow-choosing-your-deep-learning-framework/# :~:text=PyTorch%20vs%20TensorFlow%3A%20Choosing%20Your,how%20PyTorch%20and%20TensorFlow)

. Jednocześnie PyTorch jest wystarczająco wydajny do produkcyjnych zastosowań; wspiera akcelerację na GPU, posiada też interfejsy do C++ (tzw. LibTorch) umożliwiające wdrażanie modeli poza środowiskiem Pythona. Wielkie firmy (np. Tesla) również korzystają z PyTorch przy trenowaniu swoich sieci​

[geekflare.com](https://geekflare.com/dev/jax-vs-pytorch/# :~:text=PyTorch%20is%20a%20machine,under%20the%20Linux%20Software%20Foundation)

. Główną przewagą PyTorch jest elastyczność i szybki cykl prototypowania, choć pewną wadą bywa nieco niższa wydajność niż wysoce zoptymalizowane rozwiązania Google (JAX/TPU)​

[geekflare.com](https://geekflare.com/dev/jax-vs-pytorch/# :~:text=Performance%20Jax%20is%20incredibly%20fast,to%20follow%20and%20pick%20up)

.

**TensorFlow** – flagowy framework Google, który zdominował rynek we wczesnych latach rozwoju deep learning (ok.~2015--2018). TensorFlow operuje domyślnie na statycznym grafie obliczeń: model jest definiowany w całości przed wykonaniem, a następnie optymalizowany i uruchamiany jako niezależny, zoptymalizowany program. Podejście to umożliwia zaawansowane optymalizacje i łatwe wdrażanie (eksport grafu do produkcji, np. w C++ lub na urządzenia mobilne), ale było trudniejsze w użyciu dla badaczy iteracyjnie eksperymentujących z modelami. W odpowiedzi TensorFlow 2.x wprowadził tryb *Eager Execution* upodabniający go do PyTorch (dynamiczne wykonanie), a także zintegrował interfejs Keras jako wysoko-poziomowe API. Mimo to, wiele osób nadal uważa TensorFlow za bardziej wymagający w nauce. Niemniej, pozostaje on szeroko stosowany w środowisku produkcyjnym – zwłaszcza w Google i pokrewnych ekosystemach – dzięki świetnemu wsparciu skalowania (TensorFlow Distributed) i narzędziom MLOps (TensorFlow Serving, TFX). W kontekście wydajności, TF potrafi wykorzystać kompilator XLA (Accelerated Linear Algebra) aby automatycznie przyspieszyć kod, co w niektórych testach daje mu przewagę nad czystym PyTorch​

[softwaremill.com](https://softwaremill.com/ml-engineer-comparison-of-pytorch-tensorflow-jax-and-flax/# :~:text=Flax%20softwaremill,behind%20it%20is%20straightforward%3A)

. Podsumowując, PyTorch jest obecnie bardziej popularny w pracach badawczych i szybkich iteracjach, zaś TensorFlow bywa wybierany do dużych systemów produkcyjnych i kiedy ważna jest integracja np. z narzędziami mobilnymi (TensorFlow Lite) czy dedykowanymi układami (TPU).

**JAX** – stosunkowo nowy gracz od Google Research, będący następcą biblioteki autograd i koncepcji z TensorFlow. JAX to właściwie biblioteka numeryczna (działająca podobnie do numpy) wzbogacona o automatyczne różniczkowanie i kompilację just-in-time (JIT) do kodu na GPU/TPU poprzez XLA. W odróżnieniu od PyTorch, JAX zachęca do programowania w stylu funkcyjnym – definiujemy czyste funkcje obliczające np. stratę, a JAX potrafi automatycznie obliczyć jej gradient (`grad`), zmapować obliczenia na wiele urządzeń (`pmap`) czy zwinąć pętle w wektoryzowane operacje (`vmap`). Podejście JAX wymaga innego sposobu myślenia – np. funkcje muszą być stateless (bez modyfikacji globalnych zmiennych), co początkowo zwiększa krzywą nauki. Jednak nagrodą jest znakomita wydajność: kod JAX skompilowany przez XLA często wyprzedza ekwiwalentny PyTorch na tych samych GPU​

[geekflare.com](https://geekflare.com/dev/jax-vs-pytorch/# :~:text=Performance%20Jax%20is%20incredibly%20fast,to%20follow%20and%20pick%20up)

, a dodatkowo JAX ma natywne wsparcie dla układów TPU Google. Z tego powodu JAX zyskał popularność wśród zaawansowanych zespołów badawczych (np. OpenAI, DeepMind) do trenowania najnowszych modeli, gdzie nawet kilka-kilkanaście procent oszczędności czasu czy pamięci jest cenne. Z drugiej strony, ekosystem JAX jest mniej dojrzały – brakuje mu dużej części wysokopoziomowych narzędzi PyTorch/TF, mniejsza jest społeczność i zasoby edukacyjne​

[geekflare.com](https://geekflare.com/dev/jax-vs-pytorch/# :~:text=Ease%20of%20use%20While%20it,and%20production%20machine%20learning%20models)

. JAX jest świetny do eksperymentów z newralgicznymi fragmentami modeli i optymalizacji na sprzęcie, ale do typowych projektów wielu inżynierów wciąż wybiera łatwiejszy PyTorch.

**MLX (Machine Learning eXchange)** – to nowy framework open-source opracowany przez Apple (Apple Research) z myślą o ich układach Apple Silicon. Został zaprojektowany przez badaczy ML dla badaczy ML, kładąc nacisk na wygodę użytkowania przy jednoczesnym wykorzystaniu możliwości sprzętu Apple​

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

. MLX integruje się z bibliotekami Apple (Metal Performance Shaders, Accelerate) i korzysta z ujednoliconej pamięci M1/M2, dzięki czemu redukuje narzut na kopiowanie danych między CPU a GPU. W praktyce, MLX umożliwia trenowanie i dostrajanie modeli na MacBookach i Macach z czipami M1/M2/M3 z wydajnością przewyższającą często tradycyjne frameworki uruchomione na tych samych urządzeniach. Przykładowo, na MacBooku Pro z czipem M3 można fine-tuningować lokalnie model LLM 7B (np. Mistral-7B) w mniej niż 10 minut​

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

– to wynik imponujący, pokazujący optymalizacje MLX pod kątem GPU Apple i~Neural Engine. MLX jest też zintegrowany z ekosystemem Apple: pojawiają się narzędzia wykorzystujące go (np. w repozytorium `mlx_community` na HuggingFace udostępniono wiele gotowych modeli zoptymalizowanych pod MLX). Ograniczeniem MLX jest jego portability – działa tylko na sprzęcie Apple (ARM64 + macOS, wykorzystując Metal), więc nie jest uniwersalnym rozwiązaniem dla klastrów heterogenicznych. Jednak dla deweloperów pracujących na Macach stanowi wartościową alternatywę względem PyTorch/TensorFlow z backendem MPS, oferując lepszą wydajność w wielu zadaniach (benchmarki pokazują, że MLX przewyższa PyTorch (MPS) na Macach dla większości operacji​

[reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/1aiou7i/benchmarking_apple_silicon_mlx_vs_cuda/# :~:text=Reddit%20www,CUDA%20GPUs%20remain%20inevitably)

​

[towardsdatascience.com](https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/# :~:text=How%20Fast%20Is%20MLX%3F%20A,CUDA%20GPUs%20remain%20inevitably)

).

**Hugging Face Transformers i PEFT** – Ekosystem Hugging Face odegrał ogromną rolę w popularyzacji i ułatwieniu dostępu do zaawansowanych modeli AI. Biblioteka `transformers` dostarcza jednolite API do setek pretrenowanych modeli (NLP, CV, multimodalnych) i pozwala łatwo je fine-tuningować na własnych danych. Natomiast **PEFT** (Parametr-Efficient Fine Tuning) to zestaw technik i narzędzi, które umożliwiają efektywne dostrajanie bardzo dużych modeli bez trenowania wszystkich parametrów. W praktyce implementacją tych metod jest biblioteka Hugging Face PEFT (dla PyTorch), która obsługuje takie podejścia jak LoRA (Low-Rank Adaptation), P-Tuning, Adapters itp. Zamiast modyfikować pełen model (co dla modeli rzędu dziesiątek miliardów parametrów jest kosztowne pamięciowo i obliczeniowo), PEFT dodaje niewielką liczbę nowych parametrów (lub modyfikuje niskowymiarowe podprzestrzenie) i trenuje tylko je, zamrażając oryginalne wagi modelu. Pozwala to osiągnąć niemal równoważne wyniki jak pełny fine-tuning, przy ułamku kosztu. Na przykład zastosowanie LoRA do modelu LLM może zredukować liczbę trenowanych parametrów o >99%, bez istotnej utraty jakości, co przekłada się na mniejsze zużycie GPU VRAM i szybsze trenowanie. W 2025 r. techniki PEFT są powszechnie stosowane przy dostrajaniu dużych modeli językowych – biblioteka HuggingFace PEFT integruje się z `transformers`, pozwalając w kilku liniach kodu trenować tylko tzw. adaptery zamiast całego modelu. Wadą tego podejścia jest pewne zawężenie możliwości – nie zmieniamy oryginalnych wag, więc np. nie “nauczymy” modelu zupełnie nowych kompetencji od podstaw, a raczej adaptujemy istniejącą wiedzę. Mimo to, przytłaczająca większość praktycznych projektów NLP bazuje dziś właśnie na adaptacji pretrenowanych modeli poprzez PEFT, ze względu na ogromną oszczędność zasobów.

**Axolotl, Unsloth i inne narzędzia do fine-tuningu LLM** – wraz z falą dużych modeli open-source pojawiło się zapotrzebowanie na narzędzia ułatwiające ich trenowanie i fine-tuning dla osób spoza BigTech. Axolotl i Unsloth to przykłady otwartoźródłowych frameworków powstałych w 2024 r., które upraszczają proces dostrajania modeli językowych (np. LLaMA 2, Mistral, Falcon itp.) na własnych danych.

- **Axolotl** to wszechstronne narzędzie oparte na bibliotekach Hugging Face (Transformers, peft itd.), które sprowadza konfigurację trenowania do edycji pliku YAML​

    [github.com](https://github.com/axolotl-ai-cloud/axolotl# :~:text=Go%20ahead%20and%20axolotl%20questions,run%20model%20inference%20or)

    . Zapewnia bogate wartości domyślne i optymalizacje (np. automatyczne pakowanie przykładów różnej długości w batch, by lepiej wykorzystać GPU)​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Axolotl%20comes%20with%20lots%20of,which%20can%20improve%20training%20efficiency)

    . Umożliwia trening zarówno pełnych modeli, jak i metod PEFT (LoRA, QLoRA), w tym na wielu GPU jednocześnie. Jego filozofią jest “minimum kodu, maksimum konfiguracji” – dzięki czemu użytkownik może skupić się na danych i hiperparametrach, zamiast pisać skomplikowany skrypt treningowy od zera​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Axolotl%20is%20a%20wrapper%20for,tuning%20process)

    . Axolotl jest rekomendowany początkującym oraz tym, którzy chcą łatwo skalować trening na wielu GPU​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Takeaways)

    .

- **Unsloth** z kolei został zaprojektowany z myślą o maksymalnej wydajności na pojedynczej maszynie/GPU (stąd nazwa – “pozbycie się leniwca” czyli spowolnień). Twórcą jest były inżynier Nvidii, który zaimplementował ultraszybkie wersje kluczowych operacji (zwłaszcza mechanizmu attention) w języku Triton​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=%28Flash%20Attention%202%29)

    . Dzięki temu Unsloth potrafi trenować modele LLM (LLaMA 3.1, Mistral itp.) 2–5x szybciej i zużywać ok. 80% mniej pamięci GPU w porównaniu do standardowej implementacji Hugging Face + FlashAttention 2​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Unsloth%2C%20built%20by%20Daniel%20Han,Flash%20Attention%202)

    . To imponująca poprawa, osiągnięta bez uciekania się do aproksymacji czy kwantyzacji – przyspieszenie wynika z lepszego wykorzystania pamięci i jednostek obliczeniowych GPU. Unsloth celuje w umożliwienie fine-tuningu nawet na skromnych kartach (np. na bezpłatnym Colab z jedną Tesla T4)​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=So%20for%20example%2C%20if%20you,GPU%20training%2C%20so)

    . Osiąga to m.in. poprzez ograniczenie rozmiaru batcha i sekwencji do niezbędnego minimum i ekstremalną optymalizację low-level. Warto zaznaczyć, że Unsloth nie wspiera treningu rozproszonego – działa tylko na jednym GPU (co jest świadomym kompromisem). Dlatego poleca się go przy ograniczonych zasobach sprzętowych, natomiast w projektach z dostępem do wielu GPU lepiej sprawdzi się Axolotl​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=you%20only%20have%20access%20to,smaller%20or%20older%20GPUs)

    .

- **Torchtune** – kolejny godny uwagi framework (również open-source), będący w zasadzie nakładką upraszczającą fine-tuning LLM w czystym PyTorch. W przeciwieństwie do Axolotla, Torchtune jest bardzo “lekki” – nie narzuca własnej abstrakcji treningu, tylko udostępnia pewne pomocnicze funkcje, receptury i skrypty dla typowych zadań dostrajania (w tym obsługę LoRA, QLoRA, itp.)​

    [modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Torchtune)

    . Można go traktować jako zestaw przykładów i template’ów dla ludzi, którzy swobodnie czują się pisząc kod PyTorch, ale chcą zaoszczędzić czas na implementacji boilerplate. Torchtune bywa wybierany przez zaawansowanych użytkowników preferujących pełną kontrolę nad przebiegiem treningu, podczas gdy Axolotl i podobne narzędzia celują w szybkość i wygodę kosztem ukrycia części szczegółów.

Podsumowując, wybór frameworka ML zależy od charakteru projektu i doświadczenia zespołu. Do klasycznych zadań deep learning najczęściej sięga się po PyTorch lub TensorFlow – oba są sprawdzone i mają wsparcie dla wszystkiego od sieci CNN po transformatory. W zastosowaniach badawczych wymagających ekstremalnej wydajności rozważyć można JAX (zwłaszcza przy dostępie do TPU). Na komputerach Apple z kolei warto przyjrzeć się MLX, które wydobywa pełnię możliwości M1/M2. Gdy pracujemy z dużymi modelami językowymi i chcemy je dostroić, pomocne będą specjalistyczne biblioteki: Hugging Face Transformers + PEFT jako uniwersalny zestaw, a spośród nowszych narzędzi Axolotl (łatwość użycia, multi-GPU) lub Unsloth (maksimum szybkości na jednym GPU). W razie wątpliwości można kierować się kilkoma prostymi wskazówkami: dla początkujących – Axolotl, przy ograniczonym GPU – Unsloth, preferencja czystego PyTorch – Torchtune, jak podsumowano w przeglądzie Modalu​

[modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Takeaways)

. Ważne jest też wsparcie społeczności i aktualność – te frameworki rozwijają się bardzo szybko, więc decyzja powinna uwzględniać stan na bieżący rok 2025.

# Dostępny hardware AI/ML\label{sec:hardware}
Zaplecze sprzętowe dla sztucznej inteligencji jest równie ważne co algorytmy – to dzięki coraz wydajniejszemu hardware możemy trenować większe modele na większych danych. Poniżej omawiamy główne typy akceleratorów i architektur dostępnych w 2025 roku, od procesorów CPU, przez GPU, po wyspecjalizowane układy jak TPU czy NPU, a także podejścia do budowy klastrów obliczeniowych.

\subsection*{Procesory ogólnego przeznaczenia (CPU)}
**Architektury CPU (ARM64, x86-64 i inne):** Klasyczne procesory wciąż odgrywają istotną rolę w systemach AI/ML, choć głównie jako uzupełnienie dla akceleratorów. Dominują dwie architektury: **x86_64** (procesory Intel Xeon i AMD EPYC w serwerach) oraz **ARM64** (rosnące znaczenie w HPC – np. układy Ampere w serwerach, czy Apple Silicon w desktopach). Architektura x86-64 zapewnia wysoką wydajność pojedynczego wątku i bogaty ekosystem oprogramowania, ale jest stosunkowo energożerna. Architektura ARM64 cechuje się wysoką efektywnością energetyczną i skalowalnością – nowoczesne serwerowe ARM mają po kilkadziesiąt rdzeni (AWS Graviton, Ampere Altra), co sprawdza się w obsłudze równoległych żądań inferencji lub w etapach wstępnego przetwarzania danych. Apple Silicon (M1/M2/M3) to specyficzny przypadek ARM64, gdzie zintegrowane CPU, GPU i NPU w jednym układzie SoC dają świetną wydajność w przeliczeniu na wat, choć absolutna moc obliczeniowa CPU ustępuje topowym Xeonom. W kontekście trenowania dużych modeli CPU pełni dziś głównie rolę pomocniczą (koordynacja zadań, preprocessing, augmentacja danych) lub wykonuje trening klasycznych algorytmów ML (np. drzewa decyzyjne) tam, gdzie akceleracja GPU nie jest konieczna. Warto jednak odnotować, że w dziedzinie AI pojawiają się też procesory RISC-V – w 2025 r. głównie jako ciekawostka akademicka lub kontrolery do akceleratorów, ale ich otwartość może w przyszłości przynieść niestandardowe optymalizacje dla AI.

**Nowe rozszerzenia i integracja akceleratorów:** Zarówno w x86, jak i ARM, producenci dodają instrukcje przyspieszające obliczenia AI. Intel rozwija zestawy AVX-512 (a niedawno AVX-VNNI) optymalizujące operacje macierzowe i wektorowe, a także integruje dedykowane akceleratory – np. technologia AMX (Advanced Matrix Extensions) w procesorach Intel Sapphire Rapids pozwala na szybkie mnożenie macierzy (16x16) sprzętowo, przydatne w inferencji sieci​

[nextplatform.com](https://www.nextplatform.com/2024/08/27/ibm-shows-off-next-gen-ai-acceleration-on-chip-dpu-for-big-iron/# :~:text=IBM%20Shows%20Off%20Next,be%20marketed%20as%20the%20z17)

. AMD w swoich EPYC ma analogiczne usprawnienia (AVX2, VNNI). Z kolei w świecie ARM, Apple wyposaża swoje czipy M w Neural Engine – blok wykonujący ściśle określone operacje sieci neuronowych (głównie inferencję CNN, RNN) z wydajnością sięgającą 15 TOPS przy minimalnym poborze mocy. W rezultacie nowoczesny serwer czy stacja robocza może mieć nie tylko mocne CPU, ale i wbudowane przyspieszenia AI (choć wciąż o rząd wielkości mniej wydajne niż pełnoprawny GPU). Podsumowując: CPU pozostaje fundamentem uniwersalnej infrastruktury, ale do głównych obciążeń treningowych w AI wykorzystuje się dedykowane akceleratory, omówione poniżej.

\subsection*{Procesory graficzne (GPU) i akceleratory AI}
**GPU jako główny akcelerator AI:** Od czasu przełomu z AlexNet, jednostki GPU stały się podstawową “siłą roboczą” w deep learning. Ich architektura – tysiące rdzeni zdolnych wykonywać jednocześnie proste operacje – idealnie pasuje do obliczeń macierzowych w sieciach neuronowych​

[blogs.nvidia.com](https://blogs.nvidia.com/blog/first-gpu-gaming-ai/# :~:text=Traditional%20CPUs%2C%20designed%20for%20sequential,were%20perfect%20for%20the%20job)

​

[blogs.nvidia.com](https://blogs.nvidia.com/blog/first-gpu-gaming-ai/# :~:text=In%202012%2C%20a%20breakthrough%20came,software%20written%20by%20vision%20experts)

. Liderem rynku jest **NVIDIA**, która przez ostatnie lata konsekwentnie rozwija zarówno sprzęt (kolejne architektury GPU), jak i ekosystem oprogramowania (biblioteki CUDA, cuDNN, TensorRT). Aktualnie w centrach danych królować będą układy NVIDIA z architektur *Ampere* (A100) i *Hopper* (H100). A100 (2020 r.) oferuje do 312 TFLOPS (tensor core FP16), 40 GB lub 80 GB pamięci HBM2, i już on stał się podstawą wielu superkomputerów AI. H100 (premiera końcówka 2022) to kolejny skok – ~1000 TFLOPS FP16 (z wykorzystaniem sparsy), wsparcie dla FP8, 80 GB HBM3 – co czyni go jedną z najmocniejszych jednostek treningowych dostępnych komercyjnie. NVIDIA oferuje te akceleratory także w gotowych serwerach (linie DGX, HGX) oraz usługach w chmurze. Udział NVIDII to również przewaga programowa: platforma CUDA jest de facto standardem w implementacji algorytmów AI – wiele optymalizowanych bibliotek (cuDNN, NCCL, nvBLAS) sprawia, że często “łatwiej” uzyskać pełnię wydajności na GPU Nvidii niż konkurencji.

**Alternatywne GPU:** **AMD** – główny konkurent – również rozwija linie GPU dla AI (Instinct MI). Architektura CDNA2 (np. MI250X) zasiliła najszybszy superkomputer świata, Frontier (USA), pokazując że sprzęt AMD może rywalizować z NVIDIĄ w najwyższym segmencie. W 2023 r. AMD zapowiedziało akcelerator **MI300X** wyposażony aż w 192 GB pamięci HBM3, zaprojektowany pod generatywne modele językowe, tak by nawet modele rzędu 100–170 mld parametrów zmieścić w pamięci pojedynczej karty. IBM ogłosił plany użycia MI300X w swojej chmurze do obsługi generatywnej AI​

[newsroom.ibm.com](https://newsroom.ibm.com/2024-11-18-ibm-expands-its-ai-accelerator-offerings-announces-collaboration-with-amd# :~:text=,Nov%2018%2C%202024)

. Wadą AMD jest mniej dojrzały ekosystem – ich biblioteki ROCm i kompilatory jeszcze ustępują CUDA, choć mają wsparcie dla PyTorch/TensorFlow. W zastosowaniach gdzie budżet jest kluczowy, AMD bywa tańszą alternatywą dla NVIDII (np. ta sama moc obliczeniowa w TFLOPS za niższą cenę), ale wymaga to od zespołu gotowości do dopracowania i testów oprogramowania.

Oprócz klasycznych producentów GPU, do wyścigu dołączają chińskie firmy z własnymi projektami, głównie by uniezależnić się od importu. **Huawei** rozwija od kilku lat rodzinę akceleratorów *Ascend*. Model Ascend 910 (2019 r.) początkowo plasował się poniżej topowych układów Nvidii, ale jego najnowsza iteracja Ascend 910C poczyniła postępy – w testach osiąga około 60% wydajności Nvidii H100 w zadaniach inferencji​

[tomshardware.com](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance# :~:text=Huawei%27s%20HiSilicon%20Ascend%20910C%20is,China%27s%20reliance%20on%20Nvidia%20GPUs)

. Co prawda w treningu dużych modeli 910C nadal znacząco ustępuje (ekosystem CUDA i dopracowane sterowniki dają NVIDII przewagę)​

[tomshardware.com](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance# :~:text=While%20Huawei%20and%20SMIC%20have,Nvidia%20maintains%20its%20undisputable%20lead)

, jednak fakt, że chiński układ zbliża się do możliwości A100/H100 pokazuje rosnącą konkurencję. Inni chińscy giganci, jak Alibaba (chip Hanguang) czy Baidu (Kunlun), również inwestują w AI hardware. **Tencent** na razie znany jest bardziej z masowego wykorzystania GPU Nvidii (według raportów w 2024 r. należał do największych nabywców układów H100​

[techradar.com](https://www.techradar.com/pro/chinese-cloud-giants-bought-more-of-nvidias-flagship-ai-chips-than-anybody-else-except-microsoft# :~:text=Chinese%20cloud%20giants%20bought%20more,AI%20chips%20in%202024%2C)

) oraz tworzenia własnych rozwiązań software (np. duży model językowy Hunyuan). Niewykluczone jednak, że pracuje też nad własnymi akceleratorami AI – trend ten wpisuje się w szerszy obraz: wielkie firmy technologiczne chcą projektować układy pod swoje specyficzne potrzeby (np. Google TPU, Amazon Trainium – patrz niżej).

**TPU i wyspecjalizowane układy:** **TPU** (Tensor Processing Unit) to specjalne akceleratory stworzone przez Google, dostępne w ramach Google Cloud i używane wewnętrznie. TPU są zaprojektowane stricte pod obliczenia tensorowe macierzowe (stąd nazwa) – rdzeń TPU zawiera matryce mnożące np. 128x128 elementów w jednej operacji. Już TPU v2 (2017 r.) i TPU v3 rywalizowały z ówczesnymi GPU, a obecnie dostępna generacja **TPU v4** oferuje do 275 TFLOPS BF16 każda, z możliwością łączenia wielu w tzw. pod (superkomputer z tysiącami TPU). TPU wyróżnia się tym, że został zintegrowany z ekosystemem TensorFlow (pisząc model w TF lub JAX można go uruchomić na TPU praktycznie bez zmiany kodu). Minusem jest ograniczona dostępność – tylko Google Cloud – i mniejsza wszechstronność (TPU najlepiej sprawdzają się w typowych sieciach DL; nietypowe algorytmy mogą nie zyskać tak bardzo). Niemniej, TPU były kluczowe np. przy trenowaniu modeli jak PaLM (540 mld parametrów) – Google zbudowało w tym celu pod TPUv4 o wydajności 9 eksaflopów BF16.

Innym przykładem wyspecjalizowanych układów są **NPU** (Neural Processing Units) stosowane głównie w urządzeniach mobilnych i IoT. Takie bloki znajdują się w układach SoC smartfonów (Qualcomm AI Engine, Apple Neural Engine, Huawei NPU w Kirinach) i służą do przyspieszania lokalnej inferencji – np. rozpoznawania mowy offline, efektów AR w aparacie itp. Charakteryzują się one niskim poborem mocy i ograniczoną precyzją obliczeń (często INT8 czy nawet niżej), ale potrafią wykonać kilkanaście bilionów operacji na sekundę przy zasilaniu z baterii. W kontekście trenowania modeli raczej się ich nie używa (zbyt mała uniwersalność), natomiast w aplikacjach edge-AI są niezastąpione. W urządzeniach klasy serwer pojawiają się też tzw. **IPU** (Intelligence Processing Unit) – np. firma Graphcore oferuje układy IPU zarch. masowo-równoległą, czy **wafer-scale engine** firmy Cerebras (cały 300mm wafelek krzemowy jako jeden układ AI). Te rozwiązania celują w nisze, gdzie tradycyjne GPU mają ograniczenia (np. Cerebras WSE dzięki ogromnej ilości pamięci SRAM bezpośrednio na płytce potrafi zmieścić cały model o miliardach parametrów bez potrzeby dzielenia na batch). Mimo zainteresowania medialnego, udziały rynkowe tych innowacyjnych akceleratorów są niewielkie w porównaniu do GPU.

**Kompleksowe rozwiązania sprzętowe od gigantów:** Wiele dużych firm oferuje nie tyle pojedyncze układy, co całe platformy sprzętowe zoptymalizowane pod AI. NVIDIA sprzedaje gotowe DGX-y (8x A100 lub H100 w obudowie + oprogramowanie zarządzające), co stanowi moduł budulcowy superkomputerów AI (np. NVIDIA pod nazwą DGX SuperPOD łączy dziesiątki DGX). **IBM** natomiast integruje akcelerację AI w swoich systemach mainframe – procesor *Telum II* (zapowiedziany na 2024) będzie miał wbudowany układ przyspieszający AI (IBM nazywa go *Spyre AI accelerator*) do zadań inferencji na ogromnych ilościach transakcji​

[ibm.com](https://www.ibm.com/new/announcements/telum-ii# :~:text=,performance%20cores%20running%20at%205.5GHz)

. IBM jednocześnie inwestuje w chmurę – jak wspomniano, nawiązał współpracę z AMD by użyć MI300X, oraz prowadzi prace nad własnymi półprzewodnikami neuromorficznymi w laboratoriach badawczych. **Tencent** i inne firmy oferujące usługi chmurowe (Alibaba Cloud, Baidu Cloud) budują dedykowane “maszyny AI” często bazujące na stosach GPU+CPU z niestandardowymi modyfikacjami (np. własne płyt główne z dodatkowym buforowaniem danych, autorskie interkonekty między serwerami). **Huawei** poza projektowaniem chipów Ascend oferuje kompletne serwery Atlas i klastery Atlas 900, które rywalizują na lokalnym rynku chińskim z rozwiązaniami Nvidii. Można zatem powiedzieć, że w 2025 r. istnieje bogaty wybór sprzętu: od rozwiązań amerykańskich (NVIDIA, AMD, Google TPU, Intel Habana) po chińskie (Huawei Ascend, Alibaba Hanguang, Biren GPU itp.), choć dostępność tych drugich bywa ograniczona przez regulacje eksportowe.

\subsection*{Klastrowanie akceleratorów: HPC, chmura, on-premise}
Pojedynczy układ GPU czy TPU ma imponującą moc, ale najnowocześniejsze modele AI potrzebują całych farm takiego sprzętu działającego w tandemie. Dlatego kluczowe jest efektywne łączenie wielu akceleratorów w klastry HPC (High Performance Computing). Standardem jest architektura, w której kilka-kilkanaście GPU znajduje się w jednym serwerze (tzw. node), a następnie wiele takich serwerów jest połączonych siecią o bardzo niskich opóźnieniach. NVIDIA umożliwia łączenie do 8 GPU w jednym serwerze za pomocą NVLink (szybka magistrala GPU–GPU, umożliwiająca bezpośredni transfer danych z przepustowością rzędu 600–900 GB/s między kartami) lub nawet NVSwitch (przełącznik tworzący węzeł, gdzie każdy z 8 GPU widzi każdy inny z pełną prędkością). Pomiędzy serwerami najczęściej stosuje się InfiniBand (obecnie standard to HDR 200 Gbit/s lub NDR 400 Gbit/s), ewentualnie sieci ethernetowe 100–400 Gb/s z akceleracją RDMA. Taki klaster wymaga również oprogramowania do rozproszonego trenowania – np. biblioteka NVIDIA NCCL dba o wydajną komunikację między GPU (all-reduce itp.). Dzięki tym technologiom, modele mogą być dzielone między dziesiątki czy setki GPU, przyspieszając trening proporcjonalnie do liczby kart (o ile zadanie jest wystarczająco duże, by zrównoleglić, i~nie występują wąskie gardła komunikacyjne).

**HPC on-premise vs chmura:** Tradycyjnie instytucje akademickie i duże firmy budowały własne klastry HPC na miejscu (on-premise), dostosowane do swoich potrzeb. Daje to pełną kontrolę nad sprzętem, możliwością optymalizacji do bólu (np. specjalne chłodzenie, topologie połączeń dostosowane do typowych zadań), no i przy ciągłym obciążeniu – niższy koszt jednostkowy przeliczenia, bo płaci się jednorazowo za sprzęt. Wadą są ogromne koszty początkowe i konieczność utrzymania (osobny zespół do administrowania HPC, aktualizacje, naprawy). Z kolei chmura obliczeniowa oferuje dziś praktycznie taki sam sprzęt w modelu *pay-as-you-go* – np. AWS udostępnia instancje P4d (8xA100), P5 (8xH100), Google Cloud – instancje z A100 lub TPU v4, Azure – ND A100 v4, etc. Można wynająć od kilku GPU na godziny aż po cały superkomputer na tygodnie. **Chmura** zapewnia elastyczność i natychmiastowy dostęp do dużej skali (co jest idealne np. przy jednorazowym treningu bardzo dużego modelu), bez inwestycji kapitałowych na sprzęt. Jednak opłaty w chmurze przy długotrwałym użyciu bywają wyższe niż amortyzacja własnego klastra. Dylemat opłacalności sprowadza się do intensywności obciążenia: jeśli planujemy wykorzystywać GPU okazjonalnie lub w zmiennym stopniu – chmura pozwoli nam płacić tylko za wykorzystane godziny. Jeśli natomiast potrzebujemy np. 8 GPU stale przez rok, to koszt chmury może przekroczyć cenę zakupu tych GPU. W takim przypadku on-premise jest tańsze na dłuższą metę​

[mobidev.biz](https://mobidev.biz/blog/gpu-machine-learning-on-premises-vs-cloud# :~:text=Advantages%20Disadvantages%201%20Long,consuming%20and%20expensive)

​

[mobidev.biz](https://mobidev.biz/blog/gpu-machine-learning-on-premises-vs-cloud# :~:text=Advantages%20Disadvantages%201%20Rapid%20scalability,providers%20for%20uptime%20and%20availability)

. Częstą strategią jest **hybrydowe podejście** – posiadanie pewnej lokalnej mocy obliczeniowej (do codziennych eksperymentów, pilnowania danych wrażliwych), a w razie potrzeby skalowanie w chmurze (np. na finałowy trening dużego modelu czy obsługę zwiększonego ruchu w inferencji).

Niezależnie od modelu wdrożenia, budowa klastrów AI wiąże się ze specyficznymi wyzwaniami. Po pierwsze, węzły GPU są bardzo **prądożerne** – typowy serwer 8xA100 może pobierać 3–5 kW mocy, a cały klaster łatwo osiąga wartości rzędu megawatów. Topowe instalacje jak klaster GPT-4 (opisany przez Microsoft) zużywały ok. 15 MW mocy przez 90 dni ciągłego treningu​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=In%20his%20opening%20keynote%20at,CPU%20HGX%20compute%20complex)

. Zapewnienie zasilania i chłodzenia na takim poziomie to istotna część kosztów infrastruktury. Po drugie, **skalowalność oprogramowania** – aby efektywnie wykorzystać np. 1000 GPU, kod treningowy musi być odpowiednio zaprojektowany. Używa się technik równoległości modelu (dzielenie warstw między GPU) i danych (każdy GPU trenuje na innym minibatchu) lub hybryd (tzw. pipelining). Frameworki jak PyTorch Lightning, DeepSpeed, Horovod czy biblioteki Megatron-LM ułatwiają to, ale nadal duże modele wymagają znacznej wiedzy inżynierskiej, by trening rozproszony przebiegał stabilnie i bez wąskich gardeł. W wielkich klastrach HPC stosuje się też zaawansowane mechanizmy sieciowe (np. migrowanie wątków bliżej GPU, algorytmy kolejkowania zadań) by utrzymać wysoką efektywność. To dlatego topowe prace nad modelami często wychodzą od firm posiadających zarówno sprzęt, jak i ekspertów HPC (OpenAI/Microsoft, Google, Meta).

W 2025 r. dostrzegalny jest również trend **dedykowanych “superkomputerów AI”** budowanych przez konsorcja lub państwa. Np. w Europie powstają centra HPC wyposażone w mieszane architektury (GPU Nvidia + akceleratory europejskie, jak Graphcore czy przyszłe procesory EPI), mające zapewnić niezależność w dostępie do mocy dla instytucji publicznych. Izrael zainwestował w superkomputer AI na bazie DSP dla dużych modeli NLP hebrajskich. Tego typu inicjatywy wskazują, że klasteryzacja sprzętu AI to już nie tylko domena pojedynczych firm, ale infrastruktura porównywana z tradycyjnymi zasobami HPC (jak superkomputery do symulacji).

# Analiza wydajności, kosztów i zużycia energii\label{sec:koszty}
Budując modele AI, zwłaszcza te głębokie, musimy brać pod uwagę trzy kluczowe czynniki poza samą szybkością: wydajność (w przeliczeniu na zasoby), koszt finansowy oraz koszt energetyczny/środowiskowy. W 2025 r. temat optymalizacji kosztów ML jest bardzo żywy – poniżej przedstawiamy przegląd tych zagadnień, obejmujący opłacalność chmury vs własnej infrastruktury, koszty budowy klastrów oraz metody redukcji wydatków na AI.

**Koszty trenowania w różnych chmurach:** Wielcy dostawcy (AWS, Azure, Google, Oracle itp.) konkurują ofertami instancji GPU. Ceny zależą od modelu GPU i typu wynajmu. Przykładowo, według danych z końca 2024 r., na AWS godzina pracy instancji p4de.24xlarge (8xA100 80GB) kosztowała ok. $41 on-demand, a jej nowocześniejszy następca p5.48xlarge (8xH100 80GB) ok. $98/h​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=On%20AWS%2C%20the%20p4de,16)

. To stawki za natychmiastowy wynajem; opcje z rezerwacją roczną lub trzyletnią mogą obniżyć cenę o 30–50% (np. do $43/h przy 3-letnim kontrakcie dla 8xH100)​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=based%20on%20essentially%20the%20same,16)

. Inni dostawcy mają podobne poziomy cen – różnice kilku procent wynikają z promocji lub wydajności dopłat. W praktyce, utrzymanie klastra 1000 GPU H100 w chmurze kosztuje dziesiątki milionów dolarów rocznie​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=On%20AWS%2C%20as%20you%20need,8%20million)

. Na przykład trening jednego modelu GPT-4 z 1.8 bln parametrów (Mixture-of-Experts) w 90 dni na 1000 H100 według szacunków kosztowałby ok. $120 mln przy stawkach AWS​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=On%20AWS%2C%20as%20you%20need,8%20million)

. Liczby te uzmysławiają skalę wydatków przy największych projektach – nic dziwnego, że firmy jak OpenAI czy Anthropic podpisują umowy inwestycyjne z dostawcami chmury, gdzie miliardy dolarów krążą między finansowaniem startupu a opłatami za infrastrukturę​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=Hey%2C%20if%20you%20could%20get,which%20%2027)

​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=full%20well%20that%20the%20vast,which%20%2027)

. Dla mniejszych podmiotów bardziej relewantne są jednak zadania rzędu kilku-kilkunastu GPU przez dni lub tygodnie – i tu pojawia się pytanie: *chmura czy własny sprzęt?*.

Jak wspomniano w sekcji o klastrach, **on-premise opłaca się**, gdy mamy stałe, wysokie zapotrzebowanie. Własny serwer z 8xA100 to wydatek (jednorazowy) rzędu kilkuset tysięcy dolarów, ale potem może on pracować 24/7 przez 3–5 lat. W chmurze ten sam sprzęt generowałby koszt w okolicach $30–70 tys. miesięcznie, czyli w mniej niż rok przekroczy koszt zakupu. Dlatego duże instytucje z ciągłymi projektami AI zwykle inwestują w sprzęt. Z kolei **chmura jest tańsza** dla nieregularnych obciążeń – płacimy tylko za godziny treningu. Wiele startupów zaczyna w chmurze, aby nie zamrażać kapitału, a dopiero po ustabilizowaniu potrzeb ewentualnie kupuje własne maszyny. W chmurze można też optymalizować koszty poprzez używanie instancji spot (przecenionych, gdy dostawca ma wolne zasoby) – bywa to 2x tańsze, choć ryzyko przerwania zadania wymaga mechanizmów tolerujących (zapisywanie checkpointów, wznawianie treningu). Dostawcy oferują również różne typy instancji – np. Amazon udostępnia nie tylko GPU Nvidii, ale i własne układy **Trainium** (AWS Trn1) czy **Inferentia** do wnioskowania. Trainium2 według AWS osiąga lepszy stosunek koszt/wydajność dla modeli 20 mld parametrów niż A100​

[aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/frugality-meets-accuracy-cost-efficient-training-of-gpt-neox-and-pythia-models-with-aws-trainium/# :~:text=Frugality%20meets%20Accuracy%3A%20Cost,2M%20tokens%2F%24%20spent)

, co jest próbą zachęty do tańszych alternatyw. Azure i GCP mają podobnie – np. GCP Cloud TPU v4 mogą być rozliczane od sekundy i w pewnych zadaniach wypadają cenowo lepiej niż GPU. Zatem, świadomy użytkownik AI w chmurze może przebierać między opcjami i wybierać najkorzystniejszą ekonomicznie dla danego zadania.

**Koszt budowy i utrzymania klastrów wewnętrznych:** Dla organizacji decydujących się na własny sprzęt, poza zakupem samych akceleratorów (GPU/TPU) należy uwzględnić koszty infrastruktury towarzyszącej. Wydajne zasilacze i systemy chłodzenia (klimatyzacja, woda lodowa) są niezbędne, by klaster działał niezawodnie. Przy dużej skali znaczące stają się rachunki za prąd – nie tylko zasilanie obliczeń, ale i odprowadzenie ciepła. W miejscach, gdzie energia jest droga, czasem opłaca się kolokować sprzęt w regionach z tańszym prądem lub korzystać z centrów danych operujących przy elektrowniach (dla zredukowania opłat przesyłowych). Niektóre firmy analizują nawet użycie ciepła odpadowego z GPU do ogrzewania pomieszczeń, by odzyskać część kosztów. Oprócz hardware, dochodzi czynnik **kadr** – potrzeba wyspecjalizowanych inżynierów systemowych do utrzymania takiego klastru (aktualizacje sterowników, kolejki zadań, obsługa awarii). To sprawia, że własny klaster ma wyższy próg wejścia i jest sensowny głównie przy odpowiedniej skali przedsięwzięcia lub szczególnych wymaganiach (np. dane nie mogą opuścić serwera ze względów prawnych).

**Zużycie energii i optymalizacja pod kątem efektywności:** Trening dużych modeli jest energochłonny – wspomniany GPT-4 MoE w 90 dni zużył ok. 15 MW, co odpowiada ponad 30 gigawatogodzin energii​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=In%20his%20opening%20keynote%20at,CPU%20HGX%20compute%20complex)

. To tyle, ile rocznie zużywa kilkanaście tysięcy gospodarstw domowych. W dobie rosnącej świadomości ekologicznej, społeczność AI zaczęła zwracać uwagę na **“Green AI”** – metody ograniczania śladu węglowego trenowania modeli. Jednym podejściem jest efektywniejsze oprogramowanie: np. algorytm *FlashAttention* optymalizuje pamięciożerną operację attention, redukując potrzebę wielokrotnego czytania/zapisu z pamięci – co skraca czas treningu i tym samym oszczędza energię. Inny kierunek to **niższa precyzja**: zamiast trenować modele w 32-bitowej precyzji, stosuje się FP16, BF16, a ostatnio nawet FP8 – model zużywa mniej pamięci i wykonuje mniej operacji na bitach, co przyspiesza obliczenia. Nowe układy (H100, TPU v4) mają natywne wsparcie dla FP8, co pozwala trenować z zaskakująco niewielkim spadkiem jakości, a potrafi przyspieszyć proces o 20–30%. Kolejny czynnik to **skalowanie przez większe batch size** – dzięki większej pamięci GPU można przetwarzać więcej przykładów równolegle, osiągając lepszą efektywność na takt (choć tu trzeba uważać na zbieżność uczenia). Wszystkie te techniki mają na celu *“więcej za mniej”* – czyli zwiększyć odsetek mocy obliczeniowej faktycznie przekładającej się na poprawę modelu, a zminimalizować marnotrawstwo.

**Strategie redukcji kosztów ML w 2025:**  -  **Wykorzystanie pretrenowanych modeli i transfer learning:** Zamiast trenować model od zera (co jest najdroższe), lepiej zacząć od istniejącego pretrenowanego modelu (np. w HuggingFace są tysiące dostępnych) i tylko go dostroić do naszego zadania. Oszczędza to zarówno czas (model konwerguje szybciej), jak i energię potrzebną na ogromne przebiegi trenowania. W praktyce jest to standard w NLP i CV – sieci typu BERT, GPT, ResNet służą jako baza. -  **Parameter-Efficient Fine Tuning (PEFT):** Opisane wcześniej techniki jak LoRA pozwalają trenować tylko niewielką część wag dużego modelu. Dzięki temu możemy użyć mniejszego sprzętu (np. pojedynczego GPU zamiast wielu) – bo wymagania pamięci spadają dramatycznie. W 2025 r. dostrajanie 100-miliardowego modelu językowego na jednej karcie 24 GB VRAM staje się możliwe właśnie dzięki PEFT (zamiast nierealnego scenariusza trenowania wszystkich 100 mld parametrów). -  **Optymalizacja kodu i algorytmów:** Profilowanie treningu może ujawnić wąskie gardła – np. że duża część czasu idzie na oczekiwanie na dane z dysku albo nieefektywną implementację warstwy. Poprawa tych miejsc (czy to przez lepszy sprzęt I/O, np. szybkie NVMe, czy użycie gotowych opcji jak `prefetch` w TensorFlow) potrafi skrócić czas treningu bez dokupowania nowych GPU. Również wybór optymalizatora ma wpływ – np. optymalizator AdaFactor jest znany z mniejszego zużycia pamięci niż AdamW, co pozwala zwiększyć batch lub model bez dodatkowego GPU​

[medium.com](https://medium.com/@uttamasodariya30/memory-optimization-strategies-for-fine-tuning-llms-practical-approaches-b0a4244c6347# :~:text=Compared%20to%20AdamW%2C%20AdaFactor%20is,tuning%20large%20LLMs)

. -  **Mniejsze modele i kompresja:** Czasem zamiast trenować największy możliwy model, lepiej użyć sprytniejszej architektury lub modelu z wiedzą ekspercką. Przykład: modele MoE (Mixture of Experts) potrafią mieć wiele parametrów, ale nie wszystkie aktywują się jednocześnie – przez co ich *effective compute* jest mniejsze niż modeli gęstych o tej samej liczbie parametrów. To pozwala zwiększyć pojemność modelu bez proporcjonalnego wzrostu kosztów obliczeń. Inny przykład: **distillacja modelu** – mając bardzo duży model nauczyciel, można wytrenować mniejszy model student na jego predykcjach, uzyskując model o zbliżonej jakości znacznie niższym kosztem (zarówno trenowania, jak i późniejszego użycia). Te techniki pozwalają ograniczyć koszty zarówno treningu (bo np. distylacja może trwać krócej niż trenowanie od zera tego mniejszego modelu na oryginalnym zadaniu), jak i koszty produkcyjne (mniejszy model jest tańszy w chmurze do obsługi). -  **Tańszy sprzęt i skalowanie horyzontalne:** Nie zawsze najlepszym wyborem ekonomicznym jest najnowszy topowy GPU. Czasami bardziej opłacalne jest użycie większej liczby starszych/tańszych GPU. Przykładowo, zamiast 1x A100 można użyć 4x RTX 3090 (koszt porównywalny lub niższy), uzyskując sumarycznie więcej pamięci i FLOPS (oczywiście kosztem większego poboru mocy i trudniejszej konfiguracji multi-GPU). Dla małych firm inwestycja w kilka konsumenckich kart (RTX/Geforce) do eksperymentów może być rozsądna, a dopiero krytyczne treningi wykonywać na profesjonalnym sprzęcie w chmurze. W świecie cloud pojawiają się też alternatywni dostawcy z niższymi cenami (np. usługodawcy oferujący odsprzedaż niewykorzystanych zasobów data center). Uważne śledzenie rynku i korzystanie z promocji/resellerów może znacząco obniżyć rachunki. -  **Profilowanie zużycia energii i emisji:** Coraz częściej zespoły ML dodają do planowania projektów metrykę kosztu energetycznego. Wybierając między różnymi architekturami modelu, można uwzględnić nie tylko accuracy, ale też wymagany czas treningu. Model który osiąga 99% dokładności po tygodniu treningu może zostać odrzucony na rzecz takiego co ma 98% ale trenuje się w dobę – jeśli uzasadnimy, że oszczędzone zasoby są tego warte. Ponadto, jeżeli mamy dostęp do wielu centrów danych, wybór lokalizacji z zieloną energią (np. Skandynawia) dla treningu może zmniejszyć ślad węglowy.

Podsumowując, wydajność i koszty w AI to dwie strony medalu: **wydajność** (czas treningu, przepustowość) chcemy maksymalizować, a **koszty** (pieniądze, energia) minimalizować. Równoważenie tego to zadanie inżynierii. Dzięki mixowi rozwiązań sprzętowych (od tanich GPU po specjalistyczne ASIC) i programowych (optymalizacje algorytmiczne, techniki transfer learning), inżynierowie ML mają coraz większe pole do szukania optymalnego punktu. W 2025 r. duży nacisk kładzie się właśnie na **optymalizację**: nie jest sztuką wydać fortunę i zużyć gigawatogodziny – sztuką jest osiągnąć cel modelu przy rozsądnym budżecie. Firmy i zespoły, które to potrafią, zyskują przewagę zarówno ekonomiczną, jak i w zrównoważonym rozwoju AI.

# Zastosowania i konfiguracje: jak wybrać środowisko ML?\label{sec:zastosowania}
Wybór odpowiedniej platformy sprzętowej i środowiska uruchomieniowego dla projektu ML zależy od konkretnych potrzeb: typu zadania, skali modelu, budżetu, wymagań dotyczących prywatności czy mobilności. Omówmy kilka typowych scenariuszy i porównajmy strategie wyboru między opcjami takimi jak Apple Silicon, chmura obliczeniowa czy własny klaster GPU. Szczególną uwagę poświęcimy mocy obliczeniowej i pamięci Apple Silicon (192 GB unified memory) w zestawieniu z innymi rozwiązaniami.

**1. Pojedyncza stacja robocza vs. chmura – prototypowanie a skalowanie:** Na etapie **eksploracji i prototypowania** modelu (np. badacz tworzy nową architekturę sieci), kluczowa jest interaktywność i niska bariera eksperymentów. Tutaj często sprawdza się posiadanie lokalnej maszyny z GPU – np. desktop z kartą pokroju RTX 4090 (24 GB) lub laptop z GPU – co pozwala szybko uruchamiać modele średniej wielkości. Alternatywnie, osoby dysponujące **Macami z M1/M2** mogą korzystać z ich zintegrowanego GPU/Neural Engine. Dzięki frameworkom jak ML Compute czy MLX, te Maki potrafią trenować całkiem poważne modele. Przewagą Maca jest ogromna pamięć współdzielona – topowy Mac Studio z M2 Ultra ma 192 GB unified memory, co oznacza, że modele mieszczące się w takiej pamięci mogą być trenowane/inferowane bez potrzeby offloadowania danych na dysk. Przykładowo, model językowy 65 mld parametrów w pełnej precyzji (fp16) zajmuje ok. 120 GB – można go zmieścić cały w pamięci M2 Ultra, podczas gdy na typowej karcie graficznej 24 GB byłoby to niemożliwe (trzeba by rozdzielić na kilka GPU). Apple chwali się, że M2 Ultra jest w stanie trenować w pojedynczym systemie zadania, którym “nie podoła najmocniejsza dyskretna karta graficzna PC”​

[apple.com](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/# :~:text=features%20800GB%2Fs%20of%20system%20memory,3)

. Jest w tym ziarno prawdy: w scenariuszach mocno pamięciożernych (duże modele, ogromne batch size), żadna pojedyncza karta z 48 GB czy nawet 80 GB VRAM nie pomieści tego, co zmieści M2 Ultra z 192 GB. Przykładem może być przeprowadzenie pełnej inferencji dużego modelu jak Falcon 180B na Macu – społeczność pokazała, że jest to wykonalne z użyciem 4-bitowej kwantyzacji​

[youtube.com](https://www.youtube.com/watch?v=Zm1YodWOgyU# :~:text=Casually%20Run%20Falcon%20180B%20LLM,to%20run%20the%20model)

, podczas gdy na pojedynczej karcie PC to zadanie niemożliwe (wymaga co najmniej 2 x 48 GB z przeplotem pamięci).

Jednak **wydajność surowa** GPU Apple nie dorównuje najlepszym kartom Nvidii. GPU M2 Ultra (76 rdzeni) osiąga ok. 20 TFLOPS FP16, podczas gdy np. RTX 4090 ~330 TFLOPS FP16 (z tensor cores), a A100 ~312 TFLOPS. W testach praktycznych oznacza to, że trenowanie modelu, który mieści się zarówno na Macu, jak i na karcie Nvidii, zwykle będzie kilkukrotnie szybsze na karcie Nvidii. I tak np. fine-tuning średniego modelu (6–7 mld parametrów) może zająć na M2 Ultra powiedzmy 5 godzin, a na pojedynczym RTX 4090 – 2 godziny (wartości przykładowe). Wynika to z różnic architektury – GPU Apple jest zoptymalizowane pod niskie zużycie energii i współdziałanie z CPU, ale nie ma tak potężnych jednostek obliczeń tensorowych jak NVIDIA. Dlatego **GPU dyskretne nadal “rządzą” przy dużym zapotrzebowaniu na FLOPS**, podczas gdy Apple wygrywa, gdy **potrzeba dużo pamięci** (i wygody unified memory). W praktyce, ci którzy posiadają Mac Studio/Pro z M2 Ultra, mogą śmiało wykorzystywać go do wielu zadań ML – osiągną efektywność energetyczną i brak problemów z transferami danych, choć czas uczenia będzie dłuższy. Z kolei jeżeli priorytetem jest maksymalna szybkość trenowania modeli pasujących w 24–48 GB, lepszą inwestycją będzie PC z mocną kartą graficzną. Warto wspomnieć, że cena topowego Mac Studio (ok. $5000 za wersję 192 GB) jest zbliżona do kosztu stacji roboczej z RTX 4090 – wybór więc zależy od preferencji ekosystemu i potrzeb.

Kiedy prototyp staje się poważnym projektem wymagającym **skalowania**, często konieczne jest przejście do **chmury lub klastra**. Przykładowo, wytrenowanie finalnego modelu może wymagać setek godzin GPU – co na pojedynczej maszynie trwałoby miesiącami. W chmurze można rozproszyć to na np. 8 maszyn równolegle i skrócić czas 8-krotnie (kosztem równoległego wydatku). Decyzja zależy od budżetu i czasu: chmura umożliwia *“przyspieszenie za pieniądze”*. Jeśli deadline jest krytyczny, warto skalować w chmurze. Jeśli budżet ograniczony, można trenować dłużej na własnym sprzęcie.

**2. Wymagania dotyczące danych i prywatności:** W niektórych branżach (np. medycznej, finansowej) nie wolno wypuścić danych poza kontrolowane środowisko. To od razu kieruje wybór ku **rozwiązaniom on-premise**. Apple Silicon może być atrakcyjną opcją dla np. firmy medycznej prototypującej model na wrażliwych danych – badacz może bezpiecznie trenować sieć na Macu lokalnie, bez ryzyka wycieku do chmury. Gdy potrzeba więcej mocy, firma zainwestuje w własny serwer GPU w zabezpieczonym centrum danych. Cloud w takich wypadkach odpada z powodów regulacyjnych. Z drugiej strony, jeśli dane są publiczne lub generowane w chmurze (np. logi użytkowników serwisu SaaS), trzymanie przetwarzania blisko miejsca ich składowania (tj. w chmurze) może być wydajniejsze i bezpieczniejsze – unikamy transferów. Dlatego wybór środowiska powinien uwzględniać również aspekt **lokalizacji danych**.

**3. Apple Silicon (Mac) vs PC vs klaster – przykłady wydajności:** Rozważmy przykład trenowania modelu vision AI (powiedzmy sieć transformera do klasyfikacji obrazów) przy różnych konfiguracjach:

- **Mac Studio M2 Ultra (192 GB):** Pozwoli załadować cały zbiór danych do pamięci, model również, minimalizując dostęp do dysku. GPU Apple przyspieszy macierzowe operacje, Neural Engine może ewentualnie wspomóc przy niektórych warstwach. Taka maszyna z powodzeniem obsłuży batch size rzędu kilkuset obrazów 224x224 na epokę, choć czas jednej epoki będzie wolniejszy niż na topowym GPU PC. Zaletą jest prostota – wszystko działa w jednym procesie, bez rozproszenia. Energia zużyta będzie relatywnie niska (M2 Ultra jest bardzo energooszczędny względem swojej mocy). Wadą będzie dłuższy czas treningu całkowity.
- **Stacja z GPU NVIDIA (np. RTX 4090 24GB):** Ograniczona pamięcią do mniejszego batcha lub niższej rozdzielczości, ale za to każda iteracja będzie szybka. Jeśli dane są na szybkim dysku NVMe, a model nie przekracza pamięci, to taka karta może 2–4x przewyższyć prędkość M2 Ultra w czystym ML compute. W 24GB zmieści się sporo typowych modeli (do kilkuset milionów parametrów w FP16).
- **Mały klaster 2–4x GPU:** Tutaj można podzielić dane na dwie karty, co podwoi efektywny batch i skróci czas o połowę (z narzutem komunikacji). Przy 4 kartach RTX 3090 (każda 24 GB) łączna pamięć to 96 GB – wciąż połowa tego co Mac Studio – ale można trenować równolegle 4 fragmenty batcha. Taka konfiguracja prawdopodobnie pokona czasy zarówno pojedynczej karty, jak i Maca, ale wymaga już zarządzania treningiem rozproszonym (np. PyTorch DDP).

Z powyższego widać, że Apple Silicon lśni w sytuacji “wielka pamięć potrzebna, mniejsza moc wystarczy”, a klasyczne GPU – odwrotnie “mniej pamięci, ale maksymalna moc”. W kontekście np. **LLM i 192GB unified memory Apple** należy podkreślić: to rozwiązanie fantastyczne do *inference* wielkich modeli lub fine-tuningu metodami typu LoRA (gdzie nie trzeba gradientów dla wszystkich warstw). Apple podaje, że 192 GB umożliwia workflow “niemożliwy na PC”​

[apple.com](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/# :~:text=features%20800GB%2Fs%20of%20system%20memory,3)

– uściślając, chodzi o to, że na pojedynczym PC z kartą graficzną (nawet RTX 6000 Ada 48 GB) nie uruchomimy pewnych modeli w pełnej wersji, a na Macu tak. Jednak w scenariuszach treningu z gradientem dla całego modelu, zwykle i tak rozdziela się go na wiele GPU przy naprawdę dużych modelach – a wtedy przewaga unified memory maleje (bo korzystamy z sumarycznej pamięci wielu kart, np. 8×80 GB = 640 GB w klastrze).

**4. Inferencja (wdrażanie modelu):** Do tej pory skupialiśmy się na trenowaniu. W przypadku **wdrożenia modelu do produkcji**, wybór środowiska ma dodatkowe kryteria: koszt jednostkowego zapytania, opóźnienie (latency), skalowalność natychmiastowa. Jeśli budujemy np. serwis obsługujący zapytania do modelu językowego, chmura często wygrywa – pozwala dynamicznie skalować liczbę instancji zależnie od ruchu, płacić za faktyczne użycie i geograficznie rozproszyć serwery (bliżej użytkowników dla mniejszego latency). Z drugiej strony, w aplikacjach mobilnych/”na urządzeniu” preferuje się inferencję **on-device** ze względów prywatności i dostępności offline. Tutaj prym wiedzie Apple i platforma Core ML – modele zoptymalizowane do uruchamiania na Neural Engine i GPU iPhone’ów czy Maców. Taki model (np. asystent offline) może działać nawet bez internetu, bardzo szybko, ale jego rozmiar i złożoność muszą być dopasowane do możliwości urządzenia. Kontrastuje to z podejściem **cloud API** – np. wysyłania danych do OpenAI czy innego dostawcy, gdzie otrzymujemy wynik od potężnego modelu w chmurze. Wybór sprowadza się do kompromisu między *jakością i wielkością modelu* a *wymaganiami użytkowymi*. Często rozwiązaniem hybrydowym jest tzw. **edge-cloud split**: proste zadania realizuje model na urządzeniu, a trudniejsze przekazuje do chmury.

**5. Podsumowanie wyboru środowiska:**

- Dla **indywidualnego badacza czy małego startupu** – zaczynającego eksperymenty – dobrą opcją jest użyć posiadanego sprzętu (PC z GPU lub Mac z M2) do prototypów. To niski koszt początkowy. Gdy zajdzie potrzeba większej mocy na krótko – skorzystać z chmury (np. wynająć GPU na kilka godzin). Unikać od razu budowy drogiej infrastruktury zanim nie upewnimy się, czego dokładnie trzeba.
- Jeśli **projekt wymaga dużych modeli i mamy Maca M2 Ultra** – wykorzystać go maksymalnie, szczególnie do przetwarzania wstępnego i przetestowania pipeline’u z małym modelem. M2 Ultra może posłużyć jako poligon do dopracowania kodu, który potem przeniesiemy na klastry GPU dla finalnego treningu. Ewentualnie można rozważyć zakup Maca Pro z M2 Ultra, który umożliwia dołożenie akceleratorów PCIe (choć obecnie nie GPU – Apple ProRes, sieć itp.), co jednak nadal nie da możliwości instalacji kart Nvidii (brak sterowników dla macOS). Więc Mac Pro służy raczej do specyficznych zastosowań (np. obróbka video + AI jednocześnie).
- Jeśli **liczy się każdy dzień przewagi**, a budżet istnieje – chmura pozwoli zacząć trenować od razu na dużą skalę, zanim nawet zorganizowalibyśmy fizycznie serwery. To często praktyka startupów w wyścigu – użyć maksymalnie dużo instancji w AWS/Azure by szybciej osiągnąć wyniki (nawet jeśli to droższe).
- Gdy **projekt wchodzi w fazę produkcyjną** (stałe zapotrzebowanie na inferencję/trening), przeliczyć TCO (Total Cost of Ownership). Może się okazać, że sensowne jest kupno np. 4 serwerów z 8x GPU i postawienie ich w serwerowni – bo koszty chmury w ciągu roku przekroczyłyby cenę zakupu. Wtedy inwestycja się opłaci, pod warunkiem że zespół jest gotów zarządzać sprzętem.
- Dla **firm stawiających na Apple ecosystem** (np. dostarczających aplikacje na iPhone/Mac z AI) – posiadanie środowiska Apple do trenowania modeli zoptymalizowanych pod CoreML jest niemal koniecznością. Apple Silicon może znacznie skrócić pętlę developmentu takich modeli, bo umożliwia testowanie w warunkach zbliżonych do docelowych (uwzględniając ograniczenia NE). W tym przypadku strategia to: prototyp i trening na Macach, a ewentualnie finalne docieranie dużego modelu przełączyć na klaster GPU z odpowiednim konwerterem (np. trening w PyTorch + export do CoreML).

Kończąc, **192 GB unified memory vs inne rozwiązania**: Apple M2 Ultra imponuje przepustowością pamięci 800 GB/s i łatwością wykorzystania całej puli RAM​

[apple.com](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/# :~:text=features%20800GB%2Fs%20of%20system%20memory,3)

. Jednak pamięć to nie wszystko – przykładowo, akcelerator NVIDIA A100 ma “tylko” 40/80 GB, ale jego przepustowość HBM ~1555 GB/s i specjalizowane rdzenie tensor, dzięki czemu przy zadaniach mieszczących się w 40 GB będzie znacznie szybszy. Z drugiej strony, AMD MI300X także oferuje 192 GB (HBM3), łącząc to z ogromną mocą obliczeniową – ale MI300X to układ o TDP >500 W, przeznaczony do klastrów, podczas gdy M2 Ultra zamyka się w ~150 W. Tak więc Apple gra w innej lidze: efektywność energetyczna i integracja. Jeśli naszym ograniczeniem jest energia lub chcemy cichy, biurkowy system do eksperymentów z dużymi modelami – Mac Studio jest kuszący. Jeżeli jednak celem jest minimalny czas nauki modelu za wszelką cenę – zestaw GPU (czy to lokalny, czy w chmurze) będzie szybszy. Praktyka pokazuje, że wiele zespołów wykorzystuje Maci do **developingu i testów**, a **GPU do heavy lifting**. Na szczęście ekosystem narzędzi (Docker, kompatybilność Pythona itd.) pozwala dość płynnie przechodzić między tymi środowiskami.

\documentclass[11pt, oneside]{article}   	% use "amsart" instead of "article" for AMSLaTeX format
\usepackage{geometry}                		% See geometry.pdf to learn the layout options. There are lots.
\geometry{letterpaper}                   		% ... or a4paper or a5paper or ...
%\geometry{landscape}                		% Activate for rotated page geometry
%\usepackage[parfill]{parskip}    		% Activate to begin paragraphs with an empty line\usepackage{graphicx}				% Use pdf, png, jpg, or eps§ with pdflatex; use eps in DVI mode
								% TeX will automatically convert eps --> pdf in pdflatex

%\usepackage{{Gratzer-Color-Scheme}	% Active to color theorems red and lemmas blue

\usepackage{amssymb}

%SetFonts

%SetFonts

\title{Brief Article}
\author{The Author}
%\date{}							% Activate to display a given date or no date

\begin{document}
\maketitle
%\section{}
%\subsection{}

\end{document}  Technologie machine learning w 2025 r. oferują niespotykany dotąd wachlarz możliwości – od przyjaznych bibliotek, poprzez wyspecjalizowany sprzęt, aż po skalowalne usługi chmurowe. Kluczowym wyzwaniem dla zespołów AI jest dokonanie **świadomego wyboru** spośród tych opcji, tak aby osiągnąć cele projektu skutecznie i efektywnie. Poniżej zbieramy najważniejsze wnioski i zalecenia:

**1. Dobór frameworka i biblioteki:** W większości przypadków najlepiej postawić na rozwiązanie popularne i aktywnie rozwijane – to zapewnia wsparcie społeczności i kompatybilność z innymi narzędziami. PyTorch pozostaje bezpiecznym wyborem jako główny framework deep learning (szczególnie dla modeli neuronowych), z ogromnym ekosystemem (Hugging Face, PyTorch Lightning, FastAI itp.). TensorFlow jest alternatywą, jeśli planujemy głęboką integrację z Google Cloud, TPU lub potrzebujemy wydajnego deploymentu na Androidzie/JS (TF Lite, TF.js). JAX warto rozważyć w projektach badawczych nastawionych na maksymalną wydajność na TPU/GPU oraz eksperymenty z nowymi optymalizacjami – ale wymaga on bardziej doświadczonego zespołu. Nowe biblioteki, takie jak Apple MLX, mają sens głównie w specyficznych kontekstach (tutaj: ekosystem Apple). Natomiast przy pracy z dużymi modelami (szczególnie LLM) warto od początku uwzględnić narzędzia typu HuggingFace Transformers/PEFT, a do samego fine-tuningu wykorzystać dedykowane frameworki (Axolotl, Unsloth), które oszczędzą mnóstwo czasu i zasobów dzięki gotowym optymalizacjom​

[modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Axolotl%20is%20a%20wrapper%20for,tuning%20process)

​

[modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Unsloth%2C%20built%20by%20Daniel%20Han,Flash%20Attention%202)

. Ogólna zasada: **nie reinventuj koła**. Jeśli istnieje biblioteka implementująca nasz problem – lepiej jej użyć lub przynajmniej zaczerpnąć z niej pomysły, zamiast zaczynać od zera. Pozwoli to skupić się na unikalnych aspektach projektu zamiast rozwiązywaniu rozwiązanego.

**2. Wybór sprzętu i architektury:** Przy planowaniu zasobów sprzętowych należy przeanalizować potrzeby w cyklu życia projektu. Na **etapie R&D** (research and development) często wystarczy pojedyncza mocna karta GPU lub nawet akcelerator w laptopie – ważne, by iteracje były szybkie i dostępne od ręki. Tutaj świetnie spisują się np. laptopy gamingowe z GPU Nvidia, stacje robocze z kartami GeForce/RTX, czy MacBooki Pro z M-serii (zwłaszcza dla zadań, gdzie Apple Neural Engine przyspieszy np. modele CNN). Gdy przechodzimy do **trenowania pełnoskalowego modelu**, musimy ocenić wymaganą moc i pamięć. Jeśli model jest niewielki (kilkadziesiąt mln parametrów), nadal pojedyncza karta (np. RTX A6000 48GB) może wystarczyć. Dla modeli rzędu miliardów parametrów – praktycznie wymagany będzie **klaster multi-GPU** albo użycie pamięciooszczędnych technik (LoRA, postępowanie etapami itp.). W przypadku bardzo dużych modeli (dziesiątki mld+) nie obędzie się bez sprzętu klasy data center – A100/H100, TPU lub podobnych – co często sprowadza się do korzystania z chmury, chyba że jesteśmy instytucją dysponującą własnym superkomputerem. Wybierając między **Nvidia vs alternatywy**: jeśli zależy nam na minimalizacji ryzyka i maksymalnej kompatybilności, Nvidia jest wciąż numerem 1 (CUDA, cuDNN gwarantują, że większość kodu z Internetu “po prostu zadziała”). AMD GPU z ROCm mogą być opłacalne, ale wymagają trochę więcej uwagi (sprawdzenie czy nasz framework/testy wspierają ROCm). Specjalizowane układy (TPU, Habana) mogą dać oszczędność kosztów w chmurze przy pewnych obciążeniach – warto porównać oferty. Dla aplikacji edge – wybór jest najczęściej narzucony platformą (np. Qualcomm w Androidzie, Apple Neural Engine w iOS) i tu należy skorzystać z dostarczonych przez nich narzędzi optymalizacji (TensorFlow Lite, CoreML Tools). Ogólnie, **dobierzmy sprzęt do problemu**: nie ma sensu płacić za 8xH100, by trenować mały model, ale też nie ma sensu próbować zmieścić 100-miliardowego modelu na jednej karcie 16 GB – bo utknie to na miesiące. Zbalansujmy budżet vs wymagania, często rozwiązania pośrednie (np. 2–4 średnie GPU zamiast 1 ogromnego lub 8 topowych) są najbardziej uniwersalne.

**3. Chmura czy własny klaster:** Jeśli dopiero zaczynamy lub projekt jest krótkotrwały – **chmura obliczeniowa** daje niezrównaną elastyczność. Możemy wynająć 1 GPU na godzinę lub 100 GPU na tydzień, płacąc stosownie do użycia. Unikamy wydatków kapitałowych i zyskujemy czas (sprzęt jest dostępny od razu). Jednak w miarę jak projekt staje się stałym elementem działalności, rachunki za chmurę mogą przekroczyć koszt posiadania sprzętu. Zwykle granicą jest ciągłe wykorzystanie >50–70% w skali roku – wtedy lepiej mieć swój klaster (własny lub kolokowany). Rekomendacja: **zacznij w chmurze, monitoruj koszty**. Gdy zauważysz, że GPU są używane intensywnie non-stop, zrób kalkulację TCO na 1–3 lata i rozważ zakup. Można też stosować podejście mieszane: trzon stałych obliczeń trzymać on-prem, a peak’i obciążenia obsługiwać chmurą. Co do wyboru dostawcy cloud – wszystkie większe (AWS, GCP, Azure) mają porównywalny sprzęt. W praktyce często decydują czynniki poboczne: istniejące umowy, wygoda interfejsów, bliskość do innych usług (np. dane w BigQuery sugerują trenowanie w Google Cloud dla minimalizacji opóźnień). Nieco tańsze mogą być mniejsze firmy oferujące GPU (np. Lambda Labs, Paperspace, Vast.ai dla community). Jednak dla poważnych projektów enterprise często pozostaje wielka trójka ze względu na support i pewność dostępu do najnowszych układów.

**4. Optymalizacja i koszty energii:** Upewnij się, że **monitorujesz metryki** swojej infrastruktury – jak wykorzystanie GPU (czy czasem nie czekają one na dane z CPU?), wykorzystanie pamięci, czy sieć nie jest wąskim gardłem przy multi-GPU. Te dane pozwalają ulepszyć pipeline treningowy. Czasem drobna zmiana – np. zastosowanie mieszanej precyzji (FP16) – potrafi skrócić trening 2x bez pogorszenia wyników. Innym razem zmniejszenie częstotliwości logowania lub wyłączenie nadmiarowych statystyk usunie narzut na CPU. Wszystko to przekłada się na **oszczędność czasu i pieniędzy**. Pod kątem energii, jeśli mamy kontrolę nad lokalizacją – warto rozważyć zielone centra danych. Ale nawet bez tego, dobra optymalizacja to wyraz odpowiedzialności: model który osiąga ten sam wynik przy mniejszym zużyciu zasobów jest bardziej “eco-friendly”. W erze, gdy AI jest masowo wdrażane, zwracanie uwagi na efektywność będzie coraz ważniejsze – również wizerunkowo (klienci i regulatorzy mogą wymagać raportowania śladu węglowego AI).

**5. Trendy na horyzoncie:** Rok 2025 przynosi zapowiedzi nowych architektur zarówno sprzętu, jak i oprogramowania. NVIDIA planuje kolejną generację GPU (arch. *Blackwell*), AMD rozwija następców MI300, Google eksperymentuje z TPU v5. Pojawiają się też innowacje jak procesory neuromorficzne i fotoniczne do AI – choć to na razie prototypy. Ważne jest **trzymanie ręki na pulsie** – śledzenie konferencji (NeurIPS, HotChips) i blogów technologicznych pozwoli dowiedzieć się np. czy nowe biblioteki (jak PyTorch 2.0 z kompilatorem) mogą przyspieszyć nasz kod za darmo, albo czy pojawił się usługodawca z dużo tańszymi GPU. AI to niezwykle dynamiczna dziedzina i przewaga konkurencyjna często wynika z szybkiej adopcji nowych usprawnień. Oczywiście, nie należy bezkrytycznie przeskakiwać na każdą nowinkę – ale mieć świadomość opcji.

Podsumowując, **optymalny wybór technologii ML w 2025 r.** to zbalansowanie wielu czynników: łatwości rozwoju, wydajności obliczeń, kosztów operacyjnych, dostępności sprzętu i specyfiki zadania. Nie ma uniwersalnej odpowiedzi – inny będzie stack dla startupu robiącego aplikację mobilną z AI (tu: mały model, on-device, Apple/Android frameworks), a inny dla zespołu naukowego trenującego model 100B (tu: PyTorch+FSDP na klastrze A100, np. w chmurze AWS). Niemniej, rady które można dać każdemu: *wykorzystaj istniejące narzędzia, testuj w małej skali przed skalowaniem, monitoruj i usprawniaj wydajność, a decyzje sprzętowe podejmuj w oparciu o dane (profil zużycia, TCO) a nie hype.* W ten sposób można ujarzmić bogactwo technologii AI/ML i skierować je na realizację naszych celów, minimalizując przy tym koszty i ryzyka.

**Bibliografia:**

- sep0pt
-  Krizhevsky, A., Sutskever, I., Hinton, G. (2012). _ImageNet Classification with Deep Convolutional Neural Networks_. NeurIPS 2012. 

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=2012)

-  NVIDIA (2023). _Game-Changer: How the World’s First GPU Leveled Up Gaming and Ignited the AI Era_. NVIDIA Blog. 

[blogs.nvidia.com](https://blogs.nvidia.com/blog/first-gpu-gaming-ai/# :~:text=By%202011%2C%20AI%20researchers%20had,deep%20learning%E2%80%99s%20immense%20processing%20needs)

​

[blogs.nvidia.com](https://blogs.nvidia.com/blog/first-gpu-gaming-ai/# :~:text=In%202012%2C%20a%20breakthrough%20came,software%20written%20by%20vision%20experts)

-  TechTarget (2023). _The history of artificial intelligence: Complete AI timeline_. 

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=Google%20researchers%20developed%20the%20concept,LLMs)

​

[techtarget.com](https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline# :~:text=Intel%20claimed%20its%20FakeCatcher%20real,accurate)

-  Modal (2025). _Best frameworks for fine-tuning LLMs in 2025_. Modal Blog. 

[modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Takeaways)

​

[modal.com](https://modal.com/blog/fine-tuning-llms# :~:text=Unsloth%2C%20built%20by%20Daniel%20Han,Flash%20Attention%202)

-  Niklas Heidloff (2024). _Fine-tuning LLMs with Apple MLX locally_. Blog post. 

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

-  Tom’s Hardware (2025). _Huawei's Ascend 910C delivers 60% of Nvidia H100 inference performance_. 

[tomshardware.com](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance# :~:text=Huawei%27s%20HiSilicon%20Ascend%20910C%20is,China%27s%20reliance%20on%20Nvidia%20GPUs)

​

[tomshardware.com](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance# :~:text=While%20Huawei%20and%20SMIC%20have,Nvidia%20maintains%20its%20undisputable%20lead)

-  The Next Platform (2024). _Amazon Gives Anthropic $2.75 B So It Can Spend It On AWS GPUs_. 

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=In%20his%20opening%20keynote%20at,CPU%20HGX%20compute%20complex)

​

[nextplatform.com](https://www.nextplatform.com/2024/03/27/amazon-gives-anthropic-2-75-billion-so-it-can-spend-it-on-aws-gpus/# :~:text=On%20AWS%2C%20as%20you%20need,8%20million)

-  MobiDev (2025). _Using GPUs for AI & ML: On-Premises vs Cloud_. 

[mobidev.biz](https://mobidev.biz/blog/gpu-machine-learning-on-premises-vs-cloud# :~:text=Advantages%20Disadvantages%201%20Long,consuming%20and%20expensive)

​

[mobidev.biz](https://mobidev.biz/blog/gpu-machine-learning-on-premises-vs-cloud# :~:text=Advantages%20Disadvantages%201%20Rapid%20scalability,providers%20for%20uptime%20and%20availability)

-  Apple (2023). _Apple introduces M2 Ultra – Newsroom Release_. 

[apple.com](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/# :~:text=features%20800GB%2Fs%20of%20system%20memory,3)

-  Geekflare (2025). _JAX vs. PyTorch: Differences and Similarities_. 

[geekflare.com](https://geekflare.com/dev/jax-vs-pytorch/# :~:text=Performance%20Jax%20is%20incredibly%20fast,to%20follow%20and%20pick%20up)

#####
######
#####
######
# Analiza architektury Apple Silicon

Apple Silicon to rodzina układów SoC (System-on-a-Chip) zaprojektowanych przez Apple, łączących w jednym chipie wiele komponentów, takich jak CPU, GPU, NPU (Neural Engine) oraz zunifikowaną pamięć. Poniżej przedstawiono ewolucję architektury tych układów od M1 do najnowszego M4 Max, ze szczególnym uwzględnieniem aspektów istotnych dla zadań ML/AI.

## Ewolucja procesorów od M1 do M4 Max

**M1 (2020)** – pierwszy układ Apple dla komputerów Mac (5 nm) integrujący 8-rdzeniowe CPU (4 rdzenie wydajnościowe + 4 energooszczędne), do 8-rdzeniowego GPU oraz 16-rdzeniowy Neural Engine. M1 wprowadził zunifikowaną pamięć (Unified Memory) do 16 GB, co było przełomem w architekturze PC. Już M1 oferował wysoką wydajność przy niskim poborze mocy, jednak w zastosowaniach ML jego GPU odpowiadał mniej więcej możliwościom średniej klasy kart graficznych z tamtego okresu, a Neural Engine był wykorzystywany głównie do przyspieszania inferencji modeli na urządzeniu.

**M2 (2022)** – drugiej generacji układ (ulepszony 5 nm) przyniósł ok. 18% szybsze CPU, 35% mocniejszy GPU i 40% szybszy silnik neuronowy względem M1​

[apple.com](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/# :~:text=Apple%20silicon%20designed%20specifically%20for,inch%20MacBook%20Pro)

. M2 zawiera 20 mld tranzystorów (+25% vs M1) i zwiększa przepustowość pamięci o 50% (do 100 GB/s) oraz maksymalną pojemność Unified Memory do 24 GB​

[apple.com](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/# :~:text=Apple%20silicon%20designed%20specifically%20for,inch%20MacBook%20Pro)

​

[apple.com](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/# :~:text=The%20system,larger%20and%20more%20complex%20workloads)

. Poprawiono także dedykowane akceleratory (np. jednostkę do kodowania/dekodowania wideo ProRes). Dzięki większej pamięci i przepustowości M2 lepiej radzi sobie z większymi modelami ML i większymi batchami danych niż M1.

**M1 Pro / Max / Ultra** – rozszerzone warianty M1 dla komputerów MacBook Pro i Mac Studio. M1 Pro dodaje więcej rdzeni CPU/GPU (do 10 CPU, 16 GPU, 32 GB RAM), M1 Max jeszcze więcej (10 CPU, 32 GPU, 64 GB RAM, większa przepustowość pamięci ~400 GB/s), zaś M1 Ultra łączy dwa chipy M1 Max (20 CPU, 64 GPU, do 128 GB RAM) przez technologię UltraFusion, osiągając przepustowość pamięci ~800 GB/s. Te układy znacznie zwiększyły wydajność grafiki i umożliwiły pracę z bardziej złożonymi modelami (np. trenowanie sieci konwolucyjnych w rozsądnym czasie), choć wciąż ich surowa moc obliczeniowa była mniejsza niż topowych GPU NVIDIA (np. M1 Pro GPU okazał się ok. 13× wolniejszy od NVIDIA A6000 w treningu modelu ResNet-18 na CIFAR-10​

[lightly.ai](https://www.lightly.ai/post/apple-m1-and-m2-performance-for-training-ssl-models# :~:text=TL%3BDR)

). Niemniej jednak oferowały one bezprecedensową wydajność *per watt* – w praktyce zużywając ułamek energii potrzebnej konkurencyjnym układom (np. cały Mac Studio z M2 Ultra zużywa ok. 1/3 mocy samej karty RTX 4080​

[techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=the%20users%20in%20Apple%E2%80%99s%20ecosystem%2C,RTX%204080%20graphic%20cards%20alone)

​

[techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=%3E%20%20%20,)

).

**M3 (2023)** – trzecia generacja, wykonana w technologii 3 nm, przyniosła największy dotąd skok architektury GPU. Wprowadzono funkcję _Dynamic Caching_ – sprzętową, dynamiczną alokację lokalnej pamięci GPU tylko w wymaganym zakresie, co zwiększa średnie wykorzystanie GPU​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=The%20next,demanding%20pro%20apps%20and%20games)

. Dodano również sprzętowe wsparcie ray tracingu i mesh shading na Macach​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=With%20the%20M3%20family%20of,architecture%20enables%20all%20of%20these)

. GPU M3 jest do 2,5× szybsze od GPU z rodziny M1 w zastosowaniach graficznych​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=biggest%20leap%20forward%20in%20graphics,quality%20video%20experiences%20from)

, a jednocześnie potrafi osiągnąć tę samą wydajność co M1 przy niemal połowie poboru mocy​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=mesh%20shading%20to%20the%20Mac%2C,more%20performance%20at%20its%20peak)

. Rdzenie CPU w M3 uległy dalszemu usprawnieniu (+30% szybkości rdzeni wydajnościowych względem M1)​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=tracing%20and%20mesh%20shading%20to,Pro%20%20and%20%2017)

. 16-rdzeniowy Neural Engine został przyspieszony o 60% vs M1​

[apple.com](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/# :~:text=M3%2C%20M3%20Pro%2C%20and%20M3,Powerful%20AI%20image%20processing%20tools)

, co przekłada się na wyższe tempo realizacji operacji AI (np. przetwarzanie obrazów z użyciem sieci neuronowych, rozpoznawanie mowy) na urządzeniu. Ponadto maksymalna obsługiwana zunifikowana pamięć wzrosła (M3 Max oferuje do 128 GB) przy zwiększonej przepustowości. Układy M3 Pro/Max tradycyjnie rozszerzają konfigurację rdzeni (np. M3 Max: do 16 CPU, 40 GPU) i pamięci, kontynuując trend zwiększania zdolności do lokalnego przetwarzania dużych zbiorów danych i modeli.

**M4 (2024)** – najnowsza generacja (druga generacja procesu 3 nm) skupia się na dalszym zwiększaniu wydajności przy zachowaniu wysokiej efektywności energetycznej, ze szczególnym naciskiem na zadania AI. CPU M4 posiada najwydajniejsze rdzenie w historii (Apple chwali się „najszybszym rdzeniem CPU na świecie”) oraz znacznie wyższą wydajnością wielowątkową​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=capabilities%20to%20the%20Mac,to%202x%20faster%20than%20the)

. GPU w M4 Pro/Max opiera się na architekturze z M3, ale z szybszymi rdzeniami i dwukrotnie szybszym silnikiem ray tracingu​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=efficiency,for%20pro%20and%20AI%20workloads)

. Przepustowość zunifikowanej pamięci wzrosła aż o 75% – M4 Pro osiąga 273 GB/s (vs ok. 156 GB/s w M3 Pro)​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Pro%20supports%20up%20to,models%20run%20at%20blazing%20speed)

, a M4 Max aż 546 GB/s​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Max%20supports%20up%20to,M4%20Max%20includes%20two%20video)

. M4 Max obsługuje do 128 GB wspólnej pamięci​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Max%20supports%20up%20to,M4%20Max%20includes%20two%20video)

, co według Apple umożliwia deweloperom „łatwą pracę z modelami językowymi ~200 miliardów parametrów” lokalnie​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Max%20supports%20up%20to,M4%20Max%20includes%20two%20video)

(oczywiście przy zastosowaniu odpowiednich optymalizacji pamięci, jak niższa precyzja czy podział modelu). Neural Engine w M4 ma 16 rdzeni jak poprzednio, ale jest do 2× szybszy niż w M3​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=and%20a%202x%20faster%20ray,themselves%2C%20while%20protecting%20their%20privacy)

​

[apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=up%20to%2075%20percent,for%20Apple%20Intelligence%2C%20the%20personal)

, osiągając aż 38 TOPS (trylionów operacji na sekundę) w układzie M4 dla iPada​

[apple.com](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/# :~:text=while%20the%20new%2010,powerful%20device%20for%20artificial%20intelligence)

​

[apple.com](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/# :~:text=M4%20includes%20Apple%E2%80%99s%20most%20powerful,Neural%20Engine%20in%20A11%20Bionic)

. Co ważne, M4 integruje także nową technologię ARM **SME (Scalable Matrix Extension)** – dedykowane akceleratory macierzowe w rdzeniach CPU, obsługujące m.in. dane w formacie **bfloat16**, int8 itp.​

[blog.roboflow.com](https://blog.roboflow.com/putting-the-new-m4-macs-to-the-test/# :~:text=Under%20the%20Hood)

​

[blog.roboflow.com](https://blog.roboflow.com/putting-the-new-m4-macs-to-the-test/# :~:text=2,as%20many%20numbers%20in%20parallel)

. SME w M4 pozwala CPU efektywniej wykonywać obliczenia macierzowe (16×16 jednostek MAC na rdzeń, szersze 512-bitowe wektory)​

[blog.roboflow.com](https://blog.roboflow.com/putting-the-new-m4-macs-to-the-test/# :~:text=The%20M4%27s%20SME%20brings%20several,key%20improvements%20over%20previous%20generations)

, co przyspiesza operacje ML (np. mnożenia macierzy w sieciach) nawet na CPU. Podsumowując, M4 Pro/Max znacząco podnosi poprzeczkę wydajności AI on-device, próbując zniwelować dystans do wyspecjalizowanych akceleratorów, przy jednoczesnym zachowaniu znakomitej efektywności energetycznej układów Apple.

## Zunifikowana pamięć (Unified Memory) i jej wpływ na ML

Jedną z kluczowych cech architektury Apple Silicon (od M1 wzwyż) jest zunifikowana pamięć operacyjna, współdzielona przez wszystkie komponenty układu (CPU, GPU, Neural Engine, media engine, itp.). W tradycyjnych architekturach PC dedykowane GPU posiada oddzielną pamięć VRAM, co wymaga kosztownych transferów danych między pamięcią główną a pamięcią karty graficznej. Apple wyeliminowało ten podział – w układach M1/M2/M3/M4 istnieje pojedyncza pula *Unified Memory*, do której mają bezpośredni dostęp zarówno rdzenie CPU, jak i GPU czy NPU​

[github.com](https://github.com/ml-explore/mlx# :~:text=the%20CPU%20and%20the%20GPU%29)

​

[github.com](https://github.com/ml-explore/mlx# :~:text=,device%20types%20without%20transferring%20data)

.

Z perspektywy zadań ML daje to kilka istotnych korzyści:

- **Brak kopiowania danych między CPU a GPU** – dane (np. macierze wejściowe, parametry modelu) utworzone przez CPU są natychmiast widoczne dla GPU i odwrotnie. Eliminuje to narzut czasowy i pamięciowy związany z duplikowaniem tensora przy przenoszeniu obliczeń na akcelerator. Framework MLX w pełni to wykorzystuje – tablice (`mlx.array`) są zawsze alokowane we wspólnej pamięci, a operacje mogą być wykonywane na dowolnym urządzeniu bez ręcznego `.to(device)`​

    [github.com](https://github.com/ml-explore/mlx# :~:text=the%20CPU%20and%20the%20GPU%29)

    ​

    [ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=Concretely%2C%20when%20you%20make%20an,have%20to%20specify%20its%20location)

    . Dla porównania, w PyTorch typowy przepływ na GPU wymaga skopiowania tensora z RAM do pamięci GPU (`.to('cuda')`) przed wykonaniem obliczeń.

- **Lepsze wykorzystanie pamięci i większe modele** – mając jedną pulę pamięci, modele mogą zajmować praktycznie całość dostępnego RAM-u i będą w stanie korzystać z GPU, o ile GPU ma dostęp do tej pamięci (co w Apple Silicon jest standardem). W praktyce oznacza to możliwość uruchomienia lub załadowania modeli, które tradycyjnie przekraczałyby ograniczenia pamięci GPU. Np. M4 Max z 128 GB Unified Memory może potencjalnie załadować model językowy zbliżony rozmiarem do GPT-3 175B (przy zastosowaniu 8-bitowej kwantyzacji lub strumieniowania parametrów), czego pojedyncza karta graficzna z 24 GB VRAM by nie pomieściła. Oczywiście wydajność takiego rozwiązania może być ograniczona przepustowością pamięci, ale Apple zapewnia tu bardzo wysokie wartości (np. 546 GB/s w M4 Max​

    [apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Max%20supports%20up%20to,M4%20Max%20includes%20two%20video)

    , co jest porównywalne z przepustowością pamięci topowych GPU – RTX 4090 ma ok. 1008 GB/s, RTX 3090 ~936 GB/s​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/14319ra/rtx_40906000_vs_m2_max_with_96gb_unified_memory/# :~:text=RTX%204090%2F6000%20vs%20M2%20max,m2%20max)

    , lecz warto dodać, że architektura Apple inaczej korzysta z tej pamięci).

- **Mniejsze zużycie energii i zarządzanie termiką** – brak konieczności utrzymywania dwóch kopii danych (w pamięci systemowej i na GPU) oznacza, że ogólny pobór mocy może być niższy, a także zmniejsza się ryzyko przerzucania dużych bloków danych przez interfejs (np. PCIe), co dodatkowo oszczędza energię. W zastosowaniach mobilnych (MacBook na baterii, iPad) ma to kolosalne znaczenie – można trenować lub wykonywać inference modeli przez dłuższy czas bez drastycznego rozładowania baterii, czego klasyczny laptop z dedykowanym GPU nie oferuje.

Warto jednak zauważyć, że zunifikowana pamięć nie jest panaceum na wszystkie problemy. Fizycznie nadal jest to pamięć RAM dzielona między CPU i GPU – jeśli jedno z zadań zajmie jej większość, inne komponenty mogą odczuć brak zasobów. Należy dbać o optymalną gospodarkę pamięcią (o czym więcej w sekcji dot. najlepszych praktyk). Niemniej architektura pamięci współdzielonej znacząco upraszcza programowanie i zwiększa efektywność pipeline’ów ML na Apple Silicon. Przykładowo, można napisać kod, w którym dane są przygotowywane na CPU, a następnie ta sama tablica jest użyta na GPU, bez żadnego specjalnego kodu do transferu:

``` import mlx.core as mx

# Tworzenie losowych wektorów a i b (domyślnie w pamięci zunifikowanej)

a = mx.random.normal((100_000,)) b = mx.random.normal((100_000,))

# Wykonanie dodawania na CPU i GPU bez kopiowania danych

res_cpu = mx.add(a, b, stream=mx.cpu) # obliczenia na CPU res_gpu = mx.add(a, b, stream=mx.gpu) # obliczenia równoległe na GPU

print(res_cpu[:5], res_gpu[:5]) # obie operacje widzą te same dane w jednolitej pamięci ```

W powyższym kodzie oba urządzenia (CPU i GPU) wykonują operację dodawania na tych samych wektorach `a` i `b】. Dzięki zunifikowanej pamięci, \texttt{a` i `b` nie muszą być nigdzie przenoszone – zarówno CPU, jak i GPU mają do nich bezpośredni dostęp​

[ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=In%20MLX%2C%20rather%20than%20moving,For%20example)

. MLX pozwala wskazać parametr `stream` określający urządzenie wykonujące daną operację (CPU lub GPU); co więcej, jeśli operacje nie mają zależności, mogą być wykonywane równolegle​

[ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=mx,gpu)

. Mechanizm ten ułatwia wykorzystanie pełni zasobów układu Apple Silicon i jest automatycznie zarządzany przez scheduler MLX (w przypadku zależności między operacjami, MLX zadba o właściwą synchronizację 

[ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=c%20%3D%20mx,gpu)

).

## Wydajność CPU, GPU i Neural Engine w kontekście ML/AI

**CPU:** Rdzenie CPU w Apple Silicon charakteryzują się wysoką wydajnością jednowątkową oraz bogatym zestawem instrukcji (w tym wektorowych NEON, a w M4 – również macierzowych SME) przy bardzo dobrej efektywności energetycznej. W kontekście ML CPU sprawdza się w przygotowaniu danych, obsłudze logiki programowej i wykonywaniu tych części obliczeń, które trudno zrównoleglić lub są zbyt małe, by efektywnie obciążyć GPU. Dzięki wysokiemu IPC i taktowaniu, rdzenie wydajnościowe mogą szybko wykonywać np. wstępne przetwarzanie danych (augmentacja obrazów, ekstrakcja cech) czy też symulować operacje niedostępne na akceleratorach. Ponadto, pojawienie się instrukcji **AMX/SME** (od M3/M4) dodaje CPU zdolność przyspieszania operacji typowych dla sieci neuronowych (mnożenie macierzy w niskiej precyzji). Niemniej, do trenowania dużych sieci neuronowych wykorzystuje się głównie GPU, gdyż oferuje ono znacznie większą przepustowość obliczeń równoległych.

**GPU:** Układy graficzne w Apple Silicon są wysoko zintegrowane i specjalnie dostrojone do pracy z wspólną pamięcią. Choć nie dorównują największym układom GPU od Nvidii czy AMD pod względem surowej mocy (np. M3 Max GPU ~14 TFLOPS FP32 vs >80 TFLOPS w RTX 4090​

[youtube.com](https://www.youtube.com/watch?v=2o42gp3VuCk# :~:text=Top%20of%20the%20TFLOPS%20,M2%20Ultra%20%3D%2027%20TFLOPS)

), to potrafią bardzo efektywnie wykorzystać posiadane zasoby dzięki architekturze zbliżonej do tile-based rendering (małe bloki pamięci podręcznej) i wspomnianemu dynamicznemu zarządzaniu pamięcią GPU od M3. W zastosowaniach ML, GPU Apple osiągają znaczne przyspieszenie względem CPU dzięki kilkuset rdzeniom wykonującym operacje wektorowe/macierze równolegle. Przykładowo, na M1 Pro 14-rdzeniowy GPU potrafił trenować model ResNet-18 ok. 8,8× szybciej niż 8-rdzeniowe CPU tego samego układu​

[lightly.ai](https://www.lightly.ai/post/apple-m1-and-m2-performance-for-training-ssl-models# :~:text=TL%3BDR)

. Wraz z kolejnymi generacjami różnica ta rosła – np. GPU w M3 Max (40 rdzeni) wykonał trening pewnego modelu transformera ponad 2× szybciej niż M1 Max (32 rdzenie)​

[github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=M3%20Max%20,51)

. Apple stale zwiększa też możliwości GPU w precyzjach poniżej FP32. Wspierane jest **FP16** (half-precision) – układy GPU M1/M2/M3 mogą wykonywać operacje 16-bitowe szybciej niż 32-bitowe, co często podwaja efektywną szybkość trenowania sieci (o ile model i dane pozwalają na redukcję precyzji). Od generacji M2 dostępna jest również **bfloat16** – format o 16-bitowej mantysie, lecz szerszym zakresie dynamiki (8-bitowy wykładnik jak w FP32). Bfloat16 ułatwia trenowanie głębokich sieci, redukując problemy numeryczne (np. z niedoszacowaniem gradientów) bez potrzeby skalowania, i jest wspierany natywnie przez Metal i biblioteki ML na Apple Silicon​

[news.ycombinator.com](https://news.ycombinator.com/item?id=36575443# :~:text=,to%20no%20need%20for%20scaling)

​

[developer.apple.com](https://developer.apple.com/videos/play/wwdc2024/10160/# :~:text=One%20of%20the%20updates%20this,cases%20like%20mixed%20precision%20training)

. Warto wspomnieć, że GPU Apple nie obsługują natywnego FP64 (double precision) – operacje `float64` mogą być wykonywane tylko na CPU (próba użycia `float64` na GPU MLX wywoła wyjątek​

[ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/python/data_types.html# :~:text=Note)

). Nie jest to dużym ograniczeniem w ML, gdzie standardem jest 32-bit lub niższa precyzja.

**Neural Engine (NPU):** Każdy układ Apple M1/M2/M3/M4 zawiera 16-rdzeniowy Neural Engine – dedykowany akcelerator uczenia maszynowego, pierwotnie zaprojektowany na potrzeby zadań AI w iPhone (od A11 Bionic). Neural Engine (ANE) osiąga imponujące wartości operacji na sekundę przy minimalnym zużyciu energii (np. 15,8 TOPS w M2​

[apple.com](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/# :~:text=Both%20M2%20Pro%20and%20M2,and%20up%20to%2040)

, 22 TOPS w M3, 38 TOPS w M4​

[apple.com](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/# :~:text=while%20the%20new%2010,powerful%20device%20for%20artificial%20intelligence)

). Jest jednak zoptymalizowany głównie pod **inference** (wnioskowanie) – doskonale sprawdza się w przyspieszaniu sieci neuronowych w aplikacjach (rozpoznawanie twarzy, mowy, analiza obrazu w aparacie itp.), ale **nie jest bezpośrednio wykorzystywany do trenowania modeli w otwartych frameworkach**. W ekosystemie Apple, ANE jest dostępny poprzez framework *Core ML*, gdzie wytrenowany model można przekształcić do formatu _.mlmodel_i wykonać na ANE. Obecnie jednak narzędzia badawcze (PyTorch, TensorFlow, MLX) nie korzystają z NPU podczas treningu – operacje treningowe realizowane są na CPU/GPU, a Neural Engine może być użyty ewentualnie do przyspieszenia części zadań inferencyjnych. Dodatkowo, programowanie ANE niskopoziomowo jest utrudnione – Apple nie udostępnia publicznego API do swobodnego wykonywania dowolnych operacji na ANE poza Core ML. Dlatego w dyskusjach społeczności często podkreśla się, że *“Apple’s neural accelerators are designed for inference. You won’t benefit from them when training models”*​

[reddit.com](https://www.reddit.com/r/MachineLearning/comments/1gf46km/d_m4_chips_for_training_ml_mps/# :~:text=%E2%80%A2)

. Podsumowując: w kontekście ML/AI na Apple Silicon, Neural Engine jest świetnym uzupełnieniem do szybkiego uruchamiania gotowych modeli na urządzeniu (szczególnie na iPhone/iPad, gdzie GPU jest słabsze), ale w treningu modeli na Macu główną rolę pełni GPU.

**Współpraca komponentów:** Należy zaznaczyć, że Apple Silicon został zaprojektowany z myślą o ścisłej współpracy wszystkich części. CPU może offloadować masywne równoległe obliczenia na GPU lub NPU, GPU może działać na danych przygotowanych przez CPU dzięki zunifikowanej pamięci, a dedykowane akceleratory (Neural Engine, Media Engine) odciążają GPU z niektórych zadań (np. inferencja wytrenowanego modelu wizji może iść na NPU, a dekodowanie strumieni obrazów – na Media Engine, podczas gdy GPU zajmuje się trenowaniem innego modelu). Takie rozproszenie zadań jest wspierane sprzętowo i programowo. MLX umożliwia np. jednoczesne wykorzystanie CPU i GPU (jak pokazano wyżej), a wytrenowany model można przekonwertować do Core ML, aby wykonać go na Neural Engine równolegle z innymi zadaniami. Ta heterogeniczność powoduje, że Apple Silicon może efektywnie realizować całe pipeline’y ML end-to-end na jednym chipie – od wczytania i przetworzenia danych, poprzez trening, po wdrożenie i inferencję – co wyróżnia go na tle klasycznych platform wymagających osobnych urządzeń do treningu (GPU) i wdrożenia (CPU lub specjalny akcelerator).

## Integracja MLX-ML z architekturą Apple Silicon

MLX został zaprojektowany od podstaw z myślą o pełnym wykorzystaniu specyfiki Apple Silicon. Integracja MLX z hardware przejawia się na kilku poziomach:

- **Wykorzystanie zunifikowanej pamięci:** Jak już omówiono, MLX traktuje pamięć jako jedną przestrzeń dostępną dla różnych urządzeń. Struktury danych (np. `MLXArray`) “żyją” w pamięci współdzielonej, co pozwala dowolnie wykonywać operacje na CPU lub GPU bez jawnego przenoszenia danych​

    [github.com](https://github.com/ml-explore/mlx# :~:text=,device%20types%20without%20transferring%20data)

    ​

    [ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=In%20MLX%2C%20rather%20than%20moving,For%20example)

    . To podejście odróżnia MLX od większości konkurencyjnych frameworków (TensorFlow, PyTorch, JAX), które na innych platformach muszą explicite zarządzać kopiami danych między hostem a urządzeniem​

    [promptengineering.org](https://promptengineering.org/apple-releases-mlx-a-new-framework-for-machine-learning-on-apple-silicon/# :~:text=MLX%20can%20target%20CPU%2C%20GPU%2C,No%20copies%20needed)

    ​

    [promptengineering.org](https://promptengineering.org/apple-releases-mlx-a-new-framework-for-machine-learning-on-apple-silicon/# :~:text=Some%20early%20adopters%20have%20already,may%20become%20even%20more%20significant)

    . MLX automatycznie zarządza zależnościami między operacjami na różnych urządzeniach, dzięki czemu deweloper może skupić się na logice modelu, a nie na logistyce danych.

- **Wsparcie akceleracji GPU (Metal):** Operacje tensoryczne w MLX są realizowane z wykorzystaniem biblioteki Metal i jej shaderów obliczeniowych. Apple udostępniło w MLX także możliwość tworzenia własnych jąder obliczeniowych w Metal Shading Language, by użytkownicy mogli dopisać operacje niestandardowe optymalizowane na GPU​

    [ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=Further%20Reading)

    . Domyślnie jednak MLX zapewnia bogaty zestaw operatorów pokrywających typowe potrzeby (algebra macierzowa, operacje tablicowe, warstwy neuronowe itp.), napisanych i zoptymalizowanych pod Apple GPU. W praktyce wydajność tych operacji jest zbliżona do analogicznych implementacji w frameworkach wykorzystujących Metal Performance Shaders (np. PyTorch MPS), a często lepsza dzięki brakowi narzutu transferów i pewnym optymalizacjom (o czym w sekcji z benchmarkami).

- **Wykorzystanie wielu języków i API:** MLX integruje się z ekosystemem Apple także przez udostępnienie interfejsów w językach C++ i Swift obok Pythona​

    [github.com](https://github.com/ml-explore/mlx# :~:text=,simplify%20building%20more%20complex%20models)

    . C++ pozwala na bezpośrednią integrację z istniejącymi projektami native (np. aplikacjami macOS czy iOS), zaś Swift – kluczowy język Apple – umożliwia pisanie kodu ML bezpośrednio w ramach projektów Swift/Xcode. Wydano nawet specjalną paczkę *MLX Swift*, która pozwala używać MLX z pełnym komfortem typowego dla Swifta (wraz ze wsparciem dla autodiff w Swift)​

    [swift.org](https://swift.org/blog/mlx-swift/# :~:text=MLX%20is%20an%20array%20framework,deployment%20of%20models%20in%20apps)

    ​

    [swift.org](https://swift.org/blog/mlx-swift/# :~:text=silicon,deployment%20of%20models%20in%20apps)

    . Dzięki temu badacze mogą prototypować modele w Pythonie, a docelowo zintegrować kod w aplikacji Swift, korzystając z tego samego “silnika” MLX pod spodem. Jest to ważny element ekosystemu Apple – płynne przechodzenie od badań do produktu.

- **Brak bezpośredniego wsparcia Neural Engine:** Jak wspomniano, MLX na razie nie wykonuje obliczeń na Neural Engine. Integracja z NPU odbywa się jedynie poprzez eksport modelu do Core ML. Oznacza to, że trening modeli w MLX będzie używał CPU i GPU, natomiast gotowy model można wyeksportować i załadować do `CoreML.framework`, który przy odpowiednich ustawieniach skompiluje model na ANE. Ten przepływ (train on GPU $\to$ deploy on ANE) jest standardowym zaleceniem Apple dla deweloperów aplikacji wykorzystujących własne modele. W ramach niniejszego przewodnika koncentrujemy się jednak na fazie treningu i eksperymentów z MLX.

- **Metal Performance Shaders (MPS) i Core ML:** Choć MLX jest niezależnym frameworkiem, pod spodem komplementarnie korzysta z tych samych technologii co inne rozwiązania Apple. Wspomniane MPS to biblioteka niskopoziomowa oferująca gotowe funkcje obliczeniowe (np. sploty, mnożenia macierzy, aktywacje) na GPU – PyTorch czy TensorFlow na Macu używają MPS jako backendu. MLX poszedł o krok dalej, dostarczając własne implementacje i narzędzia, ale nadal może wykorzystywać niektóre zasoby Metal/MPS. Z kolei **Core ML** to framework do inference na urządzeniach Apple; chociaż MLX nie generuje bezpośrednio modeli Core ML, to docelowo może powstać narzędzie konwertujące wytrenowany model MLX (np. zapisany jako `.pt` kompatybilny z PyTorch lub w onnx) do formatu .mlmodel. Apple wspiera konwersje z popularnych bibliotek (TensorFlow, PyTorch) poprzez `coremltools`, więc integracja wytrenowanych modeli MLX z Core ML jest jak najbardziej możliwa (wymaga to jednak kroku pośredniego – np. zapisania wag i odtworzenia modelu w PyTorch lub ONNX, a następnie konwersji). Najważniejsze jednak, że dzięki wspólnym fundamentom (Metal), wyniki wydajnościowe modeli MLX można przenieść na Core ML – np. jeśli model działa wydajnie na GPU w MLX, to po konwersji będzie mógł też skorzystać z GPU lub ANE w ramach Core ML.

Podsumowując, MLX jest ściśle dopasowany do sprzętu Apple. Wykorzystuje jego mocne strony (wspólna pamięć, wydajne GPU, różne języki Apple), omija ograniczenia (np. brak double na GPU, brak bezpośredniego dostępu do ANE) i wpisuje się w filozofię integracji hardware-software, z której słynie Apple. Dla programisty oznacza to, że pisząc modele w MLX, korzysta on z pełni możliwości chipu M1/M2/M3/M4 bez konieczności ręcznego dostrajania kodu do architektury – framework robi to za niego.

# Benchmarki i porównania wydajności

W tej sekcji przyjrzymy się, jak MLX-ML spisuje się w praktyce na Apple Silicon w porównaniu z innymi platformami i rozwiązaniami. Omówimy wyniki testów wydajności dla różnych typów zadań (NLP, wizja komputerowa, modele generatywne), a także porównamy osiągi Apple Silicon z popularnymi akceleratorami ML, takimi jak GPU Nvidii/AMD czy TPU od Google. Szczególny nacisk położymy na czas treningu/inferencji, koszt obliczeniowy oraz efektywność energetyczną.

## Wydajność MLX-ML na zadaniach NLP, CV i generatywnych

Aby ocenić możliwości MLX na Apple Silicon, posłużono się kilkoma typowymi zadaniami ze sztucznej inteligencji:

- **Trenowanie modelu NLP (transformer LM):** Jednym z przykładów jest trenowanie prostego modelu językowego (transformera) na korpusie tekstowym (np. Penn Treebank). Według dostępnych benchmarków, na MacBooku Pro z M1 Pro (10 CPU, 16 GPU) trening modelu transformera trwał ok. 1807 s w PyTorch (z użyciem MPS), podczas gdy w MLX zajął ~1157 s. To o ~36% krócej na korzyść MLX. Na mocniejszym M1 Max (32 GPU) różnica była jeszcze większa: 1106 s (PyTorch) vs 752 s (MLX)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=M1%20Max%20,GPU%20core%2C%2048%20GB%20RAM)

    , czyli MLX był ~1,47× szybszy. Dla najnowszego M3 Max (40 GPU) odnotowano 913 s (PyTorch) vs 426 s (MLX)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=M3%20Max%20,51)

    – MLX ponad **2×** przyspieszył trening transformera względem PyTorch na tej samej maszynie! To imponujący wynik, pokazujący jak bardzo optymalizacje MLX (lazy execution, lepsze wykorzystanie GPU i unified memory) przekładają się na praktyczną wydajność.

- **Trenowanie/finetuning modelu klasyfikacji (BERT Tiny):** W innym teście porównano fine-tuning małego modelu BERT (wersja Tiny) do zadania klasyfikacji zdań. Na M1 Max trening BERT Tiny zbiegał w 794 s (PyTorch) vs 499 s (MLX)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%20a%20transformer%20language%20model,GPU%20core%2C%2048%20GB%20RAM)

    – MLX był ok. 1,6× szybszy. Na M3 Max analogicznie: 550 s vs 408 s​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%20a%20transformer%20language%20model,51)

    . Oznacza to, że MLX radzi sobie lepiej nawet z modelami sekwencyjnymi, gdzie sporo operacji to pamięciochłonne przetwarzanie tekstu – tu zapewne korzystnie działa m.in. efektywna obsługa danych w unified memory (dane wejściowe tokenizowane przez PyTorch Dataloader były przekonwertowane do MLX array bez kopiowania, co przyspieszyło pipeline)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%2Ffine)

    ​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=pure%20training,training%20was%20the%20NLI%20dataset)

    .

- **Inferencja modelu ASR (Whisper):** Dla modelu rozpoznawania mowy Whisper (OpenAI) sprawdzono czas przetwarzania próbek audio. Na M1 Pro inferencja zajmowała ~32 s w PyTorch vs 8,5 s w MLX – MLX był **3,8×** szybszy. Na M3 Max: 17,9 s (PT) vs 4,85 s (MLX)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%20a%20transformer%20language%20model,51)

    , czyli również ~3,7× szybciej. To ogromna różnica. Wytłumaczeniem może być fakt, że MLX posiada własną implementację kerneli do modelu Whisper, które lepiej wykorzystują GPU (np. specjalizowane operacje dla warstw Transformerowych), podczas gdy PyTorch MPS miewał ograniczenia w wydajności niektórych operacji macierzowych na GPU Apple. Tak czy inaczej, MLX pozwala na znacznie sprawniejsze uruchamianie modeli generatywnych do mowy – czas analizy kilkunastosekundowego nagrania skrócono z pół minuty do kilku sekund.

- **Inferencja modelu LLM (TinyLlama):** Sprawdzono też czas generowania tekstu przez mały model LLaMA (TinyLlama). Na M1 Max: 51 s (PT) vs 20,6 s (MLX)​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%20a%20transformer%20language%20model,GPU%20core%2C%2048%20GB%20RAM)

    (~2,5× szybciej). Na M3 Max: 36,2 s vs 15,4 s​

    [github.com](https://github.com/LucasSte/MLX-vs-Pytorch# :~:text=Training%20a%20transformer%20language%20model,51)

    (również ~2,35× szybciej). Ponownie MLX wykazuje przewagę, co jest zachęcające dla zastosowań typu chatbot/LLM on-device – interakcja z modelem może być dużo płynniejsza. Warto dodać, że Apple intensywnie pracuje nad optymalizacją dużych modeli językowych na swoich układach, czemu służą zarówno ulepszenia sprzętowe (M4 z ogromną przepustowością pamięci) jak i programowe (frameworki jak MLX). Już teraz „mniejsze” modele (6B, 7B, 13B parametrów) można na Apple Silicon uruchamiać z przyzwoitą prędkością, a fine-tuning takich modeli (np. metodą LoRA) jest wykonalny lokalnie​

    [promptengineering.org](https://promptengineering.org/apple-releases-mlx-a-new-framework-for-machine-learning-on-apple-silicon/# :~:text=As%20engineer%20Ani%20Hanon%20tweeted%2C,enabled%20by%20the%20new%20framework)

    ​

    [promptengineering.org](https://promptengineering.org/apple-releases-mlx-a-new-framework-for-machine-learning-on-apple-silicon/# :~:text=Some%20early%20adopters%20have%20already,may%20become%20even%20more%20significant)

    .

- **Generowanie obrazów (Stable Diffusion):** Chociaż powyższe benchmarki nie objęły Stable Diffusion, MLX posiada przykład generowania obrazów tą metodą​

    [github.com](https://github.com/ml-explore/mlx# :~:text=,Speech%20recognition%20with%20OpenAI%27s%20Whisper)

    . Doniesienia użytkowników wskazują, że Stable Diffusion działa na Apple Silicon bardzo sprawnie – np. na M2 Max (38 GPU) generacja obrazu 512×512 trwa ok. 5–7 s na iterację przy 50 krokach samplingu (dla porównania, na RTX 3060 zajmowała ~4–5 s w podobnych warunkach). MLX powinien osiągać zbliżone czasy co PyTorch mps, być może minimalnie szybciej dzięki lepszemu zarządzaniu pamięcią. Co ważne, unified memory pozwala na wygodne generowanie obrazów o wyższej rozdzielczości – modele mogą użyć więcej pamięci, jeśli jest dostępna, bez ograniczeń typowych dla VRAM. Dla aplikacji generatywnych (obrazy, wideo) Apple Silicon jest coraz bardziej konkurencyjny, zwłaszcza w kontekście pracy mobilnej (np. artysta generujący grafiki na MacBooku bez dostępu do stacjonarnego GPU).

Ogólnie z tych testów wyłania się obraz, że **MLX na Apple Silicon potrafi przewyższyć tradycyjne frameworki o 30–100% w zadaniach ML** dzięki głębokiej optymalizacji pod sprzęt​

[promptengineering.org](https://promptengineering.org/apple-releases-mlx-a-new-framework-for-machine-learning-on-apple-silicon/# :~:text=Some%20early%20adopters%20have%20already,may%20become%20even%20more%20significant)

. Szczególnie duże zyski widać w zadaniach I/O intensywnych lub częstych przełączaniach CPU-GPU (gdzie unified memory eliminuje wąskie gardła) oraz przy modelach sekwencyjnych (gdzie lazy execution i optymalizacje grafu w MLX zmniejszają narzut Pythonowej pętli treningowej). Dla użytkownika oznacza to krótszy czas oczekiwania na wyniki eksperymentów – a więc większą produktywność w badaniach ML prowadzonych na Macu.

## Porównanie Apple Silicon + MLX do GPU Nvidia/AMD oraz TPU

Choć Apple Silicon staje się coraz mocniejszym graczem, nadal warto porównać jego osiągi do dedykowanych akceleratorów używanych powszechnie w ML:

- **Porównanie z GPU Nvidia:** Najnowsze układy jak M3 Max czy M4 Max oferują dużą część możliwości, jakie daje dyskretna karta graficzna. Jednak topowe GPU Nvidii (A100, H100, RTX 4090 itp.) wciąż znacznie przewyższają Apple Silicon pod względem surowej wydajności obliczeń i przepustowości pamięci. Na przykład, M2 Ultra (76-core GPU ~27 TFLOPS FP32) zestawiony z RTX 4090 (~82 TFLOPS FP32) osiąga około 60–70% jego wydajności w najlepszym wypadku, mimo że pobiera znacznie mniej mocy​

    [techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=M2%20Ultra%20vs%20Nvidia%20RTX,faster%20than)

    . W praktyce w dużych modelach różnica może być większa – wspomniany test trenowania modelu ResNet na M1 Pro pokazał ~14× dłuższy czas niż na GPU Nvidia A6000​

    [lightly.ai](https://www.lightly.ai/post/apple-m1-and-m2-performance-for-training-ssl-models# :~:text=TL%3BDR)

    . Co prawda od tamtej pory Apple zrobiło duży postęp (M3 Max jest ~4× szybszy od M1 Pro w podobnych zadaniach), ale wciąż do topowych GPU brakuje. Nawet marketing Apple przyznaje, że 546 GB/s w M4 Max to co prawda 4× więcej niż „najlepszy AI PC chip” (tu zapewne chodzi o konkurencyjny procesor PC z akceleratorem AI)​

    [apple.com](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/# :~:text=M4%20Max%20supports%20up%20to,M4%20Max%20includes%20two%20video)

    , ale nadal połowa tego co oferuje np. RTX 4090 (1008 GB/s). Głównym atutem Apple jest **efektywność energetyczna** – wspomniane ~27 TFLOPS M2 Ultra uzyskuje przy kilkudziesięciu watach, podczas gdy RTX 4090 aby osiągnąć 82 TFLOPS zużywa do 450 W. Oznacza to, że per wat Apple może być nawet bardziej wydajne (co potwierdza obserwacja, że Mac Studio M2 Ultra zużywa 1/3 energii potrzebnej samej karcie RTX 4080​

    [techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=%3E%20%20%20,)

    ). Dla długotrwałych obliczeń (np. wielodniowe treningi) przekłada się to na niższy koszt prądu i mniejsze wydzielanie ciepła. Jednak w kategoriach czystego czasu wykonania zadań, zwłaszcza tych największych (trenowanie sieci z setkami milionów/miliardami parametrów), dedykowane stacje z GPU Nvidia nadal są szybsze.

- **Porównanie z GPU AMD:** AMD dostarcza zarówno konsumenckie GPU (Radeon) jak i akceleratory MI dla centrów danych. Pod kątem ML AMD ustępuje Nvidii jeśli chodzi o ekosystem (sterowniki, optymalizacje bibliotek), ale sprzętowo oferuje podobne poziomy mocy obliczeniowej. Apple Silicon w porównaniu do najnowszych Radeonów (np. serii 7000) również przegrywa w maksymalnej wydajności – np. Radeon 7900 XTX osiąga ok. 40 TFLOPS FP32 przy 355 W, a więc jest kilkukrotnie szybszy niż M3/M4 Max w operacjach FP32. Jednak tu ponownie wchodzi kwestia efektywności: GPU AMD w laptopach (np. 680M iGPU lub mobilne Radeony) są słabsze niż Apple M3/M4 GPU przy porównywalnym budżecie mocy. Dodatkowo, Apple oferuje unikatową zaletę w postaci dużej wspólnej pamięci – typowy Radeon ma 8–16 GB VRAM, podczas gdy M4 Max 128 GB. W zastosowaniach, gdzie model nie mieści się w VRAM karty AMD, Mac z M4 Max może go jednocześnie załadować do pamięci i trenować (choć powoli). AMD pracuje nad technologiami zunifikowanej pamięci (Smart Access Memory), ale ze względu na architekturę PC (oddzielne moduły RAM i VRAM) nie jest to równie płynne co w Apple Silicon. Podsumowując, przeciwko AMD Apple Silicon wypada podobnie jak przeciw Nvidii – w tej chwili ustępuje topowym układom pod względem maksymalnej wydajności, ale ma przewagi w efektywności i zarządzaniu pamięcią.

- **Porównanie z TPU (Google):** TPU (Tensor Processing Unit) to wyspecjalizowane układy ML używane głównie w chmurze Google (np. do trenowania modeli w TensorFlow i JAX). TPU v4 oferuje ogromną moc – szczytowo setki TFLOPS w półprecyzji na pojedynczym urządzeniu, a zwykle używa się ich w podłączonych klastrach (TPU Pod) dla uzyskania jeszcze większej mocy. Apple Silicon nie próbuje nawet konkurować z TPU w skali data center – M-chipy są projektowane do działania w pojedynczych urządzeniach. TPU z racji specjalizacji (tylko macierze) są znacznie mniej uniwersalne, ale do masowego treningu np. modeli językowych rzędu dziesiątek miliardów parametrów nie mają konkurencji. Przykładowo, model PaLM 540B trenowano na podzie 1024 TPUv4 – coś takiego pozostaje daleko poza zasięgiem jakiejkolwiek stacji roboczej. Można jednak porównać sytuację developerską: Mac z Apple Silicon + MLX umożliwia deweloperowi prototypowanie i testowanie pomysłów lokalnie na stosunkowo dużych modelach (do kilkunastu miliardów parametrów z pewnymi ograniczeniami) bez konieczności dostępu do chmury, podczas gdy do realnego treningu SOTA modelu i tak niezbędne będzie skorzystanie z GPU/TPU w chmurze. Warto tu wspomnieć o kosztach: korzystanie z TPU (np. w Google Cloud) generuje znaczne koszty finansowe, podczas gdy trenowanie mniejszych modeli na Macu jest praktycznie “za darmo” po zakupie sprzętu. Dla wielu zespołów R&D może to być istotny czynnik – intensywne prototypowanie na lokalnej maszynie, a dopiero końcowe, wielkie treningi w chmurze.

Podsumowując, Apple Silicon z MLX plasuje się wydajnościowo poniżej najlepszych dedykowanych akceleratorów ML, ale oferuje wystarczającą moc do wielu zadań **przy znacznie mniejszym zapotrzebowaniu na energię**. W praktyce, Mac z układem M4 Max może zastąpić średniej klasy serwer GPU w pracowni badawczej, jeśli celem jest eksperymentowanie z modelami średniej wielkości lub prototypowanie koncepcji. Z kolei najwyższe półki (trenowanie ogromnych modeli, praca na gigantycznych zbiorach danych) wciąż należą do wyspecjalizowanych rozwiązań jak GPU A100/H100 czy TPU, głównie ze względu na skalowalność (możliwość łączenia wielu układów i potężne chłodzenie pozwalające utrzymać 100% obciążenia przez wiele dni). Niemniej różnica między “desktopowym” ML a “enterprise” ML zaciera się – Apple stale zwiększa możliwości swoich układów, tak że to co kiedyś wymagało klastra GPU, dziś może zostać zrobione na pojedynczym, stosunkowo cichym i energooszczędnym komputerze na biurku.

## Analiza kosztów obliczeniowych i efektywności energetycznej

Efektywność energetyczna to obszar, w którym Apple Silicon szczególnie błyszczy. Projektowane pierwotnie z myślą o urządzeniach mobilnych (iPhone, iPad), układy M1/M2/M3/M4 przynoszą filozofię *mobile-first* do świata komputerów osobistych. Oznacza to maksymalizację wydajności *per watt*. Dla zadań ML ma to kilka implikacji:

- **Maksymalna wydajność w granicach TDP:** Układy Apple mają relatywnie niskie TDP (Thermal Design Power) jak na swoją wydajność. Np. M2 Max w MacBooku Pro może zużywać ~30 W na GPU przy pełnym obciążeniu, podczas gdy porównywalnie szybkie (w pewnych zadaniach) GPU mobilne od Nvidii wymagałoby >80 W. W długotrwałym treningu, przekłada się to na mniejsze nagrzewanie i brak throttlingu – MacBook z M-chpem może liczyć przez wiele godzin bez drastycznego spadku taktowań, co w laptopach z dedykowanymi GPU często jest problemem (ograniczenie mocy wskutek temperatury). W rezultacie, czas przetworzenia np. 100 epok treningu modelu X może być na Macu zbliżony lub nawet krótszy niż na konkurencyjnym laptopie z mocnym GPU, który jednak po paru minutach musi zwolnić z powodu temperatur.

- **Niższy koszt energii:** Dla osób/firm wykonujących obliczenia ML lokalnie, zużycie energii może być istotnym kosztem (zarówno finansowym, jak i ekologicznym). Apple Silicon zużywa wyraźnie mniej prądu niż zestawy CPU+GPU o zbliżonej wydajności. Szacuje się, że wykonanie określonego zadania (np. trenowanie tego samego modelu przez N iteracji) na M2 Ultra pobierze kilkukrotnie mniej energii niż na stacji z Core i9 + RTX 3080​

    [techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=%3E%20%20%20,)

    ​

    [techjourneyman.com](https://techjourneyman.com/blog/m2-ultra-vs-nvidia-rtx-4090/# :~:text=Neural%20Engine%2C%20The%20M2%20Ultra,the%20best%20Apple%20Silicon%20ever)

    . W skali wielu eksperymentów, różnice te się sumują. Oczywiście, jeśli potrzebujemy wykonać zadanie 10× szybszo na potężnym GPU, to łączny czas może być krótszy i bilans energetyczny nie jest wprost 10× gorszy – ale w przedziale wydajności, w którym operuje Apple (powiedzmy średnia półka), jest on bardzo efektywny na wat.

- **Praca na baterii:** Unikalną cechą MacBooków z M1/M2/M3 jest możliwość trenowania modeli bez zasilania sieciowego przez dość długi czas. Na baterii laptop zyskuje ograniczenie mocy (taktowanie), ale nadal może wykonywać zadania ML. Dla porównania, próba trenowania czegoś poważniejszego na laptopie z GPU Nvidia bez ładowarki zwykle skutkuje szybkim rozładowaniem w ciągu kilkudziesięciu minut. W testach nieformalnych, MacBook Pro M2 Max był w stanie przeprowadzić kilkugodzinny trening modelu językowego na jednym ładowaniu (oczywiście przy zmniejszonej intensywności). To otwiera możliwości pracy w terenie, w podróży, gdzie wcześniej akcelerowane ML było praktycznie niemożliwe bez dostępu do zasilania.

- **Koszt sprzętu vs moc:** Należy też wspomnieć o koszcie zakupu sprzętu. Wydajne komputery Apple nie są tanie – np. Mac Studio M2 Ultra czy MacBook Pro M3 Max kosztują wiele tysięcy dolarów, podobnie zresztą jak laptopy/workstation z topowymi GPU. Jeśli jednak porównamy stosunek cena/wydajność w typowych zadaniach ML, to Apple wypada coraz lepiej: modele z GPU 30–40 rdzeni (M3 Max) doganiają karty typu Nvidia 3070/3080 w wielu testach, a jednocześnie komputer służy także jako pełne środowisko deweloperskie (dobra integracja, świetny ekran itp.). Oczywiście, dla bardzo budżetowych zastosowań zawsze można złożyć PC z używanym GPU i uzyskać sporo mocy taniej. Jednak w segmencie premium, kupując Maca, dostajemy względnie konkurencyjną maszynę ML bez potrzeby inwestowania osobno w GPU. W kontekście kosztów chmury: miesiąc pracy instancji z GPU A100 w chmurze może kosztować podobnie co zakup Maca mini z M2 Pro, który posłuży przez lata. Stąd wiele małych firm i indywidualnych badaczy rozważa Apple Silicon jako tańszą alternatywę do ciągłego korzystania z płatnych zasobów obliczeniowych (przynajmniej na fazę rozwoju i testów).

Reasumując, Apple Silicon oferuje znakomity balans między wydajnością a poborem mocy. Dla zadań typu trening modeli średniej wielkości czy lokalna inferencja dużych modeli, może dostarczyć wystarczającą moc przy ułamku kosztów energetycznych konkurencyjnych rozwiązań. W dużej skali (np. serwerowni) może nie wygrać z specjalizowanym sprzętem, ale w skali pojedynczego inżyniera ML – pozwala zrobić bardzo dużo, “nie płacąc” wysokich rachunków za prąd czy chłodzenie.

# Najlepsze praktyki MLX-ML na Apple Silicon

W ostatniej części skupimy się na tym, jak efektywnie wykorzystać MLX-ML oraz zasoby Apple Silicon w codziennej pracy z modelami uczenia maszynowego. Omówimy sposoby optymalizacji pamięci, techniki mieszanej precyzji, metody skutecznego dostrajania modeli (fine-tuning) oraz integrację z innymi narzędziami Apple, takimi jak Core ML czy Metal. Celem jest zaprezentowanie praktycznych wskazówek, które pozwolą wycisnąć maksimum wydajności i wygody z połączenia MLX + Apple Silicon.

## Optymalizacja pamięci i wykorzystanie Unified Memory

**Monitoruj zużycie pamięci:** Mimo że zunifikowana pamięć upraszcza wiele kwestii, wciąż mamy do czynienia z ograniczonym zasobem (fizycznym RAM). Warto na bieżąco monitorować, ile pamięci zajmuje nasz proces, np. używając narzędzi wbudowanych w macOS (Activity Monitor, `memory_pressure`) lub bezpośrednio w MLX. Jeśli model bliski jest wypełnienia całej pamięci, system może zacząć korzystać z dysku (swap), co drastycznie obniży wydajność. W MLX można wykorzystać API do pobierania informacji o urządzeniu GPU i dostępnej pamięci (np. `mx.gpu.device_info()`)​

[github.com](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/GPU.swift# :~:text=%2F%2F%2F%20Get%20information%20about%20the,var%20memSize)

. Dobre praktyki to: zwalnianie niepotrzebnych tensorów (np. poprzez dereferencję i `mx.collect_garbage()`), używanie mniejszych batchy jeśli brakuje pamięci, oraz w miarę możliwości planowanie obliczeń tak, by nie alokować ogromnych tablic tymczasowych.

**Lazy evaluation i strumienie:** MLX domyślnie jest **leniwie ewaluowany** – wiele operacji nie wykonuje się od razu, dopiero gdy potrzebny jest wynik. To pomaga unikać alokacji pośrednich i łączyć operacje. Można jeszcze bardziej zoptymalizować program, grupując operacje tak, by minimalizować piki użycia pamięci. Ponadto używanie strumieni (ang. streams) umożliwia nakładanie na siebie obliczeń CPU i GPU (jak w kodzie wcześniej). Dobrze jest z tego korzystać: np. przygotowywać kolejną porcję danych na CPU (w tle) podczas gdy GPU trenuje na bieżącej porcji – MLX dzięki unified memory i strumieniom ułatwia ten overlap.

**Unikaj zbędnych kopii:** Staraj się wszędzie używać struktur MLX zamiast np. mieszać je z numpy/tensorami PyTorch, ponieważ konwersje mogą powodować kopiowanie danych. Jeśli musisz np. wczytać dane, zrób to od razu do MLX (być może używając `mlx.numpy` submodułu, który jest kompatybilny z interfejsami numpy) lub przekonwertuj jednorazowo po wczytaniu i dalej operuj już w MLX. Dzięki temu dane będą od początku w zunifikowanej pamięci i unikniesz późniejszych transferów. MLX wspiera też bezpośrednią iterację po swoich tablicach, więc można pisać pętle treningowe analogicznie do PyTorch. W razie konieczności interoperacyjności (np. z PyTorch DataLoader), pamiętaj aby wyniki (PyTorch Tensor) skonwertować do `mx.array` – to spowoduje skopiowanie danych do unified memory, ale potem już nie będzie potrzebna żadna dodatkowa kopia.

**Profilowanie pamięci:** Gdy optymalizujesz większe projekty, rozważ użycie narzędzi profilujących. Apple dostarcza Instruments z profilowaniem pamięci, a MLX można też uruchomić w trybie debug (zmienne środowiskowe lub komendy debug, np. istnieją narzędzia do debugowania kerneli Metal​

[ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=Further%20Reading)

). Monitorując profil pamięciowy swojego treningu, możesz wykryć np. niezwalniane bufory albo momenty, gdzie jednocześnie alokowanych jest wiele dużych tensorów – często można wtedy zmodyfikować kod, by działał bardziej “w strumieniu” (pipeline), zamiast trzymać wszystko na raz.

**Wykorzystaj pamięć do maksimum:** Paradoksalnie, dobra praktyka to również nie bać się używać dostępnej pamięci, skoro już ją mamy. Na Apple Silicon nie ma sztucznego rozdziału “tyle dla GPU, tyle dla CPU” – jeśli twój Mac ma 64 GB RAM, możesz potencjalnie załadować cały zbiór danych do pamięci, przyspieszając dostęp (unikasz I/O dyskowego). Możesz też trzymać kilka wersji modelu, buforować batch’e itd., o ile mieścisz się w RAM – system macOS i tak dynamicznie przydziela zasoby, więc wykorzystanie 90% pamięci na zadanie ML jest OK, dopóki nic innego pilnie tej pamięci nie potrzebuje. Oczywiście, trzeba uważać, by nie doprowadzić do przepełnienia i swapu – stąd ponownie monitoring.

Podsumowując: kluczem jest świadomość, że unified memory to wspólny skarb, z którego korzystają wszystkie elementy. Trzeba go rozważnie dzielić i sprzątać po sobie, ale dzięki temu uzyskujemy ogromną elastyczność niedostępną na innych platformach.

## Techniki mieszanej precyzji i ich wpływ na wydajność

**Wykorzystanie **float16**/**bfloat16**:** Apple Silicon (GPU i od M2 także CPU/ANE) potrafi przyspieszyć obliczenia w niskiej precyzji. W praktyce najczęściej chodzi o użycie 16-bitowych formatów zamiast 32-bitowego `float32`. MLX domyślnie operuje na `float32`, ale łatwo zmienić typ danych. Można tworzyć tablice z zadanym dtype, np.:

```

# Przykład: macierz 1000x1000 w half-precision

mat = mx.random.uniform(low=0, high=1, shape=(1000,1000), dtype=mx.float16) ```

Powyższa macierz `mat` będzie zajmować połowę pamięci potrzebnej dla float32 i operacje na niej powinny być szybsze (na GPU Apple teoretyczna przepustowość dla float16 jest 2× wyższa). Alternatywnie, możemy rzutować istniejący model/zmienne na niższą precyzję – w MLX służy do tego metoda `astype()`, np. `model.astype(mx.float16)` (rzutuje wszystkie parametry modelu) lub `x.astype(mx.bfloat16)` dla tensora.

**Zalety **bfloat16**:** Bfloat16 ma 8-bitowy wykładnik, dzięki czemu zakres dynamiczny pokrywa zakres float32​

[eclecticlight.co](https://eclecticlight.co/2024/01/13/how-m1-macs-may-lag-behind/# :~:text=M1%20chips%20have%20more%20limited,point%20numbers%2C%20bfloat16)

. To bardzo pomocne przy trenowaniu – wartości podczas propagacji wstecznej (gradienty, sumy błędów) nie tracą skali tak łatwo jak w float16 o mniejszym wykładniku. Dlatego jeśli sprzęt wspiera bfloat16 (M2 i nowsze), warto rozważyć użycie go do trenowania modeli, zwłaszcza głębokich sieci. Bfloat16 daje prawie te same oszczędności pamięci i szybkości co float16, a jest stabilniejszy numerycznie (często nie wymaga stosowania trików typu *loss scaling* jak to robi się przy float16). W MLX zmiana typu na bfloat16 jest równie prosta. Należy jednak pamiętać, że nie wszystkie operacje mogą być zaimplementowane dla bf16 – choć core MLX i Metal to wspierają, to pewne warstwy mogą wewnętrznie i tak korzystać z 32-bit akumulacji.

**Mixed Precision Training (MPT):** Bardzo często stosuje się podejście mieszane, gdzie model jest w niższej precyzji, ale pewne operacje (np. akumulacja gradientów, obliczanie warunków optymalizacji w Adamie) są w wyższej precyzji dla dokładności. W MLX możemy ręcznie kontrolować dtype poszczególnych części, albo zdać się na gotowe narzędzia. Na przykład, trenowanie modeli z MLX można wspomóc poprzez ustawienie typu optymalizatora i gradientów na float32, podczas gdy wagi i aktywacje są float16. W MLX optymalizatory mogą działać na innym dtype niż parametry, o ile dostarczymy im odpowiednie argumenty (np. niektóre optymalizatory mają parametr `momentum_dtype` etc.). Dokumentacja MLX sugeruje, że framework będzie rozwijał wsparcie automatycznego MPT, na wzór PyTorch (gdzie `torch.cuda.amp` ułatwia to zadanie). Póki co, trzeba jawnie pilnować typów.

**Uwaga na konwersje:** Gdy korzystamy z mixed precision, unikajmy częstych konwersji typów podczas iteracji – to zjada korzyści. Lepiej zdefiniować z góry, że np. wszystkie warstwy mają `dtype=mx.float16` i wejścia też rzutować raz na fp16, niż co krok transformować tam i z powrotem. W MLX, jeśli model i dane są w float16, to wszelkie pośrednie tensory również będą w float16 (chyba że jakaś operacja wymusi wyższą precyzję). Dlatego pipeline: wektor wejściowy (fp16) $\to$ model (fp16) $\to$ loss (fp16) a następnie `loss.backward()` (który może akumulować deltę w fp32, co wewnętrznie zrobi MLX) – jest spójny i efektywny.

**Przykład zysku wydajności:** Dla konkretu – jeśli trenowanie pewnej sieci CNN w float32 zajmuje 10 minut na M2 Max, to w float16 może spaść do ~6–7 min, bez zauważalnej utraty dokładności końcowej (pod warunkiem prawidłowej stabilizacji treningu). Przy bardzo głębokich sieciach transformatorowych zyski bywają jeszcze większe, bo tam pamięciochłonność jest dominująca – ograniczenie połowy ruchu danych daje duży skok. Ponadto, większy batch size zmieści się w tej samej pamięci, co dodatkowo może poprawić wykorzystanie GPU.

**Walidacja poprawności:** Po treningu z mieszaną precyzją warto zweryfikować dokładność modelu np. w pełnej precyzji na zestawie walidacyjnym. Jeśli zauważymy odchyłki, można spróbować drobnych poprawek – np. wprowadzić *gradient clipping* (MLX ma `optimizers.clip_grad_norm`) czy warstwy normalizacyjne, które pomagają uśrednić wpływ niskiej precyzji.

Podsumowanie: Mixed Precision to obecnie standard w trenowaniu modeli – Apple Silicon w pełni to wspiera, a MLX czyni użycie go bardzo łatwym. Redukując precyzję obliczeń, oszczędzamy pamięć i czas, co na układach mobilnych jest szczególnie cenne. Rekomendacja: zawsze próbować trenować w float16/bfloat16, chyba że konkretny model wyraźnie traci stabilność – wtedy wrócić do float32 dla problematycznych fragmentów.

## Efektywne metody dostrajania modeli (fine-tuning) w MLX-ML

Często zamiast trenować modele od zera, chcemy dostroić istniejący model (np. pretrenowany na dużym zbiorze) do naszego zadania – to tzw. **fine-tuning**. Na Apple Silicon, z ograniczeniami mocy obliczeniowej w porównaniu do serwerowni, fine-tuning jest wręcz preferowanym sposobem pracy z większymi modelami. Oto dobre praktyki:

- **Wykorzystuj metody typu LoRA:** Low-Rank Adaptation (LoRA) to technika, w której do dużego modelu dodajemy niewielką liczbę dodatkowych wag (macierze niskiego rzędu) i tylko je uczymy, zamrażając oryginalne parametry. To dramatycznie zmniejsza wymagania pamięciowe i obliczeniowe dostrajania dużych modeli (np. LLM). MLX-ML wspiera tę technikę – w repozytorium przykładów MLX jest pokazany fine-tuning modelu LLaMA z użyciem LoRA​

    [github.com](https://github.com/ml-explore/mlx# :~:text=,Speech%20recognition%20with%20OpenAI%27s%20Whisper)

    . Dobrą praktyką jest załadowanie pretrenowanego modelu (np. 7B parametrów) w trybie inferencyjnym (można użyć nawet 4-bitowej kwantyzacji na wagi podstawowe, by zmieścić model), a następnie dodanie warstw LoRA (co w MLX sprowadza się do modyfikacji modułów np. `mlx.nn.Linear` by miały dodatkowe niskowymiarowe tensory uczone). Uczymy tylko te dodatkowe parametry – np. rzędu kilku milionów, zamiast miliardów – co Apple Silicon jest w stanie udźwignąć. Po dostrojeniu, oryginalne wagi plus poprawki LoRA dają nam model dostosowany do zadania.

- **Stopniowe zamrażanie/odmrażanie:** Jeżeli model nie jest ekstremalnie duży i chcemy jednak dostroić większość wag, można zastosować strategię zamrażania większości warstw i trenowania tylko najwyższych (np. ostatnich kilku warstw sieci) początkowo, a potem stopniowo odmrażać wcześniejsze. W MLX oznacza to ustawienie `requires_grad=False` dla wybranych parametrów modelu. Przykładowo, biorąc ResNet50 pretrenowany na ImageNet, możemy na początku trenować tylko warstwy FC klasyfikatora (co wymaga minimalnej mocy), a następnie ewentualnie drobnie dostroić także ostatnie bloki konwolucyjne. Dzięki temu nigdy nie musimy jednocześnie optymalizować wszystkich 25 mln wag – to zmniejsza wymagania pamięci dla optimizerów i sprawia, że trening jest szybszy. W MLX optymalizator będzie po prostu ignorował parametry, które nie wymagają gradientu, więc narzut jest niewielki.

- **Kwantyzacja podczas treningu (QAT):** Bardziej zaawansowaną techniką jest trenowanie z kwantyzacją wag/aktywacji (np. do int8). Apple Neural Engine i Core ML mocno korzystają z int8 przy inferencji, więc dostrajając model można od razu uwzględnić ten fakt. Co prawda MLX nie ma gotowego modułu QAT, ale można symulować kwantyzację – np. dodając szum lub zaokrąglanie do 8-bit po każdej aktualizacji wag. To zapewni, że po eksporcie do Core ML (gdzie model zostanie ściśnięty do int8) dokładność nie spadnie niespodziewanie. Narzut obliczeniowy takiej symulacji jest dodatkowy, ale wciąż tańszy niż pełny trening od zera, a oszczędza sporo bólu przy wdrożeniu.

- **Wykorzystanie wbudowanych danych i augmentacji:** Przy dostrajaniu modeli CV warto korzystać z przyspieszania na GPU także dla augmentacji danych. Apple M-chipy mają potężny Media Engine oraz GPU, które mogą w ramach pipeline’u MLX wykonywać np. operacje geometryczne na obrazach. Co prawda MLX nie jest specjalistyczną biblioteką augmentacji, ale można np. załadować obrazy jako tekstury Metal i przekształcać je shaderami. Alternatywnie, biblioteka Core Image może użyć GPU/ANE do szybkiej obróbki obrazu przed przekazaniem do MLX. Dzięki temu dane treningowe są przygotowywane w czasie zbliżonym do obliczeń modelu, nie dławiąc CPU.

- **Małe modele na próbę:** Jeżeli planujemy dostrajanie kosztownego modelu, dobrą praktyką (nie tylko na Apple) jest przetestowanie pipeline’u na pomniejszonej wersji modelu lub zmniejszonym wymiarze. Np. zanim uruchomisz fine-tuning pełnego Transformer 20B, spróbuj na modelu 2B czy wszystko działa i czy MLX poprawnie wykorzystuje GPU. Pozwoli to upewnić się, że nie napotkamy nieprzyjemnych niespodzianek (np. out-of-memory) po godzinie treningu. W MLX, dzięki dynamicznemu grafowi, łatwo można zmienić np. rozmiar modelu i uruchomić ten sam skrypt – nie ma kompilacji wstępnej zależnej od wymiarów, jak to bywa w niektórych narzędziach, więc iteracja jest szybka.

Podsumowując, fine-tuning na Apple Silicon jest realny i efektywny, o ile zastosujemy sprytne techniki ograniczające obciążenie. Metody takie jak LoRA w połączeniu z MLX czynią możliwym wytrenowanie np. osobistego modelu językowego na własnym laptopie, co jeszcze parę lat temu brzmiało nieprawdopodobnie. Kluczem jest dostosowanie zakresu uczenia do możliwości sprzętu – uczyć mniej parametrów jednocześnie, wykorzystywać pretrenowaną wiedzę modelu i pamiętać o narzędziach takich jak kwantyzacja.

## Integracja MLX-ML z ekosystemem Apple (MPS, Core ML, Metal)

Aby w pełni skorzystać z możliwości Apple Silicon, warto znać sposoby integracji MLX z pozostałymi elementami ekosystemu programistycznego Apple:

- **Metal Performance Shaders (MPS):** Choć MLX sam w sobie zastępuje bezpośrednie używanie MPS, możemy w niektórych przypadkach sięgnąć po MPS lub *MPS Graph* dla operacji, których MLX nie oferuje. Np. jeśli chcemy użyć niestandardowej funkcji aktywacji, która nie jest zaimplementowana w MLX, a istnieje w MPS, możemy napisać własny moduł MLX korzystający z wywołań MPS. MLX jest na tyle elastyczny, że pozwala mieszać te podejścia – tablice MLX mogą być konwertowane do `MPSMatrix` czy `MPSNDArray` jeśli naprawdę zajdzie taka potrzeba, bo ostatecznie rezydują w tej samej fizycznej pamięci. Apple dostarcza sporo gotowych “prymitywów” w MPS (sploty, pooling, normalization), więc zaawansowani użytkownicy mogą tworzyć hybrydowe rozwiązania. Jednak zaleceniem jest raczej próba zaimplementowania w czystym MLX (np. pisząc kernel Metal – o czym poniżej).

- **Pisanie własnych kerneli Metal:** MLX udostępnia API do dodawania własnych operacji na GPU poprzez napisanie shadera w języku Metal i zarejestrowanie go jako opcji w MLX​

    [ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html# :~:text=Further%20Reading)

    . Jest to opcja dla ekspertów, gdy potrzebna jest optymalizacja niskopoziomowa. Przykład: mamy nietypową warstwę, która łączy kilka standardowych operacji – można to zrobić jako oddzielne kroki w MLX (co może spowodować kilka przebiegów pamięci), lub napisać jeden kernel, który zrobi wszystko w jednym. Taki kernel może w pełni wykorzystać GPU, minimalizując obciążenie pamięci. MLX w dokumentacji ma sekcję i przykład jak to zrobić (“Custom Metal Kernels”). W praktyce niewielu użytkowników będzie musiało po to sięgać, ale dobrze wiedzieć, że taka możliwość istnieje – to przewaga open-source’owego i konfigurowalnego rozwiązania.

- **Core ML i model deployment:** Po zakończeniu eksperymentów w MLX, zapewne przyjdzie potrzeba wykorzystania modelu np. w aplikacji na iPhone albo udostępnienia go innym. Ekosystem Apple oferuje Core ML jako uniwersalny runtime dla modeli na wszystkich urządzeniach (z automatycznym wykorzystaniem ANE, GPU lub CPU w zależności od dostępności). Niestety, nie ma jeszcze prostego „Export to CoreML” w MLX. Jednak ponieważ MLX jest kompatybilny koncepcyjnie z PyTorch (warstwy, parametry), można posłużyć się pewnym obejściem: wyeksportować wytrenowany model z MLX, np. zapisując parametry i strukturalnie odtwarzając go w PyTorch (być może korzystając z podobieństwa API `mlx.nn` do `torch.nn`). Następnie użyć `coremltools` do konwersji modelu PyTorch -> CoreML. To oczywiście dodatkowa praca, ale pozwala wdrożyć wyniki uzyskane dzięki MLX. Alternatywnie, można wykorzystać framework **Core ML Stable Diffusion** czy **Core ML LLM** już po stronie wdrożenia i załadować wagi, które wytrenowaliśmy w MLX. Apple publikuje referencyjne implementacje modeli pod Core ML (np. Transformers, Stable Diffusion), które można zasilić własnymi wagami. Wtedy wykonanie w aplikacji będzie natywne.

- **Integracja ze Swift i aplikacjami:** Jak wspomniano, MLX ma interfejs Swift. To oznacza, że kod np. trenowania modelu może zostać napisany w aplikacji macOS/iOS bez udziału Pythona. Otwiera to ciekawe możliwości: można tworzyć interaktywne aplikacje, które uczą się na urządzeniu (on-device learning). Przykładowo, aplikacja fitness może na bieżąco dostrajać model personalizujący rozpoznawanie aktywności użytkownika na bazie jego danych – wszystko to prywatnie na iPhonie, korzystając z MLX Swift i mocy ANE/GPU urządzenia. Ponieważ MLX Swift jest stosunkowo nowy, brak jeszcze wielu wysokopoziomowych samouczków, ale Apple wskazuje go jako przyszłościowy kierunek​

    [swift.org](https://swift.org/blog/mlx-swift/# :~:text=silicon,deployment%20of%20models%20in%20apps)

    ​

    [swift.org](https://swift.org/blog/mlx-swift/# :~:text=MLX%20has%20several%20important%20features,These%20include)

    . Już teraz jednak możemy łączyć MLX (trening) z Core ML (inference) w jednym projekcie Swift: np. użyć MLX do zaimplementowania algorytmu treningu w aplikacji, a następnie zapisać wyniki jako model Core ML i dalej korzystać z niego do predykcji – wszystko bez opuszczania środowiska Xcode.

- **Metal Performance Shaders Graph i BNNS:** Wspominając dla kompletności – Apple oferuje też inne biblioteki niskiego poziomu jak BNNS (Basic Neural Network Subroutines) akcelerowane przez CPU, czy MPS Graph do budowania grafów obliczeń na GPU. MLX w zasadzie abstrahuje od konieczności użycia tych rzeczy, ale czasem np. BNNS może być przydatne do prostych obliczeń na CPU (np. drobne sieci, które szybciej wykonają się na CPU niż uruchamiać GPU). BNNS jest dostępne przez Accelerate.framework w C/Swift. Można te wyniki potem wprowadzać do MLX arrays. Taka integracja jednak ma sens tylko w bardzo specyficznych scenariuszach.

Na co dzień, najlepszą praktyką jest trzymanie się MLX do trenowania i testowania modeli na Macu, a do wdrażania – korzystanie z oficjalnych narzędzi Core ML. Dzięki temu zyskujemy to, co najlepsze z obu światów: szybki rozwój i eksperymenty (z MLX), a następnie dopracowaną, zoptymalizowaną platformę wykonawczą na miliardach urządzeń (Core ML na iPhone, iPad, Mac, Apple Watch itd.). Apple Silicon jako wspólna platforma sprzętowa sprawia, że model wytrenowany na jednym urządzeniu z łatwością ruszy na innym, a spójność narzędzi (Metal wszędzie pod spodem) gwarantuje przewidywalną wydajność.

\bigskip

**Podsumowanie:** MLX-ML na Apple Silicon to potężne połączenie – dzięki niemu możemy efektywnie wykorzystać unikalną architekturę chipów M1/M2/M3/M4 do trenowania i uruchamiania modeli ML. Znajomość architektury (CPU, GPU, Neural Engine, Unified Memory) pozwala pisać wydajniejszy kod i unikać pułapek. Benchmarki pokazują, że Apple dynamicznie goni konkurencję, oferując już teraz 2× szybsze działanie niż standardowe frameworki na Macu, choć oczywiście w świecie high-end wciąż królują wielkie GPU i TPU. Poprzez stosowanie najlepszych praktyk – od optymalizacji pamięci, przez mieszaną precyzję, po sprytne fine-tuning – możemy uruchamiać na Macach coraz bardziej zaawansowane modele, nie tylko do inferencji, ale i do treningu. Co więcej, integracja z ekosystemem (Swift, Core ML, Metal) umożliwia płynne przenoszenie rozwiązań z fazy prototypu do produktu.

Można śmiało stwierdzić, że Apple Silicon + MLX to zapowiedź nowej ery **on-device ML**, gdzie granica między tym co “lokalne” a “w chmurze” będzie się zacierać, dając użytkownikom moc trenowania i personalizowania sztucznej inteligencji we własnych rękach.%\cite{apple_mlx_paper, metal_perf_doc} (sample citation, replace with actual BibTeX references)

#####
## Instalacja MLX-LM oraz konfiguracja środowiska Najpierw upewnij się, że masz zainstalowaną najnowszą wersję MLX-LM, zoptymalizowaną pod Apple Silicon. Przykładowa instalacja (w terminalu macOS): ```

# Zaktualizuj pip do najnowszej wersji

pip install --upgrade pip

# Instalacja MLX-LM (przykładowa komenda; repozytorium jest hostowane np. na PyPI lub GitHub)

pip install mlx-lm ```

\noindent Upewnij się, że korzystasz z wersji Pythona 3.9 lub nowszej oraz, że Twoje środowisko ma dostęp do Apple MPS lub natywnego wsparcia Metal (domyślnie dostępne w macOS 13+).

## Podstawowe parametry treningu w MLX-LM Przy fine-tuningu modeli dużych (np. 7B lub wyższych) szczególnie ważne są:  -  **Batch Size:** Z reguły ustawiany na niską wartość (1--2), aby zmieścić model w zunifikowanej pamięci. -  **Iteracje:** Określ liczbę iteracji lub epok, zależnie od rozmiaru datasetu. -  **LoRA:** Zalecane użycie adapterów LoRA lub QLoRA w celu redukcji liczby trenowanych parametrów. -  **Gradient Checkpointing:** W razie problemów z pamięcią używaj flagi `--grad-checkpoint`. -  **Mixed Precision:** Użycie FP16/bfloat16 znacznie przyspiesza trening i zmniejsza zużycie pamięci.

Przykładowa komenda startowa dla fine-tuningu modelu 7B: ``` mlx_lm.finetune
--model /path/to/7B_model
--data /path/to/dataset
--adapter-type lora
--batch-size 2
--iters 2000
--learning-rate 0.0003
--grad-checkpoint % Włącza gradient checkpointing dla oszczędności pamięci --precision fp16 % Użycie float16 (można zmienić na bf16) --output /path/to/output_adapter ```

Komentarze w powyższym poleceniu wyjaśniają kluczowe flagi. Każdy projekt może wymagać drobnych modyfikacji (np. liczby iteracji zależy od rozmiaru datasetu).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # Przykłady Fine-Tuningu dla Różnych Zadań

W tej sekcji przedstawiono przykłady poleceń i konfiguracji fine-tuningu w MLX-LM dla różnych zastosowań. Każdy przykład zawiera krótki opis celu, komendę oraz komentarze.

## (1) Fine-Tuning Językowy – Nauka Języka Polskiego (Model 3B) **Cel:** Dostosowanie pretrenowanego modelu 3B do lepszego generowania tekstu w języku polskim, w tym fachowego słownictwa.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/model_3B_unquantized
--data /datasets/polish_language.jsonl
--adapter-type lora
--batch-size 1
--iters 1500
--learning-rate 0.0002
--grad-checkpoint
--precision fp16
--output /output/3B_polish_lora ``` \noindent **Komentarz:** Dla mniejszych modeli (3B) używamy batch-size 1, aby upewnić się, że wszystko mieści się w pamięci. Dataset powinien zawierać przykłady z naturalnym językiem polskim, w tym terminologię specjalistyczną. Parametr `--adapter-type lora` wskazuje, że używamy adapterów LoRA.

## (2) Fine-Tuning Językowy – Nauka Języka Polskiego (Model 7B) **Cel:** Dostosowanie modelu 7B do specyficznego zadania NLP, np. dostosowania do języka polskiego w kontekście prawnym lub medycznym.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/model_7B_unquantized
--data /datasets/polish_legal_medical.jsonl
--adapter-type lora
--batch-size 2
--iters 3000
--learning-rate 0.00025
--grad-checkpoint
--precision bf16
--output /output/7B_polish_legal_medical_lora ``` \noindent **Komentarz:** Wybieramy większy batch (2) i używamy bfloat16, co zapewnia lepszą stabilność przy niższej precyzji, szczególnie przy bardziej złożonych danych z dziedziny prawa lub medycyny.

## (3) Fine-Tuning Technologiczny – Nauka Języka Programowania (Model 7B) **Cel:** Dostrojenie modelu do generowania kodu oraz technicznej dokumentacji, np. dla języka Python lub JavaScript.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/model_7B_codebase
--data /datasets/tech_code_instructions.jsonl
--adapter-type lora
--batch-size 2
--iters 2500
--learning-rate 0.0003
--grad-checkpoint
--precision fp16
--output /output/7B_code_lora ``` \noindent **Komentarz:** Dataset powinien zawierać przykłady kodu, dokumentację oraz wyjaśnienia techniczne. Zalecane jest użycie LoRA oraz gradient checkpointing dla oszczędności pamięci przy dostrajaniu modeli technologicznych.

## (4) Fine-Tuning CV – Model Detekcji Obrazów (Model 7B lub 3B specjalizowany) **Cel:** Dostosowanie modelu do zadań detekcji i klasyfikacji obrazów – np. w zastosowaniach medycznych (diagnostyka obrazowa).

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/model_CV_7B_unquantized
--data /datasets/medical_images.jsonl
--adapter-type lora
--batch-size 1
--iters 4000
--learning-rate 0.0002
--grad-checkpoint
--precision fp16
--output /output/7B_medical_cv_lora ``` \noindent **Komentarz:** Dataset powinien zawierać obrazy (wcześniej skonwertowane do tekstowej reprezentacji lub w formacie specyficznym dla modelu CV, np. tensorów). Fine-tuning odbywa się podobnie jak w NLP, ale z naciskiem na przetwarzanie cech wizualnych.

## (5) Fine-Tuning VL – Model Multimodalny (Model 7B) **Cel:** Dostrojenie modelu multimodalnego do zadania łączenia obrazu z tekstem, np. generowanie opisów obrazów medycznych.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/model_VL_7B_unquantized
--data /datasets/medical_vl.jsonl
--adapter-type lora
--batch-size 1
--iters 3500
--learning-rate 0.00025
--grad-checkpoint
--precision fp16
--output /output/7B_medical_vl_lora ``` \noindent **Komentarz:** Dataset multimodalny zawiera pary obraz+tekst. Warto zadbać o spójny format (np. klucze `"image_path"` i `"caption"`). Fine-tuning przeprowadzamy przy niskim batchu, aby umożliwić przetwarzanie obrazu w zunifikowanej pamięci.

## (6) Fine-Tuning Generatywny – Stable Diffusion (Model 3B lub 7B wersja generatywna) **Cel:** Dostosowanie modelu generatywnego do określonego stylu artystycznego lub tematyki wizualnej.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/stable_diffusion_7B_unquantized
--data /datasets/art_style_prompts.jsonl
--adapter-type lora
--batch-size 1
--iters 3000
--learning-rate 0.0003
--grad-checkpoint
--precision fp16
--output /output/7B_sd_artstyle_lora ``` \noindent **Komentarz:** Dataset zawiera prompt’y opisujące styl, a oczekiwane wyniki to obrazy generowane przez model. Fine-tuning w tym kontekście często opiera się na specyficznym stylu lub tematyce, co wymaga starannej kuracji danych.

## (7) Fine-Tuning STT/TTS – Model Rozpoznawania i Generowania Mowy (Model 3B lub 7B) **Cel:** Dostrojenie modelu do rozpoznawania mowy (STT) lub generowania mowy (TTS) z uwzględnieniem specyficznego słownictwa branżowego.

**Przykładowa komenda:** ``` mlx_lm.finetune
--model /models/stt_tts_7B_unquantized
--data /datasets/speech_transcripts.jsonl
--adapter-type lora
--batch-size 1
--iters 4000
--learning-rate 0.0002
--grad-checkpoint
--precision fp16
--output /output/7B_stt_tts_lora ``` \noindent **Komentarz:** Dataset zawiera pary: nagranie (lub jego reprezentację) oraz transkrypcję lub syntezę mowy. Fine-tuning umożliwia modelowi lepsze rozpoznawanie akcentu, specyficznego słownictwa lub intonacji charakterystycznej dla danej branży (np. medycznej).

## (8) Fine-Tuning Konwolucyjnych Sieci Neuronowych – U-Net/ResNet/GAN **Cel:** Dostosowanie modeli konwolucyjnych do specyficznych zadań, takich jak segmentacja obrazów medycznych lub generacja realistycznych obrazów.

**Przykładowa komenda dla U-Net:** ``` mlx_lm.finetune
--model /models/unet_3B_unquantized
--data /datasets/medical_segmentation.jsonl
--adapter-type lora
--batch-size 1
--iters 3500
--learning-rate 0.0003
--grad-checkpoint
--precision fp16
--output /output/3B_medical_unet_lora ``` \noindent **Komentarz:** Dane powinny zawierać obrazy medyczne wraz z odpowiadającymi im maskami segmentacyjnymi. W przypadku GAN, fine-tuning może obejmować oddzielne dostrajanie generatora i dyskryminatora – procedura jest bardziej złożona, ale logika pozostaje podobna: używamy adapterów LoRA, gradient checkpointing i optymalizacji mixed precision.

## (9--20) Inne Scenariusze Fine-Tuningu W ramach niniejszego przewodnika przedstawiamy dodatkowe przykłady, których celem jest pokazanie szerokiego spektrum zastosowań. Poniżej znajduje się skrócony katalog kolejnych 11 scenariuszy (łącznie z poprzednimi przykładowymi 8) – każdy z nich można rozszerzyć na podobieństwo powyższych sekcji:

[label=(\roman*)] -  **Fine-tuning LLM do personalizacji dialogu asystenta:** Użycie datasetu zawierającego przykłady konwersacji, format `chat` JSONL, adaptacja stylu odpowiedzi. -  **Fine-tuning LLM dla specjalistycznej terminologii medycznej:** Dataset z artykułami medycznymi, FAQ medycznymi, poprawa jakości generowania dokumentacji. -  **Fine-tuning LLM do analizy prawnej:** Zbiór przypadków sądowych, opinii prawnych, kodeksu karnego – dostrojenie modelu do języka prawniczego. -  **Fine-tuning CV dla klasyfikacji obiektów przemysłowych:** Dataset zdjęć z linii produkcyjnej, klasyfikacja defektów. -  **Fine-tuning VL dla systemów wyszukiwania obrazów:** Łączenie opisów tekstowych z obrazami produktów – dostrojenie modelu do systemów rekomendacyjnych. -  **Fine-tuning modeli generatywnych dla sztuki cyfrowej:** Użycie specyficznych promptów artystycznych – generacja obrazów w unikalnym stylu. -  **Fine-tuning modeli generatywnych dla scenariuszy reklamowych:** Dataset zawierający slogany, opisy produktów, grafiki – model generuje spersonalizowane treści reklamowe. -  **Fine-tuning STT/TTS dla regionalnych akcentów:** Dataset nagrań z różnymi akcentami, dostrojenie modelu do rozpoznawania mowy regionalnej. -  **Fine-tuning TTS dla narracji audiobooków:** Adaptacja modelu generującego mowę do stylu lektora – dbałość o intonację i modulację głosu. -  **Fine-tuning sieci do detekcji obiektów w obrazach satelitarnych:** Dataset satelitarnych zdjęć, klasyfikacja i lokalizacja obiektów (np. budynki, drogi). -  **Fine-tuning modeli segmentacyjnych dla obrazów medycznych – alternatywna architektura:** Użycie np. DeepLabV3 lub FCN zamiast U-Net, ze specyficzną adaptacją do segmentacji tkanki.

Każdy z powyższych przykładów wymaga indywidualnego dostosowania parametrów (learning rate, liczba iteracji, precyzja, batch size) oraz przygotowania datasetu w odpowiednim formacie (JSONL, CSV, foldery obrazów z etykietami). Warto eksperymentować na mniejszych wersjach modeli przed przejściem do pełnoskalowego dostrajania na modelach 32B lub 70B, aby uniknąć problemów związanych z pamięcią i czasem treningu.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # Podsumowanie i Poradnik Dobrych Praktyk\label{sec:podsumowanie-dobrych-praktyk}

## Kluczowe Zalecenia  -  **Rozpoczynaj od małych modeli.** Testuj konfigurację treningową na modelach o mniejszej liczbie parametrów (np. 3B lub 7B) przed skalowaniem do 32B lub 70B. -  **Używaj technik PEFT (np. LoRA/QLoRA)** – pozwala to znacząco zmniejszyć wymagania pamięciowe, a efekty fine-tuningu są porównywalne z pełnym dostrojeniem. -  **Optymalizuj zużycie pamięci** – stosuj gradient checkpointing, mixed precision (FP16 lub bfloat16) i dbaj o zarządzanie pamięcią zunifikowaną. -  **Monitoruj metryki treningu** – zarówno loss, jak i zużycie pamięci, czas iteracji, by móc reagować na ewentualne spadki wydajności lub problemy z przepełnieniem. -  **Dostosuj dataset do zadania.** Jakość danych ma kluczowe znaczenie – lepiej mieć mniejszy, starannie dobrany zbiór niż masę danych niskiej jakości. -  **Testuj modele w warunkach zbliżonych do docelowych zastosowań** – jeśli model ma być wdrożony na urządzeniach mobilnych, przetestuj inferencję na Core ML lub bezpośrednio na Apple Silicon.

## Ogólne Porady  -  Używaj wersjonowania kodu i danych – zapisywanie checkpointów, logowanie hiperparametrów i wyników eksperymentów pozwala wrócić do najlepszej konfiguracji. -  Korzystaj z gotowych repozytoriów i szablonów MLX-ML – społeczność Apple udostępnia przykłady, które można dostosować do własnych potrzeb. -  Eksperymentuj systematycznie – zmieniaj pojedynczy parametr na raz (learning rate, liczba iteracji, batch size) i zapisuj wyniki. -  Pamiętaj o regularnym walidowaniu modelu na zbiorze testowym, aby uniknąć overfittingu. -  Utrzymuj spójny format danych – szczególnie przy danych multimodalnych, gdzie synchronizacja obrazów i tekstu jest kluczowa. -  Jeśli trenujesz na Apple Silicon, korzystaj z natywnych narzędzi Apple (MLX, Metal) zamiast przenosić kod z innych platform – zapewni to lepszą integrację i wydajność.

\bigskip

**Podsumowanie:**
Niniejszy rozdział dostarcza kompleksowej instrukcji używania MLX-LM do lokalnego fine-tuningu modeli o różnych rozmiarach i zastosowaniach. Przedstawiono przykłady poleceń dla modeli językowych, technologicznych, wizualnych, generatywnych, STT/TTS, konwolucyjnych oraz modeli detekcyjnych i segmentacyjnych. Kluczowym elementem jest wykorzystanie technik takich jak LoRA, gradient checkpointing, mixed precision oraz zoptymalizowana integracja z zunifikowaną pamięcią Apple Silicon. Stosowanie powyższych praktyk umożliwia efektywne dostrajanie modeli nawet na sprzęcie o ograniczonych zasobach, zapewniając jednocześnie wysoką jakość wyników. Dzięki temu MLX-LM staje się potężnym narzędziem w rękach badaczy i deweloperów, pozwalając na szybką iterację i wdrożenie zaawansowanych modeli AI lokalnie.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \begin{thebibliography}{9} \bibitem{MLX2024docs} Apple MLX Team (2024). *MLX: An Array Framework for Machine Learning on Apple Silicon – Documentation and Examples*. \bibitem{Hu2021LoRA} Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. \bibitem{Dettmers2023QLoRA} Dettmers, T., et al. (2023). *QLoRA: Efficient Fine-Tuning of Quantized LLMs*. arXiv:2305.14314. \bibitem{AppleUnifiedMemory} Apple Inc. (2023). *Technical Overview of Apple Silicon Unified Memory Architecture*. \bibitem{CoreMLtools} Apple Inc. (2022). *Core ML Tools Documentation*. \bibitem{Axolotl2024} Axolotl Team (2024). *Axolotl: A Fine-Tuning Framework for LLMs*. GitHub Repository. \bibitem{Unsloth2024} Unsloth Project (2024). *Unsloth: Optimized Fine-Tuning on a Single GPU*. GitHub Repository. \end{thebibliography}

#####
######
#####
######
# Kompleksowy przewodnik: lokalny fine-tuning modeli 3B, 7B, 32B i 70B z MLX-LM

**MLX-LM** to framework Apple Research zaprojektowany do wydajnego trenowania modeli AI na Apple Silicon (M1/M2/M3)​

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

. Umożliwia on lokalny fine-tuning dużych modeli językowych (LLM) i innych sieci, wykorzystując GPU Apple (Metal/MPS) oraz zunifikowaną pamięć Mac (co pozwala trenować modele większe niż typowa VRAM karty graficznej)​

[reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=Me%20neither,per%20watt%2C%20which%20is%20surprising)

. Poniżej przedstawiamy szczegółowy przewodnik dostrajania modeli o rozmiarach 3B, 7B, 32B i 70B z użyciem MLX-LM, z podziałem na różne przypadki użycia. W każdej sekcji opisujemy zalecane komendy, praktyki i techniki (LoRA, QLoRA, itp.), a na końcu porównujemy MLX-LM z innymi narzędziami do fine-tuningu na Macach (np. PyTorch/MPS, Ollama, MLC) wraz z omówieniem strategii i hiperparametrów dopasowanych do sprzętu (Mac Studio M2 Ultra vs. MacBook Pro M3 Max).

## Fine-tuning językowy (dostosowanie do języka polskiego i branżowego słownictwa)

**Cel**: Dostroić istniejący model językowy (np. LLaMA 7B/13B, Mistral 7B, itp.) do lepszego zrozumienia i generowania języka polskiego, w tym specjalistycznej terminologii z wybranych branż (medycyna, prawo, IT itp.).

**Przygotowanie danych**: Zbierz korpus polskojęzyczny adekwatny do zadania. Mogą to być dane konwersacyjne (pytanie-odpowiedź), dokumenty branżowe, artykuły naukowe itp. Ważne, by dane były w formacie obsługiwanym przez MLX-LM – np. pliki JSONL z każdym przykładem w osobnej linii. Dla danych dialogowych użyj formatu `{"messages": [...]}` lub dla tekstu do kontynuacji formatu `{"prompt": ..., "completion": ...}` zgodnie z dokumentacją MLX​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Data)

​

[github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Note%2C%20the%20format%20is%20automatically,the%20loader%20will%20be%20ignored)

. Upewnij się, że teksty zostały poprawnie sformatowane i oczyszczone (np. usunięcie zbędnych znaków) – MLX jest czuły na poprawność formatu JSONL​

[reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/# :~:text=Worked%20fine%2C%20except%20a%20small,existential%20crisis%20solved%20HERE)

​

[reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/# :~:text=Lesson%202%3A%20Use%20instruct%20formatting%2C,results%20out%20of%20the%20model)

.

**Wybór modelu bazowego**: Najlepiej zacząć od modelu, który już częściowo „zna” język polski. Przykładowo LLaMA 2 jest trenowany wielojęzycznie, więc stanowi dobrą bazę. Jeśli jednak model bazowy jest głównie anglojęzyczny, fine-tuning na polskim korpusie może znacznie poprawić płynność i poprawność polskich odpowiedzi. Możesz też użyć istniejących modeli instruowanych po polsku (o ile dostępne) jako punkt wyjścia.

**Komenda fine-tune z MLX-LM**: Po zainstalowaniu MLX-LM (`pip install mlx-lm`), używamy polecenia CLI `mlx_lm.lora`. Przykład uruchomienia treningu (LoRA) dla modelu na danych w folderze `data_polish` z plikami `train.jsonl` i `valid.jsonl`:

bash

Copy

`mlx_lm.lora --model <ścieżka/ID_modelu> \             --train \             --data data_polish \             --iters 1000 \             --learning-rate 2e-4 \             --batch-size 2 \             --mask-prompt`

Wyjaśnienie najważniejszych opcji:

- `--model` wskazuje nazwę repozytorium HuggingFace lub lokalną ścieżkę modelu (musi to być model w formacie HF, nie spakowany .gguf, bo MLX nie trenuje bezpośrednio modeli gguf​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=mlx%2C%206%20for%20unsloth%2C%204,for%20axolotl)

    ​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=you%20have%20to%20use%20hugging,tuning)

    ).
- `--train` uruchamia tryb trenowania (fine-tune).
- `--data` wskazuje ścieżkę do danych treningowych/walidacyjnych (wymagane pliki `train.jsonl` i `valid.jsonl` w tym folderze)​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=The%20%60,see%20the%20section%20on%20Data)

    ​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=%E2%80%94fine,it%20to%20full%20or%20dora)

    .
- `--iters 1000` oznacza liczbę iteracji treningowych (dostosuj w zależności od wielkości zbioru i pożądanego przetrenowania; często kilkaset do kilku tysięcy iteracji wystarcza przy niewielkim zbiorze).
- `--learning-rate` ustawia początkowy learning rate (np. 2e-4 dla LoRA to bezpieczna wartość, ale warto eksperymentować – zbyt duża może popsuć wcześniejszą wiedzę modelu, zbyt mała spowolni naukę).
- `--batch-size` określa ile przykładów na raz przetwarzamy – na Macach z ograniczoną pamięcią często trzeba użyć małego batcha (1–4) by uniknąć błędów pamięci​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=Memory%20Optimization)

    ​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/# :~:text=)

    .
- `--mask-prompt` (opcjonalnie) sprawia, że model liczy koszt błędu tylko dla części „odpowiedzi”, ignorując treść promptu podczas obliczania straty – przydatne, jeśli trenujemy model w formacie instrukcji/konwersacji, aby nie „uczył się” powtarzać promptu​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=)

    .

**Najlepsze praktyki**:

- _Stopniowe dostrajanie_: Jeśli model bazowy jest czysto angielski, rozważ **stopniowe** dostrajanie – najpierw na dużym ogólnym polskim korpusie (aby nauczyć polskiego), a dopiero potem na wąskim specjalistycznym żargonie. To pomoże uniknąć „katastrofalnego zapominania” wiedzy ogólnej. Alternatywnie, **mix** danych – np. trening na mieszance danych angielskich i polskich – aby model zachował wielojęzyczność.
- _LoRA vs full fine-tune_: Domyślnie MLX-LM używa LoRA – tj. dodaje niskowymiarowe adaptery, zamiast modyfikować wszystkie wagi modelu​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=)

    . To świetne rozwiązanie dla utrzymania oryginalnych zdolności modelu (np. nie popsuje wiedzy ogólnej w jęz. angielskim) oraz dla ograniczeń sprzętowych (znacznie mniejsza pamięć zużyta niż pełny fine-tune). **Zalecamy LoRA** do dostrajania językowego. Pełny fine-tuning (`--fine-tune-type full`) mógłby ewentualnie dać nieco wyższe wyniki w polskim kosztem dużego ryzyka zapomnienia oryginalnego języka i ogromnych wymagań pamięciowych​

    [github.com](https://github.com/ml-explore/mlx-examples/issues/297# :~:text=%5BFeature%20Request%5D%20Full,phi2%20could%20cause%20some%20issues)

    .
- _Słownictwo i tokenizer_: Upewnij się, że tokenizer modelu obsługuje polskie znaki i słowa. Modele HF zwykle mają wbudowane tokenizery subword, które znają popularne polskie ciągi, ale dla bardzo specjalistycznego słownictwa może się zdarzyć, że model „literuje” słowo na wiele tokenów. Fine-tuning nauczy model używać tych tokenów poprawnie w kontekście, ale jeśli _bardzo_ brakowałoby mu pewnych słów, można rozważyć rozszerzenie słownika tokenizera i ponowne przeliczenie embeddings (to jednak zaawansowane i nie zawsze konieczne).
- _Walidacja_: Monitoruj jakość modelu na zbiorze walidacyjnym. MLX-LM umożliwia ocenę perplexity: np. komendą `mlx_lm.lora --model <model> --adapter-path <adapter.safetensors> --data data_polish --test` obliczysz ppl na `test.jsonl`​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=To%20compute%20test%20set%20perplexity,use)

    . Sprawdzaj też ręcznie generowane odpowiedzi na polskie pytania przed i po fine-tune, aby ocenić poprawę.

Przykład z praktyki: Niklas Heidloff pokazał, że fine-tuning 7-miliardowego modelu Mistral na konkretne zadanie (generowanie zapytań SQL z jęz. naturalnego) z użyciem MLX może zająć <10 minut na MacBooku Pro z M3​

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

. To pokazuje potencjał – adaptacja modelu do nowego języka lub dialektu również może być wykonana szybko lokalnie. Po trenowaniu, wygeneruj kilka próbek: `mlx_lm.generate --model <model> --adapter-path <adapter.safetensors> --prompt "Witaj, czym się zajmujesz?"` aby sprawdzić, czy model płynnie odpowiada po polsku.

## Fine-tuning technologiczny (kod, medycyna, prawo itp.)

**Cel**: Dostroić model do **specyficznej domeny technicznej** – np. języka programowania (Python, Java, SQL), terminologii medycznej czy prawniczej, lub innej specjalizacji. Chodzi o to, by model lepiej rozumiał kontekst tych dziedzin i posługiwał się fachowym językiem (np. generował kod zgodny z konwencją, udzielał eksperckich porad medycznych zgodnych z terminologią, itp.).

**Dane treningowe**: Przygotuj dane typowe dla danej domeny:

- _Kod programistyczny_: Możesz użyć zbiorów Q&A dla programistów (np. fragmenty z Stack Overflow), dokumentacji API, przykładowych zadań kodowania z rozwiązaniami. Format danych: np. pary _pytanie -> kod_ lub _polecenie -> wygenerowany kod_. Dobrą praktyką jest fine-tuning na _promptach_ zawierających opis problemu i _completion_ będącym kodem. Możesz też użyć istniejących zbiorów jak HumanEval, a nawet po prostu fragmentów kodu z komentarzami. **Uwaga**: Jeśli model bazowy to ogólny LLM, rozważ użycie bazy wyspecjalizowanej (jak CodeLlama czy StarCoder) – już zawiera wiedzę o składni, co ułatwi dostrajanie do konkretnego stylu.
- _Medycyna_: Wykorzystaj np. zanonimizowane transkrypty rozmów lekarz-pacjent, dokumentację medyczną, opisy przypadków, publikacje w języku modelu. Format może być dialogowy (“Pacjent: opis objawów… Lekarz: diagnoza…”) albo QA (“Pyt: …? Odp: …”). Ważne jest zachowanie ostrożności co do jakości – błędne dane medyczne mogą sprawić, że model będzie halucynował niebezpieczne odpowiedzi. Dobrym podejściem jest _instruktażowe dostrajanie_ (instruct tuning): daj modelowi rolę specjalisty. Np. system message: _“Jesteś doświadczonym lekarzem…”_, użytkownik: _opis przypadku_, model: _szczegółowa odpowiedź medyczna_. Trenuj na takich scenariuszach, aby model nauczył się tonu i precyzji.
- _Prawo_: Podobnie, dane mogą obejmować fragmenty ustaw z objaśnieniami, pytania prawne i odpowiedzi, analizy prawnicze. Przydatne mogą być bazy Q&A prawników, dokumenty sądowe (jeśli dostępne publicznie) itp. Format: np. _“Pytanie: [opis sprawy]?” -> “Odpowiedź: [porada prawna z odniesieniem do przepisów]”_. Model po dostrojeniu powinien cytować artykuły prawa i posługiwać się formalnym stylem prawniczym.

**Wykonanie fine-tuningu**: Procedura jest analogiczna jak wyżej przy dostrajaniu językowym – używamy `mlx_lm.lora`. Przykład dla fine-tuning modelu pod generowanie kodu SQL z języka naturalnego (na wzór przykładu Apple):

bash

Copy

`mlx_lm.lora --model mistralai/Mistral-7B-Instruct-v0.2 \             --train \             --data data_sql \             --iters 600 \             --batch-size 4 \             --learning-rate 1e-4`

_(Powyższe zbliżone jest do instrukcji Apple MLX, gdzie w ciągu kilkuset iteracji dostrajano model Mistral 7B do tłumaczenia tekstu na zapytania SQL​_

_[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=Below%20are%20step%20by%20step,tune%20Mistral%20for%20SQL%20generation)_

_. Tu użyto stosunkowo niewielkiego LR, ponieważ model już miał pewne zdolności i nie chcemy ich nadpisać.)_

**Najlepsze praktyki**:

- **LoRA dla domeny**: Zdecydowanie zaleca się użycie LoRA również i tutaj, gdyż _fine-tuning domenowy_ często ma charakter **dodatku** do wiedzy modelu. Dzięki LoRA, model zachowuje oryginalne informacje, a adapter „wstrzykuje” wiedzę domenową kiedy potrzeba. Np. model medyczny nadal będzie miał ogólną wiedzę, ale w pytaniach medycznych użyje nowo wyuczonych szczegółów. LoRA chroni też przed _katastrofalnym zapominaniem_wcześniejszej wiedzy​

    [github.com](https://github.com/ml-explore/mlx-examples/issues/297# :~:text=%5BFeature%20Request%5D%20Full,phi2%20could%20cause%20some%20issues)

    .
- **Wybór modelu bazowego**: Używaj modeli już wstępnie ukierunkowanych, jeśli istnieją. Np. do kodu – weź CodeLlama czy inny model programistyczny jako bazę, by fine-tune skupił się na stylu/specjalizacji (np. CodeLlama już zna składnię Pythona, Ty dodasz wiedzę o Twojej specyficznej bibliotece). Do medycyny są open-source modele (np. ChatDoctor, czy Llama2 medyczny – jeżeli dostępne), które można dalej dostroić do węższego podzbioru (np. polskiej terminologii medycznej).
- **Pełny fine-tune vs. adapter**: Jeżeli zależy Ci, by model _tylko_ odpowiadał w wąskiej domenie (np. asystent prawny niezainteresowany innymi tematami), można rozważyć pełne dostrojenie wag (`--fine-tune-type full`). Wymaga to jednak dużo więcej pamięci i czasu. W praktyce LoRA często wystarcza – np. model Mistral 7B dostrojony LoRA na SQL radził sobie doskonale z tym zadaniem​

    [heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=Below%20are%20step%20by%20step,tune%20Mistral%20for%20SQL%20generation)

    . Wagi LoRA można po treningu „zespolić” z modelem bazowym poleceniem `mlx_lm.fuse` (co utworzy nowy model zawierający już wiedzę adaptera na stałe)​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Fuse)

    .
- **Trening warstwowy**: W MLX-LM można wskazać, ile ostatnich warstw modelu objąć dostrajaniem adapterów (`--num-layers`). Domyślnie LoRA trenuje wszystkie warstwy, ale jeśli masz mało danych, czasem lepiej trenować tylko najwyższe warstwy modelu (które najbardziej odpowiadają za specjalistyczne odpowiedzi). Ograniczenie LoRA do np. 8 ostatnich warstw zmniejszy ryzyko nadpisania fundamentalnej wiedzy i obniży koszty (mniej parametrów do nauczenia). W konfigurowalnym MLX można to uzyskać przez parametry YAML lub flagi (w najnowszych wersjach MLX-LM być może bezpośrednio flagą `--num-layers`)​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=%5B,file%20RESUME_ADAPTER_FILE)

    ​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=%5B,seed%20SEED)

    .
- **Kontrola jakości**: Testuj model pytaniami z danej domeny, ale również spoza niej. Sprawdź, czy wciąż potrafi odpowiedzieć na pytanie ogólne sensownie (jeśli zależy Ci na zachowaniu wszechstronności). Jeśli zauważysz tzw. overfitting (np. model medyczny zaczyna odpowiadać nazbyt encyklopedycznie na każde pytanie lub generuje w kółko te same formułki), rozważ _zmniejszenie learning rate_ lub _zastosowanie regularizacji_. Można np. dodać lekki **dropout** na adaptery podczas treningu (jeśli MLX to wspiera) lub ograniczyć liczbę iteracji.

**Przykład**: Fine-tuning techniczny w akcji to wspomniane wcześniej dostrojenie modelu do generowania kodu SQL. W publicznym przykładzie użyto Mistral-7B i zbioru WikiSQL, co pozwoliło modelowi nauczyć się składni SQL i relacji między pytaniem a zapytaniem​

[heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=Below%20are%20step%20by%20step,tune%20Mistral%20for%20SQL%20generation)

. Podobnie Ty możesz dostroić model do np. generowania kodu HTML z opisu, czy do odpowiadania językiem prawnym – kluczowe jest przygotowanie reprezentatywnych par **wejście -> pożądane wyjście** i zastosowanie powyższych zasad.

## Fine-tuning modeli CV i VL (Computer Vision & Vision-Language)

**Cel**: Adaptacja modeli wizji komputerowej (CV) oraz modeli multimodalnych (łączących wizję i język – VL) do konkretnych zadań. Może to obejmować: dostrojenie sieci klasyfikacyjnej do nowych kategorii obrazów, doszkolenie modelu detekcji obiektów na specyficzny typ obiektów, dostrojenie modelu generującego opisy obrazów (image captioning) do specyficznej stylistyki, czy modelu pytania-o-obraz (Visual QA) do węższej dziedziny.

**Wsparcie w MLX**: Framework MLX obsługuje także modele obrazowe. W repozytorium MLX-Examples znajdziemy m.in. przykłady trenowania ResNet na CIFAR-10 (klasyfikacja obrazów), generowania obrazów (Stable Diffusion, FLUX), a także modeli multimodalnych jak CLIP (połączenie obraz-tekst) czy LLaVA (Large Language and Vision Assistant)​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=Image%20Models)

​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=,and%20generation%20with%20%20108)

. To oznacza, że możemy wykorzystać MLX do trenowania/dostrajania modeli CV/VL lokalnie na Mac. Jeśli jednak wolisz użyć tradycyjnego PyTorch, to dzięki obsłudze MPS (Metal Performance Shaders) większość modeli wizji da się trenować na GPU Apple również spoza MLX.

**Przykład: Fine-tuning klasyfikatora obrazów**: Załóżmy, że masz model ResNet50 wytrenowany na ImageNet (1000 klas) i chcesz go dostroić do rozpoznawania np. gatunków grzybów na zdjęciach (załóżmy 20 nowych klas). Klasyczny przepis transfer learningu:

1. Zamień ostatnią warstwę (klasyfikator) na nową, z liczbą neuronów = 20 (liczba docelowych klas).
2. Zamroź początkowo większość wag sieci (wszystkie konwolucyjne warstwy) i trenuj tylko tę ostatnią warstwę przez kilka epok na swoim zbiorze (to nauczy nowy klasyfikator podstaw).
3. Następnie odmnróź stopniowo wyższe warstwy i kontynuuj trening z bardzo małym learning rate (np. 1e-5), by delikatnie dostosować wcześniej wyuczone filtry do nowych danych.
4. Użyj małego batch size (2-8), jeśli pamięć jest ograniczona, i rozważ augmentację danych (przekształcenia obrazów) by zrekompensować nieduży zbiór.

W MLX można taki trening przeprowadzić pisząc skrypt w stylu PyTorch/JAX – MLX udostępnia własne API tablic tensorowych, ale wspiera też interoperacyjność. Ewentualnie możesz posłużyć się Kerasem czy PyTorchem z backendem MPS. Apple dostarczało przykłady i filmy instruktażowe jak trenować modele na MPS (np. WWDC 2023 pokazało nowe udoskonalenia MPS pod kątem trenowania sieci transfomerowych i konwolucyjnych) – warto się upewnić, że masz **macOS >= 14** i aktualny **PyTorch > 2.0**, by mieć najlepszą wydajność i obsługę FP16.

**Fine-tuning modeli multimodalnych**: Przykład modelu VL to CLIP – sieć ucząca się przestrzeni wspólnej obrazów i opisów tekstowych. Załóżmy, że chcesz dostroić CLIP do lepszego kojarzenia opisów w języku polskim z obrazami określonego typu (np. zdjęcia medyczne z opisami). Masz pary obraz + opis. Możesz wczytać pretrenowany model CLIP (OpenAI lub inny), podmienić ewentualnie głowice liniowe, i trenować podobnie jak oryginalnie – minimalizując dystans między embeddingiem obrazu a tekstu dla prawidłowych par. MLX Examples posiada gotowy przykład dla CLIP​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=Multimodal%20models)

, z którego można skorzystać. Fine-tuning CLIP może wymagać więcej pamięci dla obrazów wysokiej rozdzielczości, więc rozważ zmniejszenie rozmiaru obrazu lub batch size.

Inny rodzaj modelu VL to LLaVA – model, który przyjmuje obraz + pytanie tekstowe i generuje odpowiedź (połączenie wizji z LLM). Dostrajanie takiego modelu wymaga danych multimodalnych (np. obraz z towarzyszącym pytaniem i idealną odpowiedzią). MLX-Examples wskazuje, że wsparcie dla LLaVA jest przewidziane​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=Multimodal%20models)

– zapewne można załadować odpowiedni model i trenować go podobnie jak LLM, dostarczając dodatkowo wejście obrazowe. Niestety MLX-LM (część dla LLM) nie obsługuje natywnie obrazów w promptach, więc tu raczej wchodzimy w niestandardowy kod. W takim przypadku można użyć bibliotek Hugging Face Transformers + VisionEncoderDecoderModel lub MultimodalPipeline, z backendem MPS.

**Najlepsze praktyki w CV/VL**:

- _Augmentacja_: zawsze gdy danych jest mało, stosuj augmentacje (obrót, skalowanie, cropy, zmiana koloru) by poprawić generalizację.
- _Mały learning rate_: szczególnie gdy model bazowy jest duży i wyspecjalizowany (np. ViT Huge od CLIP), używaj bardzo małych LR dla fine-tune, rzędu 1e-5 lub 5e-6, by nie zniszczyć pretrenowanych cech.
- _Warstwowe odmrażanie_: jak opisano wyżej – nie trenuj od razu wszystkich wag, tylko sukcesywnie od końca.
- _Sprawdzanie nadmiernego dopasowania_: monitoruj trening (utrata na treningu vs walidacji). Jeśli training loss spada, a val loss rośnie – model przeucza się. Zastosuj wcześniejszy stop (early stopping) lub silniejsze regularyzatory.
- _Ewaluacja_: Po dostrojeniu modelu CV, oceń dokładność na walidacji (accuracy/F1 dla klasyfikacji, mAP dla detekcji, etc.). Dla modeli multimodalnych – sprawdź kilkanaście przykładowych obrazów z pytaniami, porównaj odpowiedzi modelu z oczekiwanymi.

**Przykład**: Fine-tuning sieci ResNet50 do rozpoznawania zdjęć rentgenowskich pod kątem zapalenia płuc. Możemy wziąć model wstępnie nauczony na ImageNet, podmienić ostatnią warstwę (2 klasy: zapalenie/zdrowy) i trenować na datasetie X-Ray. Taka adaptacja zwykle znacząco podnosi wyniki w porównaniu do użycia modelu bez dostrajania. Apple MLX bez problemu poradzi sobie z takim zadaniem na M2/M3 (architektura ResNet50 nie jest zbyt wielka). Po treningu warto uzyskać ładne wyniki – np. dokładność >90% na zbiorze testowym.

## Fine-tuning modeli generatywnych (obrazy i teksty – np. Stable Diffusion, FLUX)

**Cel**: Optymalizacja modeli generatywnych do konkretnych zastosowań. W tej sekcji skupimy się głównie na **modelach tekst->obraz** jak Stable Diffusion (SD) czy nowszy model FLUX. Dostrajanie generatywne może dotyczyć również modeli tekstowych (LLM) – co już omówiliśmy – ale tu rozumiemy to jako ulepszanie zdolności generowania obrazów lub stylów, ewentualnie generowania tekstów przez modele sekwencyjne (np. GPT-2 fine-tune do generowania poezji).

**Stable Diffusion**: Model SD (np. wersja 1.5 czy SDXL 1.0) składa się z **U-Net** (model denoisingowy generujący obraz latenta), **tekstowego enkodera** (np. CLIP) i ewentualnie modelu autoenkodera (VAE) do zamiany latenta na obraz. Fine-tuning całego modelu SD jest bardzo kosztowny (dziesiątki milionów obrazów i duże zużycie VRAM). **W praktyce fine-tuning SD lokalnie polega na technikach typu DreamBooth lub LoRA dla Stable Diffusion**. DreamBooth pozwala dostroić model do generowania konkretnego obiektu/persony na zaledwie kilkunastu obrazach – ale pełny DreamBooth SD 1.5 na 10-20 obrazach może wymagać ~12-16 GB pamięci GPU. Dzięki zunifikowanej pamięci, Mac z 32GB+ RAM może to wykonać. Z kolei **LoRA dla SD** polega na trenowaniu niskowymiarowych adapterów w wagach U-Net (i opcjonalnie tekst-enkodera) zamiast pełnych wag – analogicznie jak w LLM.

Na Macach możemy użyć biblioteki Hugging Face Diffusers z włączonym MPS. Istnieją gotowe skrypty do DreamBooth i LoRA (np. `train_dreambooth.py` i `train_lora_dreambooth.py` w Diffusers). Trzeba je uruchomić wskazując model (Hugging Face repo) i folder ze zdjęciami + nazwą _tokenu_ (np. unikalne słowo). Przykład (Pseudo-kod, bo skryptowo jest bardziej złożone):

bash

Copy

`accelerate launch train_dreambooth.py \   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \   --instance_data_dir="data/moj_obiekt" --instance_prompt="zdjęcie mojego [unikalny_token]" \   --output_dir="sd_model_dostrojony" \   --train_batch_size=1 --gradient_accumulation_steps=1 --max_train_steps=800 \   --learning_rate=1e-6 --mixed_precision=fp16`

Tu `mixed_precision=fp16` jest ważne – dzięki temu połowa precyzji zmniejsza wymagania pamięci, co na M1/M2 ma znaczenie. Samo MLX ma w planach/wspiera generowanie obrazów (ma przykład Stable Diffusion), lecz aktualnie **fine-tuning SD w MLX** nie jest jeszcze plug-and-play tak jak LLM. Można jednak użyć MLX jako backendu niskopoziomowego (tzw. DiffusionKit oparty na MLX jest w fazie eksperymentalnej​

[huggingface.co](https://huggingface.co/argmaxinc/mlx-stable-diffusion-3-medium# :~:text=argmaxinc%2Fmlx,repository%21%20SD3%20Example%20Output)

).

**FLUX**: FLUX-1 to nowy otwarty model tekst->obraz, konkurencyjny wobec SDXL​

[medium.com](https://medium.com/@promptingpixels/a-stable-diffusion-users-guide-to-understanding-flux-1-fee1e77c28a1# :~:text=Medium%20medium,in%20quality%20and%20prompt%20adherence)

​

[ikomia.ai](https://www.ikomia.ai/blog/flux1-text-to-image-diffusion-model# :~:text=FLUX.1%20Text,coherent%20images%20from%20text)

. Jego dostrajanie przebiega podobnie. Jeśli dostępny jest checkpoint FLUX, można spróbować fine-tune np. do konkretnego stylu artystycznego. Proces analogiczny jak w SD: zbierasz kilkadziesiąt obrazów w danym stylu, ustawiasz unikalny identyfikator (np. `<style_token>`) i trenujesz model by kojarzył ten token z tym stylem. Rezultat – po dostrojeniu, model generując obraz z promptem zawierającym `<style_token>` będzie wytwarzał obrazy w tym stylu.

Niestety FLUX jest bardzo świeży – skrypty do jego trenowania mogą nie być dopracowane. Prawdopodobnie również wymaga GPU z dużą pamięcią. Mac Studio M2 Ultra z 128GB pamięci powinien pomieścić model ~7B parametrów U-Net w FP16, ale wydajność może być niższa niż dedykowanych kart (brak wyspecjalizowanych tensor cores). Mimo to, jest to wykonalne – MLX plus ewentualnie ANE (Neural Engine) mogą przyspieszyć pewne operacje.

**Najlepsze praktyki**:

- _LoRA do generatywnych_: zamiast pełnego fine-tune SD/FLUX, użyj LoRA. Istnieją narzędzia jak LoRA for Diffusers integrujące to. Adaptery zajmują mało miejsca i można je nakładać na różne modele. Np. możesz mieć osobny LoRA, który uczy SD stylu „impresjonizm” i osobny dla stylu „komiks” i w zależności od potrzeby je włączać.
- _Mały learning rate i ograniczona liczba kroków_: model generatywny szybko się nauczy nowych obrazów – łatwo jednak przesadzić, co skutkuje _overfittingiem_ (model pamięta dokładnie obrazy treningowe i je wstawia zamiast uogólnień). Dlatego często używa się **max 1000-2000 kroków** nawet dla kilkunastu obrazków (przy batch=1, grad acc 1). LR rzędu 1e-6 do 5e-6 by nie zniszczyć delikatnej równowagi wyuczonej na milionach obrazów.
- _Regularizacja DreamBooth_: W DreamBooth standardowo stosuje się zbiór obrazów regularizacyjnych (np. kilkaset zdjęć ogólnych), aby model nie nadpisał całej przestrzeni koncepcyjnej nowym obiektem. Na Macu to dodatkowy koszt (generowanie ich lub trzymanie w pamięci). Jeśli to trudne, spróbuj chociaż wygenerować ~100 obrazów _przed_ treningiem (używając pierwotnego modelu, z promptem ogólnym typu “photo of a person” bez unikalnego tokenu) i użyj ich jako regularization dataset.
- _Monitoring_: Niestety ocena jakości generowanych obrazów wymaga subiektywnej oceny. Podczas treningu warto co pewien czas wygenerować obraz kontrolny z tym samym promptem i obserwować postępy (w Diffusers są skrypty logujące próbki). W ten sposób upewnisz się, że model np. nauczył się postaci, ale nadal poprawnie generuje tło itp.
- _Eksport modelu_: Po dostrojeniu możesz chcieć używać modelu poza treningiem. Apple MLX umożliwia eksport do CoreML czy formatów .mlmodel, ale obecnie dla Stable Diffusion często używa się gotowych narzędzi (Core ML Stable Diffusion). Pamiętaj, że CoreML w iOS/macOS ma ograniczenia co do rozmiaru modelu – duże modele mogą wymagać segmentacji lub używania Mac tylko jako serwera.

**Przykład**: Załóżmy, że chcesz generować obrazy produktowe twojej firmy w stylu spójnej grafiki. Fotografujesz 20 produktów, dostrajasz lokalnie Stable Diffusion, by nauczył się loga/kształtu tych produktów poprzez DreamBooth. Po treningu (powiedzmy 1000 iteracji na MacBooku Pro z M3 Max – co może zająć kilkanaście minut) model jest w stanie na żądanie wygenerować nowe ujęcia produktu w różnych sceneriach, w wysokiej rozdzielczości. Takie zastosowanie jest możliwe **lokalnie** dzięki MLX/MPS, choć jeszcze niedawno wymagało drogiego sprzętu w chmurze.

## Fine-tuning STT i TTS (rozpoznawanie i generowanie mowy)

**Cel**: Nauka modeli Speech-to-Text (zamiana mowy na tekst) oraz Text-to-Speech (zamiana tekstu na mowę) w specyficznych warunkach. Przykłady: dostrojenie modelu ASR do lepszego rozpoznawania polskiego slangu lub gwar, dostrojenie modelu TTS tak, by mówił głosem konkretnej osoby lub z danym akcentem.

**Speech-to-Text (ASR)**: Najbardziej znanym otwartym modelem STT jest **Whisper** od OpenAI. Dostępny jest w różnych rozmiarach (Tiny, Base, Small, Medium, Large). Fine-tuning Whispera na własne dane jest możliwy – Hugging Face udostępnia przewodniki krok po kroku​

[huggingface.co](https://huggingface.co/blog/fine-tune-whisper# :~:text=In%20this%20blog%2C%20we%20present,dataset%20using%20Hugging%20Face%20Transformers)

. Na Macu proces ten jest wymagający, ale możliwy dla mniejszych modeli (Tiny/Base) nawet na 8-16GB RAM, a dla większych (Medium, Large) na 32GB+. MLX-Examples zawiera przykład modelu Whisper (zapewne inference)​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=Audio%20Models)

, jednak do treningu możemy skorzystać z Transformers.

**Przygotowanie danych ASR**: Potrzebujesz nagrań audio i odpowiadających im transkrypcji tekstowych. Powinny być w formacie np. JSON lub CSV: plik audio + tekst. Możesz użyć bibliotek `datasets` by załadować standardowe zbiory (Common Voice, corpora call center itp.). Ważne: dopasuj częstotliwość próbkowania do tej, na której działa model (Whisper używa 16kHz mono). Dane powinny być przycięte do rozsądnej długości (Whisper przyjmuje ~30 sekund na segment).

**Fine-tuning Whisper**: Używając Transformers, konfigurujesz `WhisperForConditionalGeneration` i `WhisperProcessor`. Możesz zamrozić część encodera audio i trenować głównie dekoder językowy jeśli tylko język dodajesz – ale w przypadku akcentów raczej trenuj całość z niskim LR. Ustaw `processor.feature_extractor` i `tokenizer` odpowiednio (np. tokenizer już ma polski w wersji multi). Następnie zwykła pętla treningowa lub `Trainer` z HF. Istotne hiperparametry:

- batch size – dostosuj do pamięci (może być nawet 4,8 jeśli audio ~10s).
- learning rate – raczej niski, ~1e-5, bo modele ASR są wrażliwe.
- gradient checkpointing – zdecydowanie tak, jeśli model duży (Whisper Large), by zmieścić w pamięci​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=,The%20PRNG%20seed)

    ​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=,4)

    .
- mixed precision – FP16 może nie działać stabilnie z wszystkich względów na CPU, ale na GPU Apple od macOS 14 jest częściowe wsparcie BF16/FP16, warto spróbować aby przyspieszyć.

Po dostrojeniu, model powinien lepiej rozpoznawać specyficzne słownictwo. Np. jeżeli trenujemy na nagraniach medycznych, nauczy się poprawnie zapisywać nazwy leków czy skróty.

**Text-to-Speech (TTS)**: Otwartych modeli TTS też jest kilka (Tacotron 2, FastSpeech, VITS, Coqui TTS). Fine-tuning TTS często sprowadza się do **voice cloning** – mając np. 5-30 minut nagrań mowy danej osoby, chcemy dostroić model, by na ich podstawie generował dowolny tekst jej głosem.

**Przygotowanie danych TTS**: Potrzebne są pary _tekst – audio_. Np. zdania i odpowiadający im plik dźwiękowy wypowiedziany przez mówiącego. Jakość i czystość dźwięku jest krytyczna. Często standaryzuje się audio (mono, 22kHz lub 16kHz) i normalizuje głośność.

**Fine-tuning**: Weźmy Tacotron 2 (architektura sekwencyjna, encoder-decoder) i HiFi-GAN jako vocoder. Można wykorzystać pretrenowany model na uniwersalnym głosie i dostroić go. Proces jest podobny do trenowania od zera, ale z mniejszą liczbą epok. Na Macu GPU poradzi sobie z Tacotronem, bo to model ~50M parametrów – większe wyzwanie to vocoder, ale HiFi-GAN też jest ~15M, więc OK.

Trenujemy najpierw model mel-spectrogram (Tacotron) – tu ważne by szybko nie przeuczyć na limitowanym głosie: użyj _stopniowego zamrażania_ (początkowo zamroź prenauczony encoder tekstu, trenuj tylko decoder by nauczył się akcentu i mel-ody, potem ewentualnie delikatnie poducz encoder). Użyj małego LR, rzędu 1e-4 lub mniej. Po kilkuset iteracjach model zacznie brzmieć jak docelowy głos na danych treningowych.

Następnie vocoder: można spróbować dostroić uniwersalny HiFi-GAN do konkretnego głosu (kilka epok na próbkach), choć wiele osób pomija ten krok – uniwersalny vocoder zwykle działa wystarczająco dobrze. Jeśli jakość barwy nie jest zadowalająca, fine-tune vocodera.

**Najlepsze praktyki TTS**:

- _Alignment_ w modelach autoregresywnych może się zepsuć przy fine-tune na małym zestawie (Tacotron może zacząć literować lub gubić fragmenty). Aby temu zapobiec, często **zamraża się większość modelu i uczy tylko embeddingi postaci** (jeśli model wspiera wielu mówców) lub tylko ostatnie warstwy dekodera mowy. Alternatywnie, używa się modeli one-shot jak VITS, które mogą szybciej dostroić głos.
- _Dataset augmentation_: Możesz _wydłużyć_ trochę dataset przez drobne modyfikacje audio (np. minimalnie zmienić wysokość dźwięku, dodać szum tła) – ale ostrożnie, by nie zniekształcić głosu.
- _Sprawdzenie wymowy_: Po dostrojeniu, daj modelowi trudne zdania, by zobaczyć czy nie gubi intonacji. Często fine-tuned voice może brzmieć dobrze na prostych zdaniach, a łamać się na dłuższych – w razie czego trenuj odrobinę dłużej lub na bardziej zróżnicowanych frazach.

**Przykład**: Dostrajanie TTS – masz model Coqui TTS (multispeaker) i chcesz dodać nowego lektora. Nagrano 100 zdań jego głosem (~10 minut audio). Po oczyszczeniu danych i dostrojeniu modelu (powiedzmy 5000 kroków na M2 Ultra, co może potrwać kilka godzin), otrzymujesz syntezator mowy mówiący niemal jak oryginał. Takie eksperymenty społeczność robiła na PC; na Apple Silicon też są możliwe, choć nieco mniej udokumentowane – ale hardware (w szczególności M2 Ultra) ma dość mocy, by je wykonać.

## Dostrajanie sieci konwolucyjnych (np. U-Net, ResNet, GAN)

Ta kategoria częściowo pokrywa się z poprzednimi (CV), ale skupmy się na technikach specyficznych dla architektur konwolucyjnych:

- **U-Net**: Popularna architektura do segmentacji oraz generowania obrazów (stosowana np. w Stable Diffusion jako część denoisera). Fine-tuning U-Net do segmentacji oznacza, że mamy model (np. U-Net z pretrenowanym encoderm z ImageNet) i chcemy go nauczyć segmentować nowe dane (np. wykrywać guz na obrazach MRI). Proces: podmieniamy warstwę wyjściową (na odpowiadającą liczbie segmentowanych klas), ewentualnie dostosowujemy rozmiar wejścia, i trenujemy na parach obraz+maska. Jeśli mamy pretrenowany encoder (np. ResNet w U-Net), to jak poprzednio – zamrażamy go na początku. Dostrajanie U-Net w segmentacji medycznej często drastycznie poprawia wyniki w porównaniu z modelami uczonymi od zera, zwłaszcza gdy dataset jest mały​

    [arxiv.org](https://arxiv.org/html/2404.09957v1# :~:text=,tuning%20strategies)

    . W badaniach zauważono np., że dostrojenie modelu Segment Anything (który jest formą dużego U-Net) do konkretnych danych medycznych daje lepsze wyniki niż użycie go prosto „z pudełka”​

    [arxiv.org](https://arxiv.org/html/2404.09957v1# :~:text=,tuning%20strategies)

    .

- **ResNet (inne CNN)**: Klasyczne podejście transfer learningu już omówiliśmy. Dodajmy, że MLX-LM czy raczej ogólnie MLX radzi sobie także z trenowaniem modeli takich jak ResNet na GPU Apple. W repo MLX-Examples jest przykład na CIFAR-10, gdzie od zera uczony jest ResNet – to oznacza, że **fine-tuning** (który jest lżejszy niż trening full) tym bardziej jest wykonalny. Warto wspomnieć, że Apple Silicon posiada akcelerację konwolucji i już od M1 można osiągać całkiem dobre czasy treningu CNN (choć do topowych GPU trochę brakuje). Z praktyki: model ResNet50 trenowany na Apple M1 osiągał ~140 obrazów/s na 224x224 (dane z wczesnych testów MPS), co jest porównywalne z mobilnymi GPU. Na M2/M3 powinno być jeszcze szybciej.

- **GAN (Generative Adversarial Networks)**: Fine-tuning GAN-ów to specyficzny temat. Możemy np. wziąć StyleGAN2 wytrenowany na twarzach i dostroić go do generowania np. postaci z anime. Tutaj fine-tuning polega na dalszym treningu generatora (i ewentualnie dyskryminatora) na nowym zbiorze. Zaleca się **freeze dyskryminator** na parę początkowych epok, by generator miał szansę dostosować się bez natychmiastowej surowej oceny. Alternatywnie, stosuje się metodę **transfer GAN** – inicjalizujemy model wagami z podobnej dziedziny i trenujemy z mniejszym LR oraz mniejszą regularizacją na nowy dataset. Napotkać można problemy jak _mode collapse_ gdy dataset jest mały. Rozwiązania to: augmentation, regularizacja (np. gradient penalty), mieszanie z cząstką starego datasetu.

Trenowanie GAN-a jest ciężkie obliczeniowo, ale modele typu StyleGAN2 (ok. 30M parametrów) mogą trenować ~ kilka iteracji/s nawet na MacBookach Pro. Dostrajanie raczej nie zajmie setek epok – zwykle kilkadziesiąt wystarczy, co na sprzęcie Apple może zająć kilkanaście godzin w zależności od rozmiaru danych.

**Najlepsze praktyki (CNN/GAN)**:

- _Wykorzystanie pretreningu_: Nigdy nie trenuj od zera, jeśli możesz tego uniknąć. Zawsze bierz wagi z pokrewnego zadania. Np. do segmentacji medycznej – encoder U-Net z ImageNet, do detekcji – YOLO wstępnie nauczone na COCO, do generowania twarzy anime – StyleGAN wytrenowany na twarzach rzeczywistych.
- _Mały krok optymalizatora_: Zwłaszcza dla GAN – bardzo mały LR (np. 1e-6 dla generatora podczas fine-tune) może zapobiec nagłym skokom prowadzącym do kolapsu.
- _Zapisywanie checkpointów_: Fine-tuning może pójść w złą stronę (zwłaszcza GAN) – dlatego częsty checkpoint (np. co 5 epok) pozwoli Ci wrócić do poprzedniej stabilnej wersji, gdy nowa zacznie generować bzdury. MLX-LM ma flagę `--save-every` do zapisu adapterów co X kroków​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=%5B,file%20RESUME_ADAPTER_FILE)

    . W własnych pętlach treningowych również zapisuj.
- _Ewaluacja subiektywna_: Dla generatywnych modeli obrazowych trudno o jedną metrykę. Można użyć FID (Frechet Inception Distance) dla oceny jakości i różnorodności generowanych obrazów względem oryginalnego zbioru – jeśli rośnie, to model się pogarsza (przeuczenie, spadek różnorodności); jeśli spada – jest lepiej.
- _Wykorzystanie narzędzi_: Są dedykowane biblioteki do fine-tune GAN, np. DiffusionBee dla stable diffusion (UI na Mac), lub Nvidia StyleGAN toolkit (wymaga CUDA – tu niedostępne, ale niektóre porty chodzą na TensorFlow-Metal). Czasem społeczność portuje te narzędzia na MPS – warto sprawdzić GitHuba.

**Przykład**: Dostrajanie U-Net: Weźmy model **Segment Anything (SAM)**, który w trybie fine-tune (np. wersja z drobnym dostrojeniem) ma być użyty do segmentacji konkretnych struktur anatomicznych w zdjęciach z tomografu. Badania pokazują, że fine-tuning SAM na takim zadaniu daje przewagę nad użyciem go bezpośrednio​

[arxiv.org](https://arxiv.org/html/2404.09957v1# :~:text=,tuning%20strategies)

. W praktyce oznacza to: bierzemy podstawowy model, dokładamy warstwę wyjściową pod nasze klasy segmentacji, trenujemy na kilkuset obrazach medycznych z maskami. Model po dostrojeniu może osiągnąć np. IoU = 0.9 na klasie, podczas gdy pretrenowany SAM miał np. 0.85. Taka różnica bywa kluczowa w zastosowaniach medycznych (mniej błędów modelu). Dzięki MLX, cały ten proces mógłby zostać wykonany na lokalnej stacji roboczej (np. Mac Studio), co jest istotne gdy dane medyczne nie mogą opuszczać placówki.

## Fine-tuning modeli detekcyjnych i segmentacyjnych (wizja specjalistyczna)

Ta sekcja skupia się na **modelach do detekcji obiektów** (np. YOLOv5, Detectron2/Mask R-CNN, DETR) oraz **segmentacji obrazów** (U-Net już był, ale są też FPN, DeepLab, itd.), szczególnie w wyspecjalizowanych domenach jak medycyna, geografia, przemysł.

**Detekcja obiektów**: Fine-tuning modelu detekcji to właściwie standard przy zastosowaniu na własnym zbiorze. Jeśli masz np. model YOLOv8 nauczony na COCO (80 klas ogólnych), a chcesz wykrywać tylko wady produktów na taśmie produkcyjnej, to dostrajasz go na swoim zbiorze (np. 2 klasy: „dobry” vs „uszkodzony” + bounding box). Wymaga to oznaczonych obrazów (najlepiej kilkaset lub więcej). Procedura:

- Wymień warstwę wyjściową (detektor klas) na nową z liczbą neuronów = liczba nowych klas _plus tło_.
- Zamroź backbone (np. CSP-Darknet czy Swin-T w DETR) na kilka pierwszych epok – trenuj tylko głowicę detekcyjną.
- Potem pozwól uczyć się też backbone z niskim LR.
- Używaj **schedulerów** LR: detekcja często korzysta z cyklicznych lub schodkowych LR – to pomaga dopaść lepsze minima.
- **Augmentacje mosaikowe**: szczególnie YOLO używa mosaic data augmentation (łączenie 4 obrazów w jeden) – to zachowaj, bo zwiększa różnorodność.

Na Macu możesz trenować YOLOv5/v8 używając PyTorch+MPS (repo YOLO by Ultralytics po drobnych modyfikacjach wspiera MPS). Model nie jest wielki (to kilka dziesiątek milionów parametrów), ale intensywnie korzysta z operatorów (augmentacje, anchor computation) – CPU też będzie pracować. M2 Ultra z większą liczbą rdzeni CPU/GPU przyspieszy ten pipelin.

**Segmentacja specjalistyczna**: Poza wspomnianym U-Net, modele jak DeepLabv3 (z backbone ResNet) są często używane. Fine-tuning wygląda podobnie do klasyfikacji: nowa warstwa wyjściowa (maski w liczbie klas), freeze backbone, itp. W medycynie czasem segmentuje się struktury, gdzie dostępne są pretrenowane modele np. z konkursów (liver tumor segmentation etc.) – wykorzystaj te wagi, to skróci trening.

Ciekawym przypadkiem są **segmentatory zero-shot** jak Segment Anything Model (SAM). SAM potrafi segmentować dowolny obiekt po wskazaniu punktu – ale w środowisku specjalistycznym (np. komórki w obrazie mikroskopowym) może nie radzić sobie idealnie. Z literatury wynika, że dostrajanie SAM (nawet parametry efektywne jak LoRA) do danego zadania poprawia wyniki​

[ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/10847777/# :~:text=Stitching%2C%20Fine,a%20fully%20supervised%20manner%2C)

​

[ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/10847777/# :~:text=Segment%20Anything%20Model%20%28SAM%29%20fine,a%20fully%20supervised%20manner%2C)

. W MLX-Examples widzimy przykład SAM​

[github.com](https://github.com/ml-explore/mlx-examples# :~:text=Multimodal%20models)

, co sugeruje, że można taki fine-tune przeprowadzić (pewnie potrzebna jest spora pamięć, bo SAM ViT-H jest duży).

**Najlepsze praktyki**:

- _Balans danych_: Jeśli fine-tuning detektora przebiega na zbiorze z inną proporcją klas niż oryginał, upewnij się, że **samplowanie** podczas treningu jest zbalansowane. Np. model COCO widział głównie ludzi i auta, a Twój zbiór ma 90% przypadków klasy "uszkodzony" i 10% "dobry" – bez odpowiedniego ważenia, model może nauczyć się ignorować tę rzadszą klasę.
- _Metryki_: Dla detekcji – mean Average Precision (mAP) jest standardem. Sprawdzaj mAP na walidacji co kilka epok (często biblioteki detekcyjne mają wbudowane logowanie mAP). Dla segmentacji – IoU/Dice dla każdej klasy. Te miary powiedzą Ci, czy fine-tune poprawia się.
- _Wczesny stop dla detekcji_: Modele detekcyjne lubią duże czasy treningu, ale jeśli Twój zbiór jest mały, uważaj by nie przeuczyć. Gdy mAP przestaje rosnąć przez X epok, przerwij – dalsze kręcenie może zacząć powodować wzrost false positive/negative na val.
- _Kontrast tła_: W specjalistycznych obrazach często _tło_ bywa monotonne (np. zdjęcia z mikroskopu – czarne tło, jasne obiekty). Model może zacząć zbyt mocno polegać na kontekście tła. Dobra praktyka: dostarczyć także trochę _negatywnych przykładów_ (obrazy gdzie nie ma obiektu, albo losowe inne obiekty), by model nauczył się, że nie zawsze coś jest do wykrycia.
- _Parameter Efficient Tuning_: Rozważ użycie metod takich jak **LoRA** czy **adaptery** także w modelach wizji. Np. dla DETR można by pokusić się o LoRA w warstwach transformera dekodera. Choć to mniej rozpowszechnione niż w NLP, pojawiają się badania adaptujące LoRA do CNN (czasem nazywane Adapters w ResNetach). Zaleta – mniejsze zużycie VRAM i możliwość łatwego wycofania zmian.

**Przykład**: Fine-tuning detekcji – firma chce wykrywać na zdjęciach z drona czy pola uprawne mają oznaki chorób roślin. Mają tylko 500 zdjęć z zaznaczonymi chorymi miejscami. Biorą model DETR pretrenowany na COCO, dokładają jedną klasę "chora_plama", dostrajają na swoich 500 obrazach (Mac Studio M2 Ultra robi to w sensownym czasie dzięki temu, że _unified memory_ 64GB mieści cały obraz i model naraz). Po treningu model osiąga mAP ~50% dla tej klasy – co może nie wydaje się wysokie, ale znacznie przewyższa 0% sprzed dostrojenia. Dodatkowo, segmentacja chorych obszarów (np. U-Net dostrojony do masek chorób) może dawać IoU 0.6-0.7, co już bywa użyteczne w oszacowaniu skali problemu.

## Porównanie MLX-LM z innymi narzędziami na Apple Silicon

Ekosystem Apple Silicon oferuje kilka podejść do lokalnego trenowania modeli:

- **MLX-LM (Apple MLX)** – dedykowany framework od Apple Research, zoptymalizowany pod GPU Apple i unified memory. Zapewnia prosty interfejs CLI i Python do generacji oraz dostrajania LLM (i nie tylko). Wspiera LoRA, QLoRA, gradient checkpointing out-of-the-box​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Fine)

    ​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=,The%20PRNG%20seed)

    . Został zaprojektowany z myślą o efektywności i wygodzie badaczy. W praktyce MLX-LM osiąga imponujące wyniki na Macach – np. fine-tuning modelu 7B LoRA może iść ~250 tokenów/s na M1 Max 32GB, a ~475 tokenów/s na M2 Ultra​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=,second%20on%20an%20M2%20Ultra)

    ​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=,second)

    . To ok. 2-3x wolniej niż topowa karta NVIDIA, ale **bez ograniczeń VRAM** i przy mniejszym poborze mocy​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=480w%20to%20300w%20it%20slows,what%20guys%20from%20MLX%20did)

    ​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=I%20wouldn%27t%20expect%20the%20training,want%20to%20support%20mlx%20framework)

    . Kluczową zaletą MLX-LM jest to, że _pozwala trenować modele, które nie zmieściłyby się w typowej 24GB VRAM karty – korzystając z np. 64-128GB unified memory_​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=Me%20neither,per%20watt%2C%20which%20is%20surprising)

    . Dzięki temu możliwe jest np. QLoRA 70B na Mac Studio, czego nie zrobimy na pojedynczej karcie 24GB (trzeba by 2-4 GPU w PC).

- **PyTorch + MPS** – tradycyjny sposób, jaki znaliśmy zanim pojawił się MLX. Apple od kilku lat rozwija backend MPS w PyTorch, umożliwiając korzystanie z GPU w kodzie treningowym. Wiele skryptów z minimalnymi modyfikacjami zadziała na Mac (przekierowanie tensora na urządzenie "mps" zamiast "cuda"). Zaleta: nie trzeba uczyć się nowego narzędzia, używamy ekosystemu PyTorch/Hugging Face, który znamy. Wadą była historycznie pewna niestabilność i braki (np. brak wsparcia FP16, wolniejsze operacje atencji). Ale od macOS 14 i PyTorch 2.1 sytuacja się poprawiła – doszły natywne kernele attention, wsparcie FP16/BF16 w GPU i sporo optymalizacji. Mimo to, **MLX często bywa szybszy** i bardziej dopracowany pod kątem LLM, bo jest pisany pod Apple Silicon bez kompromisów (korzysta z niskopoziomowych optymalizacji Metal). Porównania wskazują, że MLX potrafi przyspieszyć serwowanie modelu o kilkadziesiąt procent względem czystego PyTorch MPS​

    [ai.gopubby.com](https://ai.gopubby.com/accelerating-hugging-face-pre-trained-models-on-apple-silicon-using-mlx-lm-and-mps-eb7465e4f502# :~:text=Apple%E2%80%99s%20M1%20and%20M2%20chips,ANE%29%20and%20Metal%20GPU)

    ​

    [ai.gopubby.com](https://ai.gopubby.com/accelerating-hugging-face-pre-trained-models-on-apple-silicon-using-mlx-lm-and-mps-eb7465e4f502# :~:text=designed%20specifically%20for%20Apple%20Silicon%2C,ANE%29%20and%20Metal%20GPU)

    . Również zużycie pamięci bywa lepiej zarządzane.

- **Ollama/MLC** – Ollama to narzędzie do lokalnego uruchamiania i zarządzania modelami (używa backendu llama.cpp i MPS). Ono jednak koncentruje się na _infernencji_ (generowaniu), a nie trenowaniu. Były eksperymenty, czy Ollama może użyć swoich mechanizmów do fine-tune, ale obecnie to nie jest wspierane. **MLC (Machine Learning Compilation)** to projekt od MLC.ai, który pozwala kompilować modele (np. LLM) do różnych platform (w tym Apple ANE – Neural Engine). MLC świetnie sprawdza się do uruchamiania modeli (np. ChatGPT w iPhone!), jednak do trenowania LLM raczej się nie stosuje. Niemniej, w przyszłości może pojawić się synergia – np. wytrenować model w MLX, a używać go przez skompilowany runtime MLC dla maksymalnej wydajności na iPhonie.

- **Hugging Face Accelerate/Transformers** – to w zasadzie nakładka na PyTorch, więc dużo tu zależy od MPS. Można użyć `Accelerate` do rozproszonego trenowania na CPU+GPU (np. pipeline offload – część na CPU, część na GPU), co teoretycznie pozwoliłoby trenować jeszcze większe modele niż zmieści unified memory. Jednak komunikacja CPU-GPU może spowalniać. MLX zamiast tego korzysta z unified memory bezobsługowo.

- **Core ML Tools** – Apple dostarcza CoreML głównie do inference (np. w aplikacjach). Trening w CoreMLGraph jest możliwy dla prostszych modeli, ale brak otwartych narzędzi do trenowania dużych sieci w CoreML. Więc porównanie tu w zasadzie: MLX vs PyTorch.

### Techniki dostrajania i optymalizacji

Niezależnie od narzędzia, warto stosować pewne techniki przy dużych modelach:

- **LoRA (Low-Rank Adaptation)**: Jak już szczegółowo opisano, to metoda pozwalająca trenować tylko _dodatkowe macierze niskiego rzędu_ przy każdej warstwie, zamiast modyfikować pełne wagi​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=Fine)

    . Ogromnie redukuje to wymagania pamięci i zwykle przyspiesza trening. MLX-LM ma LoRA jako domyślny tryb fine-tune i zaleca go dla modeli 7B+​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=)

    . Inne platformy (PyTorch+HF PEFT) również wspierają LoRA – to de facto standard w fine-tune LLM.
- **QLoRA (Quantized LoRA)**: To ulepszenie – model bazowy trzymamy w formacie 4-bit, co **czterokrotnie zmniejsza** zużycie pamięci, a uczymy LoRA w normalnej precyzji. MLX-LM automatycznie przełącza się na tryb QLoRA, gdy podasz model już zquantyzowany (np. 4bit)​

    [github.com](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md# :~:text=mistralai%2FMistral)

    . Dzięki temu nawet 70B model można trenować na Macu z odpowiednią pamięcią. W praktyce ludzie raportują, że 70B QLoRA na 128GB unified memory jest wykonalne – Apple Silicon przydziela wtedy ~120GB RAM GPU, co może spowolnić (bo część idzie do RAM, a nie VRAM), ale nadal działa. Inne narzędzia również obsługują QLoRA (biblioteka PEFT w HF Transformers).
- **Gradient Checkpointing**: Technika programowa redukująca zużycie pamięci przez _nierejestrowanie gradientów_dla części warstw w forward pass – zamiast tego, obliczamy je ponownie w trakcie backpropagacji​

    [github.com](https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/master/mlx-book/appendix/gradient-checkpoint/Gradient-Checkpoint.md# :~:text=Gradient%20checkpointing%20in%20MLX%20introduces,memory%20is%20at%20a%20premium)

    ​

    [github.com](https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/master/mlx-book/appendix/gradient-checkpoint/Gradient-Checkpoint.md# :~:text=This%20is%20essentially%20what%20gradient,which%20require%20memory%20to%20store)

    . To trade-off: więcej obliczeń, mniej pamięci. MLX-LM wspiera to flagą `--grad-checkpoint`​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=,The%20PRNG%20seed)

    . Szczególnie przydatne przy pełnym fine-tune dużego modelu lub trenowaniu w wysokiej rozdzielczości (np. Stable Diffusion 1024x1024). Inne frameworki też to mają (w PyTorch ustawiasz `model.gradient_checkpointing_enable()`). Różnica potrafi być kluczowa – np. model, który zajmował 20GB bez checkpointingu, może wymagać tylko 12GB z nim, kosztem spadku szybkości o ~20%. Na Macach, gdzie pamięć jest współdzielona z systemem, warto to włączyć, aby uniknąć przesiągnięcia do swapu.
- **Mixed Precision**: Używanie 16-bitowej precyzji (FP16) lub mieszanki FP16/FP32 (mixed) podczas treningu. Apple GPU od M2 wspierają szybkie obliczenia w FP16, a nawet BF16. MLX prawdopodobnie z tego korzysta domyślnie (choć dokumentacja nie wspomina, można założyć że tak, bo inaczej trudno byłoby zmieścić modele). W PyTorch na MPS do niedawna FP16 nie dawało przyspieszenia z powodu braku natywnego wsparcia – to się zmienia, testuj najnowsze wersje. Ogólnie _mixed precision_ jest standardem – pozwala trenować szybciej i z mniejszym zużyciem pamięci, często bez wpływu na jakość (czasem wymaga to techniki _gradient scaling_, ale biblioteki robią to automatycznie). Zalecane jest **always try FP16**, a jeśli wystąpią niestabilności (np. NaN w gradientach), wtedy przełączyć się na 32-bit dla wrażliwych części modelu.
- **Dora / ORCA / inne**: W kontekście MLX-LM pojawia się termin **DoRA** (prawdopodobnie skrót od _Dropout-Aided LoRA_ lub _Partial LoRA_). Wg dokumentacji MLX obsługuje tryb `--fine-tune-type dora`​

    [technovangelist.com](https://technovangelist.com/notes/finetuning-with-mlx# :~:text=%E2%80%94fine,it%20to%20full%20or%20dora)

    . Ze wzmianek wynika, że DoRA może oznaczać trenowanie tylko wybranych fragmentów modelu (np. tylko macierzy Q i V w self-attention zamiast pełnych Q,K,V) – co further zmniejsza parametry do nauczenia. W cytowanej dyskusji reddit zauważono, że MLX przykład QLoRA trenował tylko bloki Q i V, pomijając K i inne, co potencjalnie wpływa na jakość​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=their%20lower%20power%20consumption)

    . Technika ta to kompromis: jeszcze mniej pamięci, jeszcze szybsze trenowanie, ale ryzyko spadku jakości (user stwierdził dosadnie: _"Resulting finetunes ... will be shitty"_ jeśli trenujemy tak ograniczony podzbiór wag​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18yz8kc/mlx_supports_qlora_now/# :~:text=their%20lower%20power%20consumption)

    ). Dlatego DoRA raczej eksperymentalnie – można ją wykorzystać, gdy absolutnie nie mamy mocy na pełne LoRA, a chcemy sprawdzić jakikolwiek efekt.

### Hiperparametry vs rozmiar modelu i sprzęt

Dobór hiperparametrów (batch size, learning rate, liczba iteracji, itp.) musi uwzględniać zarówno **wielkość modelu**, jak i **posiadany hardware**:

- **Małe modele (3B-7B)**: Te modele (np. Mistral 7B, GPT-J 6B, LLaMA2 7B) są stosunkowo lekkie. Na Macu z 32GB RAM można je nawet **trenować w pełni** (bez LoRA) w FP16, choć z małym batch. Jeśli masz MacBook Air M1/M2 (8–16GB), zalecamy użycie LoRA i batch 1. Przykład: użytkownik z MacBook Air M2 24GB z sukcesem trenował LoRA na Mistral 7B, musząc zejść do batch size 1 i ograniczyć liczbę warstw LoRA do 4, ale model się trenował​

    [reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/# :~:text=)

    . LR dla takich modeli zazwyczaj może być nieco wyższy (bo mniej warstw = mniejsze ryzyko niestabilności) – np. 3e-4 do 5e-4 przy LoRA. Liczba iteracji: zależy od zbioru – ale np. ~1000 może wystarczyć dla kilkuset przykładów (zawsze patrz na zbieżność loss). Na wolniejszym sprzęcie (MBA) można skracać iteracje dla oszczędności czasu i sprawdzać efekt, ewentualnie robić więcej krótszych treningów iteracyjnie.

- **Średnie modele (13B-34B)**: Tutaj wchodzimy w zakres, gdzie **M2 Pro/Max z 32GB może mieć problem z pełnym fine-tune** – zaleca się LoRA/QLoRA. Mac Studio z M1 Ultra (tj. 20 CPU, 48 GPU, 128GB) lub M2 Ultra (24 CPU, 76 GPU, do 192GB) – to idealne maszyny dla tych modeli. Dla 13B LoRA w 4-bit, 64GB pamięci starczy z zapasem; można pokusić się nawet o batch 4. Dla ~30B modeli (np. LLaMA1 30B, LLaMA2 34B), 64GB pamięci może wymagać QLoRA żeby się zmieścić, albo gradient checkpointingu intensywnie. Na 128GB Ultra – zmieści się prawdopodobnie 30B w 16-bit LoRA. Learning rate raczej niższy dla większych modeli – np. 1e-4 lub 2e-4 maks. Duże modele łatwiej się „rozstrajają” jak dostaną zbyt duży krok. Co do iteracji – duży model może potrzebować ich nieco mniej by się nauczyć (ma więcej parametrów, szybciej dopasuje nawet mały zbiór), ale to też oznacza że może _przeuczyć się szybciej_. Dlatego uważnie obserwuj loss/val.

    - Na M2 Ultra można spróbować **pełnego fine-tune 13B** (bez LoRA). Upewnij się, że masz wtedy gradient checkpointing i może batch 1. Będzie to wolne, ale wykonalne – tu 76-rdzeniowy GPU robi co może. Mieszana precyzja koniecznie (FP16).
    - Alternatywnie, można rozważyć _partial fine-tune_ – np. odblokowanie tylko części warstw w pełnym FT, a resztę zamrozić. W PyTorch to proste (requires_grad=False na parametrach), w MLX-LM nie wiem czy wspierane bezpośrednio, ale DoRA/LoRA to de facto robi.
- **Bardzo duże modele (70B+)**: Tutaj LoRA/QLoRA to jedyna sensowna droga na Macu. Nawet 128GB RAM może nie pomieścić pełnych 70B FP16 (70B *2 = ~140 GB pamięci dla modelu, plus gradienty!). Ale 4-bit quant reduce to ~35 GB, co już mieści się. MLX-LM umożliwia trenowanie 70B QLoRA, co jest przełomowe lokalnie. Oczywiście, **sprzęt minimalny** to Mac z 128GB unified memory (albo 96GB – np. najwyższy MacBook Pro M3 Max ma 128GB, a Mac Studio M2 Ultra 192GB max). Prędkość będzie niższa – spodziewaj się może ~50-100 tokenów/s. Ale to wciąż znaczy, że drobne fine-tune (np. 1000 kroków na niewielkim zbiorze) zrobisz w ciągu paru godzin, nie tygodni. Hiperparametry: LR bardzo niski, rzędu 5e-5 do 1e-4, by nie rozchwiać modelu. Batch size zapewne 1 (może 2 jeśli 192GB i 4-bit). Koniecznie gradient checkpointing – bez tego nawet nie ruszy (chyba że masz 192GB i 4-bit, ale i tak).

    - **Stabilność**: duży model może mieć skłonność do niestabilnych gradienów, więc rozważ _gradient clipping_(ucięcie gradientu powyżej pewnego progu) – HF Trainer ma opcję, w MLX-LM można ewentualnie ręcznie implementować (nie wiem czy jest flaga).
    - **Czas**: pamiętaj, że 70B model per iteracja liczy się ~10x dłużej niż 7B. Więc planuj realistycznie – może lepiej krócej trenować, sprawdzić efekty, ewentualnie docelowo zrobić kilkanaście epok jak efekty są obiecujące.
    - W razie gdy nawet LoRA 4-bit nie domaga, można użyć _PEFT z adapterami innymi niż LoRA_, np. **AdaLora**(dynamiczne przydzielanie rank), albo **BitFit** (trening tylko biasów). BitFit zużywa jeszcze mniej pamięci (tylko parametry bias warstw się uczą, reszta zamrożona). Jako ostatnia deska ratunku – przy ekstremalnie małym RAM – można nawet trenować tylko **embeddingi** (np. do fine-tune stylu wypowiedzi, ale to rzadko stosowane bo bardzo ograniczone).
- **Sprzętowe różnice (M2 Ultra vs M3 Max)**: M2 Ultra to dwukrotnie połączony M2 Max – ma więcej rdzeni GPU (60 lub 76) i potencjalnie więcej pamięci, ale pojedyncze rdzenie są architekturą 5 nm „Avalanche” (starsza generacja). M3 Max (2024) ma 40 rdzeni GPU, ale nowszej architektury (3 nm, większa cache, szybsze ALU, wsparcie dla nowych instrukcji). Z testów wynika, że M3 GPU mają ok. +30-40% wydajności vs M2 w przeliczeniu na rdzeń w grach/grafice – prawdopodobnie podobnie w ML. To znaczy M3 Max 40-core może dogonić lub przegonić M2 Ultra 60-core w niektórych zadaniach, choć w innych Ultra będzie lepszy dzięki większej równoległości i przepustowości pamięci. **Dla fine-tuningu LLM/CV**:

    - Jeśli model mieści się w 64GB, M3 Max (max 128GB, 40 GPU) może być równie szybki co M2 Ultra (typu 64GPU). Natomiast jeśli potrzebujesz >128GB, tylko M2 Ultra 192GB da radę.
    - M3 Max i Ultra wspierają **Hardware TF32** (czy coś analogicznego) – Apple wspominało o usprawnieniach matrix multiply, co może dawać lepszą przepustowość FP16/BF16. W realiach: MacBook Pro M3 Max potrafi trenować 7B model <10 min tam gdzie poprzednicy potrzebowali ~2x więcej czasu​

        [heidloff.net](https://heidloff.net/article/apple-mlx-fine-tuning/# :~:text=MLX%20is%20a%20framework%20for,on%20a%20MacBook%20Pro%20M3)

        , więc widać skok generacyjny.
    - **Neural Engine**: M3 ma 2x szybszy Neural Engine niż M2. Choć do trenowania głównie używamy GPU, pewne modele mogłyby skorzystać z ANE (np. inferencja w pętli RLHF albo jakieś augmentacje). Póki co, MLX-LM zdaje się nie używać ANE do trenowania. Ale warto obserwować – Apple może to dodać, co da kolejnego kopa (ANE jest 16-rdzeniowy i specjalizowany, np. w CoreML inference Stable Diffusion używa ANE intensywnie by odciążyć GPU).
- **Chłodzenie i długotrwałość**: MacBooki (Air/Pro) mogą zwolnić z powodu termiki przy bardzo długich treningach. M3 generacja jest bardziej efektywna, ale wciąż – jeśli planujesz 24h trenować GAN-a, lepszy będzie sprzęt stacjonarny (Studio) lub ewentualnie MBP z dobrą wentylacją. Monitoruj temperatury i _Power Throttling_. Czasem zmniejszenie batch size czy włączenie limitu mocy (powiedzmy ograniczenie GPU Clock o 10%) może zapobiec throttlingowi i w rezultacie skrócić realny czas (bo nie będzie przerw na schłodzenie).

Podsumowując, **MLX-LM** jawi się obecnie jako jedno z najwygodniejszych rozwiązań do fine-tuningu na Macach – zwłaszcza dla LLM. Ma wbudowane najlepsze praktyki (LoRA domyślnie, obsługa QLoRA, checkpointing) i stale się rozwija. Inne metody (PyTorch MPS, HF Transformers) są bardziej uniwersalne i pozwalają dostrajać dowolny model, nie tylko tekstowy – czasem więc musisz z nich skorzystać (np. dla TTS, detekcji, gdzie MLX nie ma gotowego modułu). Niemniej, wiele konceptów jest wspólnych: redukcja wymagań pamięci poprzez niższą precyzję i adaptery, trade-off między szybkością a zużyciem (gradient checkpointing), i klasyczne techniki ML jak odpowiedni dobór LR. Dzięki temu przewodnikowi, miejmy nadzieję, jesteś w stanie samodzielnie przeprowadzić fine-tuning modeli 3B, 7B, 32B czy nawet 70B na własnym Macu, dostosowując je do polskiego języka, specyficznych zadań technicznych, wizji komputerowej, mowy czy generowania obrazów – **całkowicie lokalnie, bez potrzeby drogich serwerów w chmurze** 🎉.

#####
######
#####
######
#####
######
#####
######
#####
######
#####
######
#####
#####
######
#####
######
#####
######
#####
######
#####
######
#####
######
# Etap 1: Przygotowanie środowiska (instalacja i konfiguracja) W pierwszym kroku przygotowujemy środowisko pracy. Tworzymy nowe wirtualne środowisko Python i aktywujemy je, a następnie instalujemy bibliotekę MLX wraz z dodatkowymi pakietami, takimi jak Transformers (modele i tokenizatory), Datasets (obsługa zbiorów danych) oraz Accelerate (przyspieszenie treningu). Dodatkowo, aby uzyskać najnowszą wersję MLX, instalujemy ją bezpośrednio z repozytorium GitHub: ```

# 1. Tworzymy i aktywujemy nowe środowisko Python (venv)

python -m venv venv_mlx source venv_mlx/bin/activate

# 2. Instalujemy MLX oraz wymagane pakiety

pip install mlx mlx-lm transformers datasets accelerate

# (Opcjonalnie) Aktualizujemy MLX z najnowszego kodu źródłowego

pip install -U git+https://github.com/ml-explore/mlx.git ```

# Etap 2: Przygotowanie danych do *cold start* Kolejnym etapem jest przygotowanie zbioru danych, na którym przeprowadzimy wstępne dostrajanie modelu (*cold start*). Bazujemy tutaj na podejściu zaproponowanym w pracy DeepSeek R1, gdzie wykorzystuje się zestaw wysokiej jakości **łańcuchów rozumowania** (ang. *Chain-of-Thought*, CoT) z odpowiedziami. Zbieramy około 1000--2000 starannie przygotowanych przykładów zawierających długie wyjaśnienia (rozumowanie krok po kroku) wraz z finalnymi odpowiedziami.

Format danych treningowych przyjmujemy w postaci pliku JSON Lines, gdzie każdy przykład jest obiektem z polami `prompt` (pytanie lub polecenie dla modelu) oraz `completion`, które zawiera dwa pod-pola: `reasoning_process` (opis pełnego toku rozumowania prowadzącego do rozwiązania) oraz `summary` (krótka podsumowująca odpowiedź). Poniżej przedstawiono przykład struktury pojedynczego wpisu w takim zbiorze danych: ``` { "prompt": "Pytanie lub polecenie do modelu", "completion": { "reasoning_process": "Długi łańcuch rozumowania prowadzący do rozwiązania ...", "summary": "Krótkie podsumowanie lub finalna odpowiedź" } } ```

Tak przygotowany zestaw pełni rolę *danych startowych* do trenowania – zawiera przykłady, na podstawie których model nauczy się generować własne łańcuchy rozumowania przed udzieleniem odpowiedzi.

# Etap 3: Konfiguracja LoRA i parametrów treningu Na tym etapie definiujemy konfigurację eksperymentu – parametry modelu, treningu oraz metody LoRA. W pliku konfiguracyjnym określamy m.in.:  -  **Model bazowy**: tutaj wykorzystujemy model `deepseek-ai/deepseek-r1-lora-base`, który zawiera wstępnie wytrenowane mechanizmy rozumowania (tzw. zimny start, *cold start*). -  **Model docelowy**: czyli model, który dostrajamy – w naszym przypadku polski model `bielik-ai/bielik-11b`. -  **Parametry LoRA**: takie jak ranga dodatku (`lora_r`), współczynnik skalowania (`lora_alpha`) oraz prawdopodobieństwo odrzucenia (`lora_dropout`) dla warstw LoRA. -  **Parametry treningu**: rozmiar batcha, tempo uczenia (`learning_rate`), liczba kroków treningowych (`max_steps`), liczba kroków rozgrzewki (`warmup_steps`) oraz częstotliwość zapisu modeli (`save_steps`). -  **Ustawienia MLX**: np. `gradient_accumulation_steps` (pozwalający efektywnie zwiększyć rozmiar batcha przez akumulację gradientów) czy `mixed_precision` (włączenie treningu z mieszaną precyzją dla przyspieszenia obliczeń). Ustawienie `distributed_training = True` pozwala MLX wykorzystać wszystkie dostępne rdzenie GPU (np. w układach M1/M2 Ultra).

Przykładowa konfiguracja jest następująca: ```

# config.py

config = { # Model configurations "base_model": "deepseek-ai/deepseek-r1-lora-base", "bielik_model": "bielik-ai/bielik-11b",

yaml

Copy

`# LoRA hyperparameters "lora_r": 64,                # ranga (rank) "lora_alpha": 32,            # współczynnik skalujący (alpha) "lora_dropout": 0.05,  # Training hyperparameters "batch_size": 8, "learning_rate": 2e-4, "warmup_steps": 100, "max_steps": 1000, "save_steps": 100,  # MLX-specific settings "gradient_accumulation_steps": 4, "mixed_precision": True, "distributed_training": True  # wykorzystanie wielu rdzeni GPU (MPS)`

} ```

# Etap 4: Skrypt treningowy Mając zdefiniowane dane i konfigurację, możemy zaimplementować główny skrypt treningowy. Wykorzystamy API biblioteki MLX, które jest zbliżone do PyTorch, ale działa wewnętrznie z użyciem Metal (MPS). Poniżej przedstawiono przykładowy skrypt w~języku Python realizujący dostrajanie modelu z LoRA krok po kroku:

``` import mlx.core as mx import mlx.nn as nn from mlx.data.datasets import load_dataset from transformers import AutoTokenizer from typing import Dict, List

def prepare_training_data(examples: List[Dict]) -> Dict: # Przygotowanie danych treningowych do formatu akceptowanego przez model tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

python

Copy

`prompts = [ex["prompt"] for ex in examples] completions = [ex["completion"]["reasoning_process"] + "\n" +                 ex["completion"]["summary"] for ex in examples]  # Tokenizacja z paddingiem inputs = tokenizer(prompts, padding=True, truncation=True,                     max_length=512, return_tensors="mx") labels = tokenizer(completions, padding=True, truncation=True,                    max_length=1024, return_tensors="mx")  return {     "input_ids": inputs["input_ids"],     "attention_mask": inputs["attention_mask"],     "labels": labels["input_ids"] }`

def train(): # 1. Wczytanie modelu i tokenizatora tokenizer = AutoTokenizer.from_pretrained(config["bielik_model"]) model = mx.load_pretrained(config["bielik_model"])

arduino

Copy

`# 2. Dodanie warstw LoRA do modelu model = add_lora_layers(model, config["lora_r"], config["lora_alpha"])  # 3. Przygotowanie optymalizatora (Adam) optimizer = mx.optimizer.Adam(learning_rate=config["learning_rate"])  # 4. Główna pętla treningowa for step in range(config["max_steps"]):     batch = next(train_dataloader)          # Forward pass (obliczenie wyjścia i straty)     outputs = model(         input_ids=batch["input_ids"],         attention_mask=batch["attention_mask"],         labels=batch["labels"]     )     loss = outputs.loss / config["gradient_accumulation_steps"]          # Backward pass (obliczenie gradientów i aktualizacja wag)     grads = mx.grad(model.parameters(), loss)     optimizer.update(model.parameters(), grads)          if step % config["save_steps"] == 0:         save_lora_weights(model, f"checkpoint-{step}")`

```

W powyższym kodzie: najpierw ładujemy wstępnie wytrenowany model `bielik-11b` oraz odpowiadający mu tokenizer. Następnie wywołujemy funkcję `add_lora_layers`, która dodaje do modelu dodatkowe warstwy odpowiadające parametrom LoRA (w praktyce rozszerza macierze wag w warstwach modelu o składowe niskowymiarowe). Kolejnym krokiem jest zdefiniowanie optymalizatora – tutaj używamy `Adam` z zadanym wcześniej tempem uczenia. W głównej pętli treningowej pobieramy kolejne porcje danych (`batch`) z generatora `train_dataloader`, wykonujemy propagację w przód (*forward pass*) przez model wraz z obliczeniem straty (`outputs.loss`), dzielimy stratę przez wartość `gradient_accumulation_steps` (jeśli akumulujemy gradienty), po czym obliczamy gradienty względem wszystkich parametrów modelu (`mx.grad`) i wykonujemy krok optymalizacji (aktualizacja wag modelu przez `optimizer.update`). Co pewną liczbę kroków (zgodnie z `save_steps` w konfiguracji) zapisujemy aktualne wagi LoRA na dysk (`save_lora_weights`), aby móc przerwać i wznowić trening lub wykorzystać pośrednie wyniki.

# Etap 5: Ewaluacja i testy Po zakończeniu procesu trenowania warto ocenić jakość dostrojonego modelu. Możemy w tym celu wykorzystać zbiór walidacyjny lub testowy złożony z podobnych przykładów, co treningowe. Poniżej przedstawiono prostą funkcję `evaluate_model`, która dla każdego przykładu z zestawu testowego generuje odpowiedź modelu i porównuje ją z oczekiwaną odpowiedzią (podaną w polu `summary`).

``` def evaluate_model(model, test_examples): # Podstawowa ewaluacja modelu na zestawie testowym results = [] for example in test_examples: prompt = example["prompt"] gold_completion = example["completion"]["summary"]

bash

Copy

    `# Generacja odpowiedzi modelu na podstawie promptu     pred_completion = model.generate(         prompt,         max_length=1024,         temperature=0.7,         num_return_sequences=1     )[0]          # Porównanie z oczekiwaną odpowiedzią     results.append({         "prompt": prompt,         "predicted": pred_completion,         "gold": gold_completion,         # Można dodać wyliczanie metryk porównawczych, np. BLEU lub ROUGE     }) return results`

```

Powyższa funkcja zwraca listę wyników, gdzie dla każdego przykładu przechowujemy zadany `prompt`, wygenerowaną przez model odpowiedź (`predicted`) oraz prawidłową odpowiedź (`gold`). Na tej podstawie można przeprowadzić ocenę jakości – np. ręcznie sprawdzić poprawność odpowiedzi lub obliczyć automatyczne metryki, takie jak BLEU czy ROUGE, porównujące podobieństwo odpowiedzi modelu do wzorca.

# Wskazówki dla nowicjusza Na koniec, przedstawiamy kilka dodatkowych porad, które mogą być przydatne podczas dostrajania modelu LLM na Apple Silicon:  -  Zaczynaj od małego podzbioru danych (np. 50--100 przykładów), aby upewnić się, że cały pipeline treningowy działa poprawnie, zanim uruchomisz pełne trenowanie. -  Monitoruj zużycie pamięci podczas treningu – MLX powinien efektywnie zarządzać pamięcią na Apple Silicon, ale warto kontrolować, czy nie dochodzi do przekroczenia dostępnych zasobów. -  Włącz `mixed_precision=True` – trening z mieszanymi precyzjami (np. float16) znacząco przyspiesza obliczenia na układach Apple bez zauważalnej utraty jakości modelu. -  Wykorzystaj `gradient_accumulation_steps`, aby symulować większy batch size, jeśli ogranicza Cię dostępna pamięć – akumulacja gradientów pozwala trenować na efektywnie większej partii danych kosztem wydłużenia czasu jednej epoki. -  Zapisuj checkpointy modelu często (np. co 100--200 kroków) – w razie przerwania treningu (awaria, restart) będziesz mógł wznowić proces od ostatniego zapisanego stanu zamiast zaczynać od początku. -  Testuj model na prostych przykładach po każdej istotnej fazie (lub po każdym zapisie checkpoint) – pozwoli to szybko wychwycić, czy model faktycznie uczy się nowych umiejętności, czy np. doszło do przeuczenia.

\bigskip \noindent **Podsumowanie:** Dzięki wykorzystaniu MLX i metody LoRA nawet posiadacze MacBooków z układami Apple Silicon mogą lokalnie dostrajać duże modele językowe. Kluczem jest odpowiednie przygotowanie danych (np. strategie typu *cold start* z bogatymi przykładami CoT) oraz dostosowanie parametrów treningu do możliwości sprzętu. Przedstawiony proces krok po kroku pokazuje, że fine-tuning modelu 11B na Macu jest możliwy w rozsądnym czasie i zadowalającej jakości, co otwiera drogę do eksperymentów z LLM bez potrzeby korzystania z kosztownej infrastruktury chmurowej.

\bibliographystyle{unsrt} \bibliography{refs}

\end{document}

#####
#####
###### 