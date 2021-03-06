# Отчеты о состоянии проекта

#### 08.11.17 - Встреча с кураторами проекта

Узнали, что было сделано в проекте до нас. Анна М. и Алексей А.:
1. Создали транскрипцию стимула и синхронизировали её с аудио
2. Преобразовали слова в векторы с помощью word2vec
3. Интерполировали стимул в соответствии с частотой измерений томографа (раз в 2 секунды)
4. Построили линейную модель (ридж-регрессия) отдельно для каждого вокселя. В качестве признаков - значения стимула, соответствующие текущему измерению и трем предыдущим измерения. 
5. Визуализировали результаты применения модели для 6 испытуемых
6. Разработали демонстрационный интерфейс, который позволяет посмотреть слова, с наибольшей вероятностью вызывающие отклик в любом из вокселей
7. Исследовали нейросетевые модели отклика (их предсказательная сила близка к линейной модели)

Мы обсудили наше предварительное представление о целях проекта в целом и наших задачах. Идея – работать с транскрибцией стимула
-	Собственные и другие
-	Одушевленные и неодушевленные (как?)
-	Структура сложных слов
-	Немецкие и заимстованные слова 
-	Разметка по эмоциям

#### 09.11.17 - Выступление на НИС

Выступили на НИС без презентации, осознали большое количество вопросов, думали о том, как сделать разметку более семантической 

#### 31.01.2018 - Еженедельная встреча

1. После анализа ТЗ стало очевидно, что необходимо методом проб и ошибок **составить список категорий для каждой разметки*** и представить его в зафиксированном виде. 
* В схеме лексико-грамматической разметки необходимо проверить доступность инструментов для анализа всех грам.категорий глагольных форм. + ВАЖНО: **найти инструмент для вычленения сложных слов** (2 и больше корня)
* Для семантической разметки: нужно вернуться к **WordNet**, поскольку, например, отношения большей\меньшей абстракции могут быть полезны для нашего проекта.
2. Следует обсудить **способ представления** результатов совместной разметки: это будет размеченный текст или датасет? 
* Если датасет, то необходимо определить, каков способ представления слова - лексико-грамматические и семантические характеристики (например, самые значимые\частотные) в рамках одного вектора или разных? 
3. Намечен **брэйнсторм по статье, нужно подобрать подходящее под него время** 
4. Озвучены и приняты во внимание организационные моменты.
 
#### 21.02.2018 - Еженедельная встреча

1. Были обнаружены проблемы технического плана при разметке сложных слов(решить до 28.02). Относительно итогового датасета было принято решение размечать не конкретные слвоа, а словоформы в стимуле (включая словоизменительные категории). Задачей на следующие две недели является совмещение готовых инструментов для разметки в единый комплекс для датасета. 2. В связи с затруднениями относительно немецкого ворднета, было предложено использовать перевод на английский (разработан скрипт). До следующей недели необходимо завершить перевод и оценить его качество. 3. Связаться с кураторами курса по поводу формата предоставляемых данных.

#### 28.02.2018 - Еженедельная встреча

1. Утверждены инструменты для разметки сложных слов. Для разметки по признаку одушевлённости необходимы следующие семантические категории из GermaNet: animal,human,group. Задача на неделю: разметка стимула. 2. Сделана семантическая классификация части стимула (больше половины), предложено 15 групп глаголов, 23 группы существительных и 16 групп прилагательных. Возникшая проблема с лемматизацией части слов из стимула может быть решена использованием библиотеки spacy.

#### РЕЗУЛЬТАТЫ СПРИНТА: 

Сделана начальная семантическая классификация с использованием GermaNet, подготовлена схема лексико-грамматической разметки и найдены инструменты для её воплощения. 

#### 10.03.2018 - Еженедельная встреча

1. Charsplitter подходит для разметки сложных слов, хотя размечает и заимствованные слова (Para+Mount). 2.Проведена частеречная разметка с использованием spacy (две языковых модели, предстоит выбрать наиболее точную, но задача на будущее)+указаны синтаксические зависимости (может быть полезно). 3. Улучшен показатель лемматизированных слов:  лемматизация от spacy дала +60 новых лемм. Задача: поиск и оценка необходимости дополнительных инструментов лемматизации.  4. Выбранные морфологические анализаторы (DEMorphy, zurich morphological analyzer) вызвали ряд технических проблем, задача: решить в ближайшее время. 5. Новые способы очистки данных эксперимента могут существенно сказаться на весах - > Нужно запланировать возможность замены тензоров с весами, чтобы при пересчете весов уже был готовый инструмент, в который нужно просто подать новую информацию.


#### 16.03.2018 - Еженедельная встреча
1. Решены тех.проблемы с морфологической разметкой, есть части речи, грамматическая форма (в общем виде), синтаксическая роль. 2. Нужно выбрать, какой сплиттер подходит лучше. 3. Вопрос с выбором доминантной семантической категории для вокселя остаётся открытым. 4. Вопрос по статье: мы пишем обзорную статью или краткое вступление + своё исследование описываем? 


#### 21.03.2018 - Еженедельная встреча
1. Выбран сплиттер - Charsplitter. 2. Предложение исследовать морфологические характеристики повоксельно. 3. Кластеризация: как лучше оценить качество? 4. Статья: распределение по темам. Амир: обзор векторных моделей. Артём:  способы представления семантики в мозге. Даня: нейровизуализация. Настя: введение и заключение. 

#### РЕЗУЛЬТАТЫ СПРИНТА: 
Сделана начальная морфологическая разметка, существенно улучшена лемматизация слов для семантической разметки, распределены роли для написания статьи. 

#### 03.04.18- Встреча в Школе Лингвистики

Были обсуждены исследовательские вопросы, стоящие перед проектом:

1. Определение соотношения пространственной локализации аудиального стимула и лингвистических признаков слов (грамматических/семантических категорий) в ассоциированных вокселям списках слов. 

Этот вопрос, собственно, и ставился изначально в данном проекте. 

2. Оценка эмбеддингов при помощи данных локализации слов (например, определение ошибки регрессионной модели при предсказании координат вокселей), в частности, влияние алгоритма обучения и обучающего корпуса на генерируемый для каждого вокселя список слов, — иначе говоря проверка робастности используемых в первом исследовательском вопросе списков. 

Этот вопрос был поставлен из-за того, что списки слов, получаемые для каждого слова, очень сильно зависят от модели эмбеддингов (и, возможно, вообще будут меняться при обучении другой модели при использовании того же алгоритма, корпуса и гиперпараметров). При этом сами эмбеддинги являются далеко не идеальной моделью семантики, и вообще не очень интуитивно понятно, почему моделируемая ими семантика, основанная на подсчёте контекстов, должна соотноситься с некоторым представлением семантики в мозге, получаемым из фМРТ-данных.

Мы пришли выводу, что наибольший интерес для научного сообщества является первый исследовательский вопрос; таким образом, основная цель нашей исследовательской работы должна быть ответом на этот вопрос. При этом:

а) Помимо исследовательской статьи, мы будем писать и обзорную статью, которая будет посвящена сравнению компьютерных лингвистических моделей (не только эмбеддингов, но и, например, вероятностных языковых моделей) и данных репрезентаций лингвистических стимулов в мозге.
б) У нашего исследования есть некоторый формальный майлстоун, который будет сводиться к сдаче проекта по НИСу для получения по нему оценки. Не очень понятно, что ожидает увидеть Анастасия Александровна в результате работы — кажется, это должен быть некоторый датасет с набором лингвистических признаков для каждого слова (вроде как он почти готов) и сама обзорная работа. При этом исследовательская статья вроде как не является обязательным требованием для сдачи проекта, поэтому работать над исследованием можно будет и после сдачи НИСа.

Текущий бейзлайн ответа на поставленный исследовательский вопрос (то есть текущая задача) заключается в нахождении попарных корреляции метрик схожести списков слов с воксельным пространственным расстоянием (то, что я описывал в предыдущем письме). Чем больше будет корреляция, тем лучшее соотносятся характеристики слов. Это не единственный и далеко не самый репрезентативный способ ответить на поставленный вопрос, поэтому после того, как мы закончим с этой задачей (ожидается, что это будет к концу этой недели), можно будет подумать и над другими способами. Пока мы предполагаем рассмотреть возможности кластеризации списков слов, для которых можно однозначным образом определить категорию.

Кроме всего прочего, к нам сегодня приходил Вадим Викторович Ушаков из Курчатников, у которого есть интерес делать межъязыковое исследование на фМРТ данных, в частности, определить устойчивость частей речи в разных языках: находятся ли зоны существительных и глаголов в одном месте в мозге. Эта задача не входит в понимаемую нами цель исследования, но при желании, кажется, можно будет поработать и над этим исследованием. Также, лаборатория Ушакова занимается проведением похожих фМРТ-экспериментов и для русского языка, так что, возможно, у нас могут появиться и русскоязычные данные, с которыми тоже интересно будет поэкспериментировать. 

Следующая встреча назначена на 14:00 12 апреля (четверг), и будет в Шаде. 

#### 12.04.18- Встреча в ШАДе  

Мы решили, что для научного сообщества в целом будет полезен обзор на тему попыток связывания компьютерных лингвистических моделей и нейролингвистических данных, поэтому неплохо было бы его писать в параллель с нашей основной статьёй (и потом выложить на arxiv). Чтобы написать действительно хороший и полный обзор нужен не один месяц, поэтому мы решили, что как курсовую можно будет сдать не совсем полный обзор (вряд ли рецензенты курсовой будут настолько экспертами, чтобы заметить это), и продолжить работать над ним летом.

В качестве названия обзора мы придумали следующее «Computational Linguistic Models reveal Patterns of Human Semantic Processing: A Survey». 

Далее, мы обсудили эксперименты, которые провёл Артём. Их результаты отражены в нашей статье на Оверлифе. TL;DR: взяли в качестве признаков координаты, но координаты это небольшое признаковое пространство (всего три признака), потом добавили туда усредненные эмбеддинги слов, ассоциированных вокселю. Используя как таргеты семантические категории из GermaNet и части речи, на данной векторно-матричной структуре были выполнены задачи кластеризации (KMeans с евклидовой близостью) и классификации (градиентный бустинг). Для частей речи выяснилось, что в каждом кластере лежит одна группа существительных, и кластеризация существительных провалилась. 

Основной момент, по которому возникли вопросы у Алексея, состоял в том, что лингвистический признак вокселя (семантическая категория либо часть речи) определялся на основе списка слов вокселя, которые в свою очередь определялись с помощью эмебеддингов (скалярное произведение эмбеддинга на вес вокселя). Таким образом, между таргетом и признаковым вектором существовала прямая зависимость, и задача классификации в этом случае была не совсем корректна. 

Было предложено сосредоточить внимание на кластеризации, а не на классификации, и попытаться оценивать её качество просмотром визуализиации групп вокселей «на глаз» (действительно ли воксели образуют какие-то группы).

Кроме того, в числе других предложений по улучшению дизайна эксперимента значились:

-нормировка координат вокселей
-вставка в статью таблицы с примерами категорий для сэмпла из нескольких слов (потому что непосвященному читателю не очень понятно, что это вообще за категории)
-в качестве признаков использовать bag-of-categories, и кластеризовать по ним.

Помимо экспериментов Артёма с тезаурусной семантикой, мы подумали и о том, какие лингвистические фичи можно будет использовать в дальнейших экспериментах. Настя сделала проект, в котором supervised модель определяет, является ли слово сложным. Предсказания этой модели можно использовать как одни из лингвистических фич, нужно подумать и проконсультироваться с людьми, знающими немецкий, о том, какие ещё признаки немецких слов мы можем использовать (род? одушевленность?).

Планируется закончить с экспериментами с семантическими категориями, и на следующей неделе подключиться к Насте и её работе над морфо-синтаксической частью.
