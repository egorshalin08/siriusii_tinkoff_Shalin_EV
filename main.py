import requests
from bs4 import BeautifulSoup

url = 'https://otzovik.com/reviews/bank_tinkoff_kreditnie_sistemi/'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

respose_texts = []
response_values = []
response_dates = []

for page_number in range(1, 423):
    if page_number != 1:
        url1 = url + str(page_number) + '/'
    else:
        url1 = url

    page = requests.get(url1, headers=headers)

    soup = BeautifulSoup(page.text, 'lxml')

    url_to_full_response = soup.select("div.item-right > div.review-body-wrap > h3 > a")

    for i in range(len(url_to_full_response)):
        url_to_full_response1 = url_to_full_response[i].split
        url_to_full_response1 = 'https://' + url_to_full_response1[2][6:-1]

        page_full_response = requests.get(url_to_full_response1, headers=headers)

        response_value = soup.select("div.item-right > div.rating-wrap > div[class='rating-score tooltip-right'] > span")
        response_values.append(response_value[0].text)

        rev_topic = soup.select("div.item-right > h1")
        rev_topic = rev_topic[0].text

        rev_plus = soup.select("div.item-right > div.review-plus")
        rev_plus = rev_plus[0].text

        rev_minus = soup.select("div.item-right > div.review-minus")
        rev_minus = rev_minus[0].text

        rev_text = soup.select("div.item-right > div[class='review-body description']")
        rev_text = rev_text[0].text

        all_rev_text = rev_topic + ' ' + rev_plus + ' ' + rev_minus + ' ' + rev_text

        respose_texts.append(all_rev_text)

        rev_date = soup.select("div.item-right > div.rating-wrap > span[class='review-postdate dtreviewed'] > span")
        response_dates.append(rev_date[0].text)



from gensim.models import Word2Vec
import numpy as np
import re
import pymorphy3
import nltk
from sklearn.cluster import KMeans

morph = pymorphy3.MorphAnalyzer()
stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend(['документ', 'номер', 'тинькофф', "отзыв", "недостаток", "достоинство", "банк"])

def cl_text(text):
    c = text.lower()
    c = re.sub(r'crm[^\n]+', '', c)
    c = re.sub(r'\n+', ' ', c)
    c = re.sub(r'\s+', ' ', c)
    c = re.sub(r"[A-Za-z!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+", ' ', c)
    return c.strip()


def lemmatize(text):
    tokens = []
    text = re.sub(r"\d+", '', text.lower()) 
    for token in text.split():
        token = token.strip()
        token = morph.normal_forms(token)[0].replace('ё', 'е')
        if token and token not in stopwords: tokens.append(token)
    if len(tokens) > 2: tokens = ' '.join(tokens)
    return tokens


response_texts = [
    'Отзыв: Тинькофф банк - Отличный банк Достоинства: Отличный Недостатки: Нет Вообще сначала не хотела даже дебитовую карту заводить так как очень уж плохого слышала о данном банке и ужасно опасалась, но потом меня уговорили и я взяла дебетовую карту и не могла понять почему люди так негативно отзывались о банке. Пользуюсь данным банком уже второй год и всем довольна, а недавно год назад я еще и кредитную карту оформила и тоже ни разу не пожалела дали пробный период кучу кэшбэка получаю каждый месяц что очень удобно и главное приятно. Так же с кредитной картой все понятно у меня так же есть кредитная карта альфа банка и с ней я до сих пор не могу разобраться хотя пользуюсь дольше, а с кредитной картой тинькофф банка вообще никаких вопросов не возникает',
    'Отзыв: Тинькофф банк - Просто волшебно! Достоинства: Скорость Выгода Простота Недостатки: Нету ! Произошло ДТП, была оформлена страховка осаго Тинькофф(номер 0318788764), такой оперативности решения проблемы я не ожидала, потрясающая менеджер Алина, помогла на все 100%! Оперативно помогла скорректировать сумму выплаты, высший уровень ! Всем советую страховку от Тинькофф, выгодно и оперативно Все решили меньше чем за сутки !',
    'Отзыв: Тинькофф банк - Хороший банк Достоинства: Удобное обслуживание Красивый интерфейс Легко переводить на разные нужды Недостатки: Плата за уведомления об операциях, которую я не сразу обнаружила Я пользуюсь данной банковской картой примерно 3 месяца, обслуживанием довольна. Порекомендую в обязательном порядке своим знакомым)',
    'Отзыв: Тинькофф банк - Удобный мобильный банк , хороший кэшбек Достоинства: Удобный мобильный банк Недостатки: Мало офлайн офисов Хороший кэшбек, удобный мобильный банк, быстрое оформление и доставка карт, очень была удивлена тому, что сотрудники банка реально вникают в проблему и решают ее, отправила деньги человеку не на ту карту, после звонка с объяснениями всей ситуации вернули дс на мою карту .',
    'Отзыв: Тинькофф банк - Бесконечные пустые отписки, нарушение законных прав вкладчиков, при требовании предоставить установленные законодательством документы, отключает техподдержку Достоинства: Не увидела Недостатки: Техподдержка пишет бесконечные пустые отписки, нарушение законных прав вкладчиков при требовании предоставить установленные законодательством документы По рекомендации знакомых решила начать пользоваться услугами банка Тинькофф. Для того, чтобы получить первоначальное впечатление и оценить качество предлагаемых услуг, оформила с представителем банка Заявление-анкету и получила пластиковую карту. Затем я попыталась оформить Договор банковского вклада (депозитный договор). С этой целью положила на счет определенную сумму денежных средств, с использованием личного кабинета оформила договор банковского вклада, перевела деньги на депозитный вклад, но от банка почему-то получила непонятный документ - Справку о доступном остатке. Следует отметить, что размещение денежных средств в банках с целью получения процентов строго регламентировано Гражданским кодексом Российской Федерации, а также Федеральным Законом "О банках и банковской деятельности". Договор банковского вклада (депозитный договор) – это соглашение, в силу которого одна сторона (банк), принявшая поступившую от другой стороны (вкладчика) или поступившую для другой стороны денежную сумму (вклад), обязуется возвратить сумму вклада и выплатить проценты на нее на условиях и в порядке, предусмотренных договором (ст. 834 ГК РФ). Договор банковского вклада ДОЛЖЕН БЫТЬ ЗАКЛЮЧЕН в письменной форме. В статье 36 Федерального Закона «О банках и банковской деятельности» говорится, что привлечение средств во вклады ОФОРМЛЯЕТСЯ ДОГОВОРОМ В ПИСЬМЕННОЙ ФОРМЕ в двух экземплярах, ОДИН ИЗ КОТОРЫХ ВЫДАЕТСЯ ВКЛАДЧИКУ. Таким образом, каждый гражданин, обратившийся в банк с целью открытия банковского вклада, ВПРАВЕ И ДОЛЖЕН ПОЛУЧИТЬ НАДЛЕЖАЩИМ ОБРАЗОМ ОФОРМЛЕННЫЙ ЭКЗЕМПЛЯР ДОГОВОРА БАНКОВСКОГО ВКЛАДА, заключенного с банком (не справку, не анкету, не общий договор банковского обслуживания, в рамках которого, якобы происходит оформление остальных документов). Но сотрудники банка этого не знают и не хотят знать! С целью получения своего экземпляра договора банковского вклада, оформленного в соответствии с требованиями законодательства Российской Федерации я обратилась в техподдержку банка. Однако вскоре выяснилось, что предоставлять данный документ работники банка мне не собираются. На протяжении более, чем недели мне высылали бесконечные отписки о том, что прислали справку, что не знают, какой документ нужен, пересылали мне скан имеющегося у меня заявления - анкеты, сообщали, что мои требования передали в профильное подразделение банка и документ скоро подготовят С указанием дат и времени, которые они все время нарушали. В результате бесконечных отписок работники банка не только не предоставили мне подписанный уполномоченным лицом банка и заверенный установленным законодательством РФ ДОГОВОР БАНКОВСКОГО ВКЛАДА, но и перестали отвечать на мои сообщения в чате техподдержки. Учитывая, что офисов в данном банке нет, обратиться мне, кроме как в Управление Федеральной службы по надзору в сфере прав потребителей и благополучия человека, а также в прокуратуру и суд более некуда. Результат пользования услугами банка крайне неутешительный. Техподдержка работает на отписки, результаты нулевые, денежные средства лежат на счете без законного документального подтверждения, то есть без оформленного договора банковского вклада. Общение с работниками банка вызывает крайнее недоумение ввиду их некомпетентности и не желания изучать и неукоснительно соблюдать требования законодательства РФ. Остается задуматься о том, как они будут исполнять свои обязательства по вкладу, если они так бояться оформить необходимые документы. Поэтому я настоятельно НИКОМУ НЕ РЕКОМЕНДУЮ ПОЛЬЗОВАТЬСЯ услугами данного банка.',
    'Отзыв: Тинькофф банк - Возьмите кредит до 30 млн., только мы вам их никогда не одобрим. Как обманывает Тинькофф Достоинства: Когда то была очень клиентоориентированная компания. Скорость ответа Недостатки: Исчезло человеческое отношение- создаётся впечатление, что общаешься с роботом Я являюсь давним клиентом банка и всегда была крайне лояльна. Да и банк платил тем же. Все вопросы мы решали мирно и я очень активно пользовалась многими продуктами. Был опыт и с кредитом и дебетовая карта имеется и активно пользуемся кредитной картой. На Тинькоффе была завязана вся жизнь. Все покупки, путешествия, рестораны, все проходило через руки банка. Случались и проблемы, куда же без них! Например цена в приложении на сеанс кино была заявлена одна, а при покупке билета становилась выше. Дважды была такая ситуация. И каждый раз сотрудники поддержки пытались меня уверить, что - то билеты для детей предназначены, то им вовремя информацию не предоставили. В целом все было прекрасно и жили мы душа в душу, пока не случилось это. Еще несколько месяцев назад, когда ставки упали, я решила попробовать перекрыть ипотеку потребительским кредитом. Благо ставки позволяли. Оформила заявку на 1,3млн руб. Однако одобрили мне всего 750 тыс. Тогда вопрос отпал сам собой, но было обидно, что банк не дал нужную сумму. Ни разу не была замечена в просрочках или иных действиях, что могло бы испортить мою репутацию как клиента. Сначала я подумала, ну ладно, по каким-то причинам приняли такое решение. Может время неспокойное было или иные моменты. Потребность в кредите отпала. Сейчас мне снова потребовалась сумма в размере 1,5 млн рублей для покупки дачи. Я направила запрос и вместо запрошенных 1,5 млн мне снова одобрили 750 тыс. Ради интереса оформили заявку и на супруга на сумму 1,5 млн. Невероятное совпадение - снова одобрили 750 тыс.Что получается, банк заведомо вводит в заблуждение своих клиентов, обещая разные возможности, а по факту оказывается есть скрытый лимит и размер его 750 тысяч. 3 заявки, 3 одинаковых суммы. Очень неприятно было сознавать этот факт. Банк, которому я и моя семья доверили все финансы, просто обманывает. Не думайте, что я такая вредная и сразу побежала писать отзыв. Я направила свое обращение в Тинькофф, описав указанные моменты и предложила решение, мне - рассматривают кредит размером 1,2 млн сроком на 3 года, а я со своей стороны пишу отзыв. А уж какой он будет зависит от решения банка. Как вы понимаете чуда не случилось. Поэтому пишу отрицательную историю. А что же дальше? А вот на этом все. Отношения с банком я планирую прекратить в самое ближайшее время. Отзыв будет размещен на всех площадках, которые только могут быть. Клиенты должны знать, что их вводят в заблуждение. Еще один момент, который хотелось бы осветить это работа поддержки в данном случае наверное уже второго звена. Специалист Эльвира, просто налила воды в ответ, не поняв суть. Дважды меня пытались убедить, что на сайте Тинькофф указана информация, что лимит не до 30 млн, а только до 5 млн, однако если зайти на сайт с компьютера этой информации там нет. И кредит до 30 млн обещан каждому. Даже мой скриншот не помог. Ну а дальше мне холодно объяснили, что в моей просьбе отказано, никакого рассмотрения не предусмотрено и вежливо подчеркнули, что если не надо, то вам и очень-то и рады. Не знаю зачем, но мне пишут и акцентировать внимание представители банка о том, что на сайте кредит без залога до 5 млн. Хорошо, вопросов нет, мне всего нужно 1 млн 200 тыс, и эта сумма бы спокойно проходит в лимит аж 4 раза. А вот ещё вспомнила, процент мне так никто и не сказал. Индивидуальные условия, так пояснили. И в заключении хотелось бы обратить внимание компании на то, что иногда можно отойти в сторону от правил, а не придерживаться принципиальной позиции. Больше Тинькофф неклиенториентированная компания. В моем лице вы потеряли очень лояльного и верного клиента. Искренне жаль разрывать сотрудничество, банк- то в целом был самый лучший. Пойду искать другой банк, а точнее его я уже нашла и кредит мне там одобрили на нужную сумму без всяких проблем и заморочек. Всем удачи!'
]
response_values = ['5', '5', '4', '4', '1', '1']
response_dates = ['30 мар 2024', '29 мар 2024', '26 мар 2024', '22 мар 2024', '21 мар 2024', '15 мар 2024']


for i in range(len(response_texts)):
    response_texts[i] = cl_text(response_texts[i])
    response_texts[i] = lemmatize(response_texts[i])
    

word2vec_model = Word2Vec(sentences=response_texts, vector_size=100, window=5, min_count=1, workers=4)

def document_vector(word2vec_model, doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    return np.mean(word2vec_model.wv[doc], axis=0)

X = [document_vector(word2vec_model, doc) for doc in response_texts]
X = np.nan_to_num(X)

kmeans = KMeans(n_clusters=5) #можно менять
kmeans.fit(X)
kmeans_clusters = kmeans.labels_

print("K-means Clusters:", kmeans_clusters)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(response_texts)

for cluster_idx in np.unique(kmeans_clusters):
    cluster_indices = np.where(kmeans_clusters == cluster_idx)[0]
    cluster_texts = [response_texts[i] for i in cluster_indices]
    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
    cluster_tfidf = np.asarray(cluster_tfidf).reshape(-1)
    
    significant_indices = np.argpartition(cluster_tfidf, -10)[-10:]

    significant_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in significant_indices]


good_resp = response_values.count("5") + response_values.count("5")
bad_resp = response_values.count("2") + response_values.count("1")
neutral_resp = response_values.count("3")

num_resp = len(response_values)

print(f"Положительных отзывов {good_resp}, это {good_resp / num_resp * 100}% от общего количества.")
print(f"Положительных отзывов {bad_resp}, это {bad_resp / num_resp * 100}% от общего количества.")
print(f"Положительных отзывов {neutral_resp}, это {neutral_resp / num_resp * 100}% от общего количества.")




