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





