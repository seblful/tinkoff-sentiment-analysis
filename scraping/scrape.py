import requests
from bs4 import BeautifulSoup

url = 'https://www.tinkoff.ru/invest/stocks/SBER/pulse/'

s = requests.Session()
payload = {"count": 1000}

response = s.post(url, data=payload)
soup = BeautifulSoup(response.content, 'html.parser')

entry_elem = soup.find_all('div', {'class': 'pulse-posts-by-ticker__gXTx8X'})
counter = 0
for elem in entry_elem:
    counter += 1
    text_elem = elem.find(
        'div', {'class': 'pulse-posts-by-ticker__fGGBmY'})

    text_elem = text_elem.find(
        'div', {'class': 'pulse-posts-by-ticker__cfTK6Z'})

    text = text_elem.get_text(strip=True)

    print(text)

print(counter)
