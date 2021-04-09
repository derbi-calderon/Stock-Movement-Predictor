import bs4
from urllib.request import urlopen as uReq
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as Soup
from bs4 import Comment


strategies_signals = ["https://finviz.com/screener.ashx?v=111&o=-volume&r=1"]
list = []
for signals in strategies_signals:
    baseName = signals
    stock_name = signals.replace("https://finviz.com/screener.ashx?v=111&o=-volume&r=", " ")
    print("==========================================================")
    print("==================" + stock_name + "=====================")
    print("==========================================================")

    page_nums = ["1", "21", "41", "61", "81", "101", "121", "141"]
    data_pages = [baseName]
    baseName = baseName + '&r='
    for n in page_nums:
        data_pages.append(baseName + n)

    for name in data_pages:
        # opening up connection, grabbing the page
        req = Request(name, headers={'User-Agent': 'Mozilla/83'})
        webpage = urlopen(req).read()

        # html parsing
        page_soup = Soup(webpage, "html.parser")

        # grabs each item
        comments = page_soup.find_all(string=lambda text: isinstance(text, Comment))

        if comments.__len__() > 1:
            nameString = str(comments[0])
            nameString = nameString.replace("TS", " ")
            nameString = nameString.replace("TE", " ")
            nameString = nameString.split('\n')
        else:
            nameString = "0"

        for c in nameString:
            st = ""
            for i in c:
                if i == '|':
                    break
                else:
                    st += i
            list.append(st)


    print("\n\n")
print(list)