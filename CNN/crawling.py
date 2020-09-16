from icrawler.builtin import GoogleImageCrawler
from openpyxl import load_workbook, Workbook

f = load_workbook('../datasets/nutrition.xlsx')
xl_sheet = f.active
rows = xl_sheet['F2:F840']
food_list = []
for row in rows:
    for cell in row:
        food_list.append(cell.value)
print(food_list)

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': '../images'})

for idx, food in enumerate(food_list):
    print(idx)
    google_crawler.crawl(keyword=food, max_num=100,
                        min_size=(200,200), max_size=None)