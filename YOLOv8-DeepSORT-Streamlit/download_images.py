from icrawler.builtin import BingImageCrawler

def download_images(keyword, folder, num):
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=keyword, max_num=num)

# Ambulance
download_images(
    "ambulance traffic road india emergency vehicle",
    "dataset/images/ambulance",
    1500
)

# Vehicles
download_images("cars traffic indian road", "dataset/images/car", 2000)
download_images("bus traffic india road", "dataset/images/bus", 2000)
download_images("truck highway traffic india", "dataset/images/truck", 2000)
download_images("motorcycle traffic road india", "dataset/images/motorcycle", 2000)

# People
download_images("people crossing road traffic india", "dataset/images/person", 1500)

print("✅ Download completed")