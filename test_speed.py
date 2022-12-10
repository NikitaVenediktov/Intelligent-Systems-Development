import asyncio
from time import perf_counter

import aiohttp
import requests
from tqdm import tqdm

# url = "http://localhost:8080/reco/pop14d/"
# url = "http://95.165.161.126:777/reco/pop14d/"

# url = "http://localhost:8080/reco/user_knn_v1/"
# url = "http://95.165.161.126:777/reco/user_knn_v1/"


headers = {
  'Authorization': 'Bearer reco_mts_best'
}

# ******SIMPLE REQUESTS******
# list_of_items = range(1, 11)
# start = perf_counter()
# for x in tqdm(list_of_items):
#     r = requests.get(url + f'{x}', headers=headers)
# stop = perf_counter()
# print("Total time:", round((stop - start), 4))
# mean_time = round((stop - start)/len(list_of_items), 4)
# print('Mean time for 1 request', mean_time)


# ******ASYNC REQUESTS******

async def fetch(s, url):
    async with s.get(f'http://95.165.161.126:777/reco/user_knn_v1/{url}', headers=headers) as r:
        if r.status != 200:
            r.raise_for_status()
        return await r.text()


async def fetch_all(s, urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(s, url))
        tasks.append(task)
    res = await asyncio.gather(*tasks)
    return res


async def main():
    urls = range(1, 400)
    async with aiohttp.ClientSession() as session:
        htmls = await fetch_all(session, urls)
        # print(htmls)


if __name__ == '__main__':
    start = perf_counter()
    asyncio.run(main())
    stop = perf_counter()
    print("Total time:", round((stop - start), 4))
