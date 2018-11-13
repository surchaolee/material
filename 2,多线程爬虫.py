import requests
from lxml import etree
from threading import Thread, Lock
import queue
import time
from urllib.error import URLError
import random
import json

UserAgent_List = [
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2224.3 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 4.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.3319.102 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.2309.372 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.2117.157 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1866.237 Safari/537.36",
]

# 全局退出标志
exit_parse_flag = False

# 锁
lock = Lock()


# 爬虫线程
class SpiderThread(Thread):
    def __init__(self, id, url, page_queue, parse_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
        self.page_queue = page_queue
        self.parse_queue = parse_queue
        self.url = url

    def run(self):
        # 从page_queue取出要爬的页码.拼出url
        while True:
            if self.page_queue.empty():
                break
            try:
                page = self.page_queue.get(block=False)
                # 等该做的事情全部做完之后再去发送信号.

                print('%d线程获取了%d页码' % (self.id, page))
                url = self.url % str(page)
                print(url)
                # 使用requests 请求页面.
                # 避免因为网络问题,尝试4次
                times = 4
                while times > 0:
                    try:
                        response = requests.get(url, headers={'User-Agent': random.choice(UserAgent_List)})
                        time.sleep(1)
                        # 把获取到的结果put到parse_queue
                        self.parse_queue.put(response.text)
                        # 处理完一个get出来的page之后，调用task_done将向队列发出一个信号，表示本任务已经完成
                        # 放入解析队列，才表示本次get的page完成
                        self.page_queue.task_done()
                        # self.parse_queue.task_done()
                        break
                    except URLError:
                        print('网络错误!')
                    finally:
                        times -= 1
            except queue.Empty:
                pass


# 解析线程
class ParseThread(Thread):
    def __init__(self, id, parse_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_queue = parse_queue
        self.id = id

    def run(self):
        # 拿到队列中的待解析数据.
        global exit_parse_flag
        # print('解析线程开始执行...')
        while True:
            # print(exit_parse_flag)
            if exit_parse_flag:
                break
            try:
                data = self.parse_queue.get(block=False)
                self.parse(data)
                print('%d解析线程解析了一页数据' % self.id)
                self.parse_queue.task_done()
            except queue.Empty:
                pass

    # 解析的是糗百
    def parse(self, data):
        # 生成一个etree
        tree = etree.HTML(data)

        # 使用xpath取出想要的东西.
        div_list = tree.xpath('//div[contains(@id,"qiushi_tag_")]')
        # print(div_list)
        # 对div_list中的每一个div进行数据解析
        global results
        for div in div_list:
            # 头像的url
            head_shot = div.xpath('.//img/@src')[0]
            # 作者名字
            name = div.xpath('.//h2')[0].text
            # 内容
            content = div.xpath('.//span')[0].text.strip('\n')
            # 弄一个字典来保存
            item = {
                'head_shot': head_shot,
                'name': name,
                'content': content
            }
            results.append(item)


def save_data():
    # 写入一个本地文件.
    with open('./qiubai.json', mode='w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False)


# 两个队列
if __name__ == '__main__':
    results = []
    print('主线程开始执行')
    # page_queue
    page_queue = queue.Queue(10)
    for i in range(1, 11):
        page_queue.put(i)

    # parse_queue
    parse_queue = queue.Queue(10)

    # url
    url = 'https://www.qiushibaike.com/8hr/page/%s/'
    # 生成爬虫线程
    for i in range(1, 3):
        SpiderThread(id=i, page_queue=page_queue, parse_queue=parse_queue, url=url).start()
    # 生成解析线程
    for i in range(1, 3):
        ParseThread(id=i, parse_queue=parse_queue).start()
    # 使用队列锁.等队列为空的时候
    # 等到队列为空并监视item，直到当前队列中所有item都调用了task_done之后主线程才继续向下执行
    page_queue.join()
    parse_queue.join()
    # 给定爬虫线程关闭的条件
    exit_parse_flag = True

    save_data()
