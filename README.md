

#### Python基础知识点回顾

1. 数据结构和算法
   - 排序算法（冒泡和归并）和查找算法（顺序和二分）

     ```Python
     def bubble_sort(items, comp=lambda x, y: x > y):
         """高质量冒泡排序(搅拌排序)"""
         for i in range(len(items) - 1):
             swapped = False
             for j in range(len(items) - 1 - i):
                 if comp(items[j], items[j + 1]):
                     items[j], items[j + 1] = items[j + 1], items[j]
                     swapped = True
             if swapped:
                 swapped = False
                 for j in range(len(items) - 2 - i, 0, -1):
                     if comp(items[j - 1], items[j]):
                         items[j], items[j - 1] = items[j - 1], items[j]
                         swapped = True
             if not swapped:
                 break
     ```

     ```Python
     def merge_sort(items, comp=lambda x, y: x <= y):
         """归并排序(分治法)"""
         if len(items) < 2:
             return items[:]
         mid = len(items) // 2
         left = merge_sort(items[:mid], comp)
         right = merge_sort(items[mid:], comp)
         return merge(left, right, comp)
     
     
     def merge(items1, items2, comp=lambda x, y: x <= y):
         """合并(将两个有序的列表合并成一个有序的列表)"""
         items = []
         idx1, idx2 = 0, 0
         while idx1 < len(items1) and idx2 < len(items2):
             if comp(items1[idx1], items2[idx2]):
                 items.append(items1[idx1])
                 idx1 += 1
             else:
                 items.append(items2[idx2])
                 idx2 += 1
         items += items1[idx1:]
         items += items2[idx2:]
         return items
     ```

     ```Python
     def seq_search(items, key):
         """顺序查找"""
         for index, item in enumerate(items):
             if item == key:
                 return index
         return -1
     ```

     ```Python
     def bin_search(items, key):
         """折半查找(循环实现)"""
         start, end = 0, len(items) - 1
         while start <= end:
             mid = (start + end) // 2
             if key > items[mid]:
                 start = mid + 1
             elif key < items[mid]:
                 end = mid - 1
             else:
                 return mid
         return -1
     ```

   - 使用生成式（推导式）语法

     ```Python
     prices = {
         'AAPL': 191.88,
         'GOOG': 1186.96,
         'IBM': 149.24,
         'ORCL': 48.44,
         'ACN': 166.89,
         'FB': 208.09,
         'SYMC': 21.29
     }
     # 用股票价格大于100元的股票构造一个新的字典
     prices2 = {key: value for key, value in prices.items() if value > 100}
     print(prices2)
     ```

   - 嵌套的列表

     ```Python
     def main():
         names = ['关羽', '张飞', '赵云', '马超', '黄忠']
         courses = ['语文', '数学', '英语']
         # 录入五个学生三门课程的成绩
         # 错误 - 参考http://pythontutor.com/visualize.html#mode=edit
         # scores = [[None] * len(courses)] * len(names)
         scores = [[None] * len(courses) for _ in range(len(names))]
         for row, name in enumerate(names):
             for col, course in enumerate(courses):
                 scores[row][col] = float(input(f'请输入{name}的{course}成绩: '))
         print(scores)
     
     
     if __name__ == '__main__':
         main()
     ```

     [Python Tutor](http://pythontutor.com/) - VISUALIZE CODE AND GET LIVE HELP

   - heapq、itertools等的用法
     ```Python
     """
     从列表中找出最大的或最小的N个元素
     """
     import heapq
     
     
     def main():
         list1 = [34, 25, 12, 99, 87, 63, 58, 78, 88, 92]
         list2 = [
             {'name': 'IBM', 'shares': 100, 'price': 91.1},
             {'name': 'AAPL', 'shares': 50, 'price': 543.22},
             {'name': 'FB', 'shares': 200, 'price': 21.09},
             {'name': 'HPQ', 'shares': 35, 'price': 31.75},
             {'name': 'YHOO', 'shares': 45, 'price': 16.35},
             {'name': 'ACME', 'shares': 75, 'price': 115.65}
         ]
         print(heapq.nlargest(3, list1))
         print(heapq.nsmallest(3, list1))
         print(heapq.nlargest(2, list2, key=lambda x: x['price']))
         print(heapq.nlargest(2, list2, key=lambda x: x['shares']))
     
     
     if __name__ == '__main__':
         main()
     ```

     ```Python
     """
     排列 / 组合 / 笛卡尔积
     """
     import itertools
     
     
     def main():
         for val in itertools.permutations('ABCD'):
             print(val)
         print('-' * 50)
         for val in itertools.combinations('ABCDE', 3):
             print(val)
         print('-' * 50)
         for val in itertools.product('ABCD', '123'):
             print(val)
     
     
     if __name__ == '__main__':
         main()
     ```

   - collections模块下的工具类

     ```Python
     """
     找出序列中出现次数最多的元素
     """
     from collections import Counter
     
     
     def main():
         words = [
             'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
             'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around',
             'the', 'eyes', "don't", 'look', 'around', 'the', 'eyes',
             'look', 'into', 'my', 'eyes', "you're", 'under'
         ]
         counter = Counter(words)
         print(counter.most_common(3))
     
     
     if __name__ == '__main__':
         main()
     ```

   - 穷举法、贪婪法、分治法、动态规划

     ```Python
     """
     穷举法 - 穷尽所有可能直到找到正确答案
     """
     
     
     def main():
         # 公鸡5元一只 母鸡3元一只 小鸡1元三只
         # 用100元买100只鸡 问公鸡/母鸡/小鸡各多少只
         for x in range(20):
             for y in range(33):
                 z = 100 - x - y
                 if 5 * x + 3 * y + z // 3 == 100 and z % 3 == 0:
                     print(x, y, z)
         # A、B、C、D、E五人在某天夜里合伙捕鱼 最后疲惫不堪各自睡觉
         # 第二天A第一个醒来 他将鱼分为5份 扔掉多余的1条 拿走自己的一份
         # B第二个醒来 也将鱼分为5份 扔掉多余的1条 拿走自己的一份
         # 然后C、D、E依次醒来也按同样的方式分鱼 问他们至少捕了多少条鱼
         fish = 1
         while True:
             total = fish
             enough = True
             for _ in range(5):
                 if (total - 1) % 5 == 0:
                     total = (total - 1) // 5 * 4
                 else:
                     enough = False
                     break
             if enough:
                 print(fish)
                 break
             fish += 1
     
     
     if __name__ == '__main__':
         main()
     ```

     ```Python
     def fib(num, temp={}):
         """用递归计算Fibonacci数(动态规划)"""
         if num in (1, 2):
             return 1
         try:
             return temp[num]
         except KeyError:
             temp[num] = fib(num - 1) + fib(num - 2)
             return temp[num]
     ```

2. 函数的使用方式

   - 将函数视为“一等公民”

   - 高阶函数的用法（filter、map以及它们的替代品）

   - 位置参数、可变参数、关键字参数、命名关键字参数

   - 参数的元信息（代码可读性问题）

   - 匿名函数和内联函数的用法（lambda函数）

   - 闭包和作用域问题（LEGB）

   - 装饰器函数（使用装饰器和取消装饰器）

     输出函数执行时间的装饰器。

     ```Python
     from functools import wraps
     from time import time
     
     
     def record(output):
     	
     	def decorate(func):
     		
     		@wraps(func)
     		def wrapper(*args, **kwargs):
     			start = time()
     			result = func(*args, **kwargs)
     			output(func.__name__, time() - start)
     			return result
                 
     		return wrapper
     	
     	return decorate
     ```

     ```Python
     from functools import wraps
     from time import time
     
     
     class Record(object):
     
         def __init__(self, output):
             self.output = output
     
         def __call__(self, func):
     
             @wraps(func)
             def wrapper(*args, **kwargs):
                 start = time()
                 result = func(*args, **kwargs)
                 self.output(func.__name__, time() - start)
                 return result
     
             return wrapper
     ```

     用装饰器来实现单例模式。

     ```Python
     from functools import wraps
     
     
     def singleton(cls):
         instances = {}
     
         @wraps(cls)
         def wrapper(*args, **kwargs):
             if cls not in instances:
                 instances[cls] = cls(*args, **kwargs)
             return instances[cls]
     
         return wrapper
     
     
     @singleton
     class Singleton(object):
         pass
     ```

3. 面向对象相关知识

   - 三大支柱：封装、继承、多态

     ```Python
     """
     月薪结算系统
     部门经理每月15000 程序员每小时200 销售员1800底薪+销售额5%提成
     """
     from abc import ABCMeta, abstractmethod
     
     
     class Employee(metaclass=ABCMeta):
         """员工(抽象类)"""
     
         def __init__(self, name):
             self._name = name
     
         @property
         def name(self):
             """姓名"""
             return self._name
     
         @abstractmethod
         def get_salary(self):
             """结算月薪(抽象方法)"""
             pass
     
     
     class Manager(Employee):
         """部门经理"""
     
         def get_salary(self):
             return 15000.0
     
     
     class Programmer(Employee):
         """程序员"""
     
         def __init__(self, name):
             self._working_hour = 0
             super().__init__(name)
     
         @property
         def working_hour(self):
             """工作时间"""
             return self._working_hour
     
         @working_hour.setter
         def working_hour(self, hour):
             self._working_hour = hour if hour > 0 else 0
     
         def get_salary(self):
             return 200.0 * self.working_hour
     
     
     class Salesman(Employee):
         """销售员"""
     
         def __init__(self, name):
             self._sales = 0.0
             super().__init__(name)
     
         @property
         def sales(self):
             return self._sales
     
         @sales.setter
         def sales(self, sales):
             self._sales = sales if sales > 0 else 0
     
         def get_salary(self):
             return 1800.0 + self.sales * 0.05
     
     
     def main():
         emps = [
             Manager('刘备'), Manager('曹操'), Programmer('许褚'),
             Salesman('貂蝉'), Salesman('赵云'), Programmer('张辽'),
             Programmer('关羽'), Programmer('周瑜')
         ]
         for emp in emps:
             if isinstance(emp, Programmer):
                 emp.working_hour = int(input('本月工作时间: '))
             elif isinstance(emp, Salesman):
                 emp.sales = float(input('本月销售额: '))
             print('%s: %.2f元' % (emp.name, emp.get_salary()))
     
     
     if __name__ == '__main__':
         main()
     ```
     扑克牌类

     ```python
     """
     定义扑克和玩家把牌发到玩家手上
     is-a - 继承
     has-a - 关联/聚合/合成
     use-a - 依赖
     """
     import random
     
     from enum import Enum, unique
     
     
     # 经验1: 符号常量优于字面常量
     # 经验2：枚举类型是定义符号常量的首选
     @unique
     class Suite(Enum):
         """花色"""
         SPADE = 0
         HEART = 1
         CLUB = 2
         DIAMOND = 3
     
         def __lt__(self, other):
             return self.value < other.value
     
     
     class Card():
         """牌"""
     
         def __init__(self, suite, face):
             self.suite = suite
             self.face = face
     
         def __str__(self):
             suites = ['♠️', '♥️', '♣️', '♦️']
             faces = ['', 'A', '2', '3', '4', '5', '6', '7',
                      '8', '9', '10', 'J', 'Q', 'K']
             return f'{suites[self.suite.value]} {faces[self.face]}'
     
         def __repr__(self):
             return self.__str__()
     
     
     class Poker():
         """扑克"""
     
         def __init__(self):
             self.cards = [Card(suite, face) for suite in Suite
                           for face in range(1, 14)]
             self.index = 0
     
         def shuffle(self):
             """洗牌"""
             random.shuffle(self.cards)
             self.index = 0
     
         def deal(self):
             """发牌"""
             card = self.cards[self.index]
             self.index += 1
             return card
     
         @property
         def has_more(self):
             """还有没有牌"""
             return self.index < len(self.cards)
     
     
     class Player():
         """玩家"""
     
         def __init__(self, name):
             self.name = name
             self.cards = []
     
         def get_card(self, card):
             """摸牌"""
             self.cards.append(card)
     
         def sort(self, key=lambda x: (x.suite, x.face)):
             """整理手上的牌"""
             self.cards.sort(key=key)
     
     
     def main():
         """主函数"""
         poker = Poker()
         poker.shuffle()
         players = [Player('东邪'), Player('西毒'), Player('南帝'), Player('北丐')]
         for _ in range(13):
             for player in players:
                 if poker.has_more:
                     player.get_card(poker.deal())
         for player in players:
             player.sort()
             print(player.name, ':', end=' ')
             print(player.cards)
     
     
     if __name__ == '__main__':
         main()
     ```

     

   - 对象的复制（深复制和浅复制）

   - 垃圾回收、循环引用和弱引用

   - 魔法属性和方法（请参考《Python魔法方法指南》）

   - 混入（Mixin）

     ```Python
     """
     限制字典只有在指定的key不存在时才能设置键值对
     """
     
     
     class SetOnceMappingMixin:
         __slots__ = ()
     
         def __setitem__(self, key, value):
             if key in self:
                 raise KeyError(str(key) + ' already set')
             return super().__setitem__(key, value)
     
     
     class SetOnceDict(SetOnceMappingMixin, dict):
         pass
     
     
     def main():
         dict1 = SetOnceDict()
         try:
             dict1['username'] = 'jackfrued'
             dict1['username'] = 'hellokitty'
             dict1['username'] = 'wangdachui'
         except KeyError:
             pass
         print(dict1)
     
     
     if __name__ == '__main__':
         main()
     ```

   - 元编程和元类

     用元类实现单例模式。

     ```Python
     """
     通过元类实现单例模式
     """
     
     
     class SingletonMeta(type):
         """单例的元类"""
     
         def __init__(cls, *args, **kwargs):
             cls.__instance = None
             super().__init__(*args, **kwargs)
     
         def __call__(cls, *args, **kwargs):
             if cls.__instance is None:
                 cls.__instance = super().__call__(*args, **kwargs)
             return cls.__instance
     
     
     class Singleton(metaclass=SingletonMeta):
         """单例类"""
     
         def __init__(self, name):
             self._name = name
             from random import randrange
             self._value = randrange(100000)
     
         @property
         def name(self):
             return self._name
     
         @property
         def value(self):
             return self._value
     
     
     def main():
         sin1 = Singleton('Lee')
         sin2 = Singleton('Wang')
         print(sin1 == sin2)
         print(sin1.value, sin2.value)
         print(sin1.name, sin2.name)
     
     
     if __name__ == '__main__':
         main()
     ```

4. 迭代器和生成器

   ```Python
   """
   生成器和迭代器
   """
   
   
   def fib1(num):
       """普通函数"""
       a, b = 0, 1
       for _ in range(num):
           a, b = b, a + b
       return a
   
   
   def fib2(num):
       """生成器"""
       a, b = 0, 1
       for _ in range(num):
           a, b = b, a + b
           yield a
   
   
   class Fib3:
       """迭代器"""
   
       def __init__(self, num):
           self.num = num
           self.a, self.b = 0, 1
           self.idx = 0
   
       def __iter__(self):
           return self
   
       def __next__(self):
           if self.idx < self.num:
               self.a, self.b = self.b, self.a + self.b
               self.idx += 1
               return self.a
           raise StopIteration()
   
   
   def main():
       for val in fib2(20):
           print(val)
       print('-' * 50)
       for val in Fib3(20):
           print(val)
   
   
   if __name__ == '__main__':
       main()
   ```

5. 并发和异步编程
   - 多线程和多进程
   - 协程和异步I/O
   - concurrent.futures
