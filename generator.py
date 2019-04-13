# A simple generator function
def my_gen():
    n = 1
    print('This is printed first')
    # Generator function contains yield statements
    yield n

    n += 1
    print('This is printed second')
    yield n

    n += 1
    print('This is printed at last')
    yield n
    
# Using for loop
for item in my_gen():
    print(item)        
###############################################################################   
def rev_str(my_str):
    length = len(my_str)
    for i in range(length - 1,-1,-1):
        yield my_str[i]

for char in rev_str("hello"):
     print(char)    
###############################################################################
# Program to show the use of lambda functions
double = lambda x: x * 2

# Output: 10
print(double(5)) 
###############################################################################
# Program to filter out only the even items from a list

my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(filter(lambda x: (x%2 == 0) , my_list))

# Output: [4, 6, 8, 12]
print(new_list)
###############################################################################
#function
def greet(name):
	"""This function greets to
	the person passed in as
	parameter"""
	print("Hello, " + name + ". Good morning!")
###############################################################################
#list
pow2 = [2 ** x for x in range(10)]

# Output: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
print(pow2)

#this code is equivalent to:
pow2 = []
for x in range(10):
   pow2.append(2 ** x)
###############################################################################
#list
for fruit in ['apple','banana','mango']:
    print("I like",fruit)
###############################################################################
print(list(range(10)))   
# Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
###############################################################################
digits = [0, 1, 5]

for i in digits:
    print(i)
else:
    print("No items left.")
###############################################################################
def greet2(name, msg = "Good morning!"):
   """
   This function greets to
   the person with the
   provided message.

   If message is not provided,
   it defaults to "Good
   morning!"
   """

   print("Hello",name + ', ' + msg)

greet2("Kate")
greet2("Bruce","How do you do?")
###############################################################################
# Intialize the list
my_list = [1, 3, 6, 10]

a = (x**2 for x in my_list)
# Output: 1
print(next(a))

# Output: 9
print(next(a))

# Output: 36
print(next(a))

# Output: 100
print(next(a))

# Output: StopIteration
next(a)
###############################################################################
#class
class PowTwo:
    def __init__(self, max = 0):
        self.max = max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.max:
            raise StopIteration

        result = 2 ** self.n
        self.n += 1
        return result
###############################################################################
#generator, this one equals to PowTow class
def PowTwoGen(max = 0):
    n = 0
    while n < max:
        yield 2 ** n
        n += 1        
###############################################################################
# list comprehension
doubles = [2 * n for n in range(50)]
 
# same as the list comprehension above
doubles = list(2 * n for n in range(50))        
###############################################################################
# Build and return a list
def firstn(n):
     num, nums = 0, []
     while num < n:
         nums.append(num)
         num += 1
     return nums
 
sum_of_first_n = sum(firstn(1000000))
###############################################################################
class firstn(object):
    def __init__(self, n):
        self.n = n
        self.num, self.nums = 0, []

    def __iter__(self):
        return self 
    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num+1
            return cur
        else:
            raise StopIteration()

sum_of_first_n = sum(firstn(1000000))
###############################################################################
# list comprehension
doubles = [2 * n for n in range(50)]

# same as the list comprehension above
doubles = list(2 * n for n in range(50))
###############################################################################
#An example of a class
class Shape:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return 2 * self.x + 2 * self.y

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x = self.x * scale
        self.y = self.y * scale
###############################################################################
# in , not in
a = [1, 2, 3, 4, 5]   
5 in a
5 not in a 
###############################################################################
#global
globvar = 10
def read1():
    print(globvar)
def write1():
    global globvar
    globvar = 5
def write2():
    globvar = 15
    
read1()
write1()
read1()
write2()
read1()    
###############################################################################
for i in 'hello':
    print(i)
###############################################################################
'''
is is used in Python for testing object identity 
While the == operator is used to test if two variables are equal or not, 
is is used to test if the two variables refer to the same object. 
'''  
True is True
[] == [] #True
[] is [] #False
{} == {} #True
{} is {} #False
'' == '' #True
'' is '' #True
() == () #True
() is () #True
'''
An empty list or dictionary is equal to another empty one. 
But they are not identical objects as they are located separately in memory. 
This is because list and dictionary are mutable (value can be changed).
string and tuple are immutable (value cannot be altered once defined). 
Hence, two equal string or tuple are identical as well. 
They refer to the same memory location.
'''
###############################################################################
'''
Lambda is used to create an anonymous function (function with no name). 
It is an inline function that does not contain a return statement. 
It consists of an expression that is evaluated and returned
'''
a = lambda x: x*2
for i in range(1,6):
    print(a(i))
###############################################################################
#nonlocal
'''
The use of nonlocal keyword is very much similar to the global keyword. 
nonlocal is used to declare that a variable inside a nested function 
(function inside a function) is not local to it, 
meaning it lies in the outer inclosing function.
'''    
def outer_function():
    a = 5
    def inner_function():
        nonlocal a
        a = 10
        print("Inner function: ",a)
    inner_function()
    print("Outer function: ",a)

outer_function()
###############################################################################  
#pass
#pass is a null statement in Python. Nothing happens when it is executed.
#empty function, empty class
def function(args):
    pass  

class example:
    pass
###############################################################################  
'''    
with statement is used to wrap the execution of a block of code
 within methods defined by the context manager
'''
with open('example.txt', 'w') as my_file:
    my_file.write('Hello world!')
###############################################################################  
#del is used to delete the reference to an object. Everything is object in Python. 
a = [1,2,3]
del a
###############################################################################
#yield
#yield is used inside a function like a return statement. But yield returns a generator.
def generator():
    for i in range(6):
        yield i*i

g = generator()
for i in g:
    print(i)
    
#the same as upper code    
g = (2**x for x in range(100))
next(g)
next(g)
next(g) 
next(g)
next(g) 
###############################################################################
matrix = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
[[row[i] for row in matrix] for i in range(4)]
#output [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
transposed = []
for i in range(4):
    transposed.append([row[i] for row in matrix])
#output [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]    
###############################################################################
chr(97) #a
ord('a') #97
###############################################################################
a = set('abracadabra') #{'a', 'r', 'b', 'c', 'd'}
b = set('alacazam')
a # unique letters in a
a-b #letters in a but not in b 
a|b #letters in a or b or both
a&b #letters in both a & b
a^b #letters in a or b but not both
###############################################################################
#dictionary
'''
Dictionaries are sometimes found in other languages as “associative memories” or “associative arrays”. 
Unlike sequences, which are indexed by a range of numbers, dictionaries are indexed by keys, 
which can be any immutable type; strings and numbers can always be keys. 
Tuples can be used as keys if they contain only strings, numbers, or tuples; 
if a tuple contains any mutable object either directly or indirectly, 
it cannot be used as a key. 
'''
tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
del tel['sape']
list(tel) #['jack', 'guido', 'irv']
'guido' in tel #True
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]) #The dict() constructor builds dictionaries directly from sequences of key-value pairs
{x: x**2 for x in (2, 4, 6)} #{2: 4, 4: 16, 6: 36}

'''
When the keys are simple strings, it is sometimes easier to specify pairs using keyword arguments
'''
dict(sape=4139, guido=4127, jack=4098) #{'sape': 4139, 'guido': 4127, 'jack': 4098}

knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)

#gallahad the pure
#robin the brave

for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
#0 tic
#1 tac
#2 toe
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q, a))

#What is your name?  It is lancelot.
#What is your quest?  It is the holy grail.
#What is your favorite color?  It is blue.

#to create a dictionary
# empty dictionary
my_dict = {}

# dictionary with integer keys
my_dict = {1: 'apple', 2: 'ball'}

# dictionary with mixed keys
my_dict = {'name': 'John', 1: [2, 4, 3]}

# using dict()
my_dict = dict({1:'apple', 2:'ball'})

# from sequence having each item as a pair
my_dict = dict([(1,'apple'), (2,'ball')])

#access the elements
my_dict = {'name':'Jack', 'age': 26}

# Output: Jack
print(my_dict['name'])

# Output: 26
print(my_dict.get('age'))
# add item
my_dict['address'] = 'Downtown'

# update value
my_dict['age'] = 27

# remove a particular item
print(my_dict.pop('age'))  

# remove an arbitrary item

print(my_dict.popitem()) #prints the first item

#remove all items
my_dict.clear() #output {}

list(my_dict.keys()) #['name', 'age']
list(my_dict.values()) #['Jack', 26]

del my_dict 
###############################################################################
for i in reversed(range(1, 10, 2)):
    print(i)
# 9 7 5 3 1   
###############################################################################
for i in reversed(range(1, 10, 2)):
    print(i)    
#apple banana orange pear    