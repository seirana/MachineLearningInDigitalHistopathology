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

###############################################################################
    
    