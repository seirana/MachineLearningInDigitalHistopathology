#global x
#x = 7
#
#def foo(x):
#    global y
#    y = 2
#    x = x * 2
#    print(x, y)
#foo(x)
#print (x, y*2)


#global x
x = "x"
def outer(x, a):
    
    b = "b"
    def inner(x):
        y = 2
        x = 2
        z  = "z"
        print("inner,:", x, y, z, a, b)
        if x ==5:
            print ("returnd")
            return
        x, y = inner2(x, z)
        return x, y 
        
    def inner2(x, z):
        y = 5
        print("inner2: ", x, y, a, b)
        inner3(z)
        return x, y
        
    def inner3(z):
        y = 5
        print("inner3: ", x, y, z, a, b)
        return x, y        
        
    print("step1:", x, a, b)
    x, y = inner(x)
    print("outer:", x, y, a, b) 
    return x, y
    
x, y = outer(x, "a")
print ("last step:", x, y)