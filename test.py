print("tester...")

class Foo(object):
    def __mul__(self, other):
        print ('__mul__')
        other = float(other)
        return other
    def __rmul__(self, other):
        print('__rmul__')
        return other


x = Foo()

print(2*x, x*2)

