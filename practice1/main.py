from matrix import Matrix

def main():
    print("Hello World!")
    a = Matrix(2, 3)
    b = Matrix(3, 2)
    print("a matrix is:")
    print(a)
    print("b matrix is:")
    print(b)
    print("sum is: ")
    print(a + b)
    print("suntraction is: ")
    print(a - b)
    print("multiplication is: ")
    print(a * b)
    print("transposed is: ")
    print(a.transpose())

if __name__ == "__main__":
    main()