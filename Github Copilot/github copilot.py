# print hello world
print("Hello World")

# check a number is even or odd
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False
is_even(4)

# check a number is prime or not
def is_prime(number):
    for i in range(2, number):
        if number % i == 0:
            return False
    return True
is_prime(4)

# check a number is palindrome or not
def is_palindrome(number):
    number = str(number)
    if number == number[::-1]:
        return True
    else:
        return False
print(is_palindrome(1234321) )

# print 1 to 100 using loop
for i in range(1, 101):
    print(i)

# print 1 to 100 using while loop
i = 1
while i <= 100:
    print(i)
    i += 1

# print fibonacci series 
def fibonacci(number):
    if number == 1:
        return 1
    elif number == 2:
        return 1
    else:
        return fibonacci(number - 1) + fibonacci(number - 2)
print(fibonacci(10))

# check a value using binary search
def binary_search(list, value):
    low = 0
    high = len(list) - 1
    while low <= high:
        mid = (low + high) // 2
        if value == list[mid]:
            return mid
        elif value < list[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return -1
print(binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))

# sort a array using bubble sort
def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - i - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list
print(bubble_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

# print a triangle using loop
def print_triangle(number):
    for i in range(1, number + 1):
        print("*" * i)
print_triangle(5)

# print a rectangle using loop
def print_rectangle(width, height):
    for i in range(1, height + 1):
        print("*" * width)
print_rectangle(5, 5)

# implement a function to calculate factorial
def factorial(number):
    if number == 1:
        return 1
    else:
        return number * factorial(number - 1)
print(factorial(5))

# implemet bfs algorithm
def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited
graph = {'A': set(['B', 'C']), 'B': set(['A', 'D', 'E']), 'C': set(['A', 'F']), 'D': set(['B']), 'E': set(['B', 'F']), 'F': set(['C', 'E'])}
print(bfs(graph, 'A'))

# implement dfs algorithm
def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited
graph = {'A': set(['B', 'C']), 'B': set(['A', 'D', 'E']), 'C': set(['A', 'F']), 'D': set(['B']), 'E': set(['B', 'F']), 'F': set(['C', 'E'])}
print(dfs(graph, 'A'))

# multiply two matrices
def multiply_matrix(matrix1, matrix2):
    result = [[0 for i in range(len(matrix1))] for j in range(len(matrix2[0]))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]
print(multiply_matrix(matrix1, matrix2))


