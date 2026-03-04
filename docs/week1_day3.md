# Day 3: Functions and Modules - Building Reusable Code

Welcome to Day 3! Yesterday we learned how to make our code think and loop. However, as your scripts grow, writing everything in one massive file becomes a nightmare. 

In real-world AI engineering, you have to write code that is clean, maintainable, and reusable. Today, we learn how to achieve that using **Functions** and **Modules**.

## Defining Functions with `def`

A function is simply a block of code that only runs when it is called. You can pass data into it (parameters) and it can send data back (returns). 

We define functions using the `def` keyword.

```python
# A simple function that takes two parameters and returns their sum
def add_numbers(a, b):
    c = a + b
    return c

# Calling the function
result = add_numbers(10, 5)
print("Sum: ", result)
```

## Scope: Local vs Global Variables

Understanding *where* your variables live is critical. This is called "Scope".

1.  **Global Scope:** A variable created at the top of a script. It can be accessed *anywhere*, including inside functions.
2.  **Local Scope:** A variable created *inside* a function. It only exists inside that function!

```python
# Global Scope Example
greeting = "Hi"

def say_hello():
    # We can access the global variable here
    print(greeting + " from inside the function") 

# Local Scope Example
def greet_local():
    message = "Hello World" # This is Local!
    print(message)

# If we tried to print(message) outside the function, python would throw an error!
```

## Importing and Using Modules

What if you write a brilliant function that you want to use in five different projects? Do you copy-paste it? No! You turn it into a **Module**.

A Module is just a python file containing a set of functions or variables. Python also comes with huge built-in modules.

```python
import math as m # We import the built in math module and give it an alias 'm'

# We can now use functions from that module
print(m.sqrt(16)) # Output: 4.0
```

## Hands-On: Factorials and Custom Modules

Let's look at today's exercises!

### Exercise 1: Recursive Factorial
A great way to use functions is by having them call themselves! This is called recursion. Here is a script calculating the factorial of a number:

```python
# exercise1.py
def factorial(n):
    # Stop condition
    if n == 0 or n == 1:
        return 1
    # Recursive call
    else:
        return n * factorial(n - 1)

def print_factorial(n):
    result = factorial(n)
    print(f"The factorial of {n} is {result}")
    
print_factorial(5)
```

### Exercise 2: Building your own Module
Imagine we took our calculator functions from Day 2 and put them in a separate file named `math_operations.py`. We can now import that file into a new script just like we imported the built-in `math` module!

```python
# exercise2.py

# Import our custom file as a module
import math_operations as mo

num1 = 10
num2 = 5

# Call the functions defined in the other file!
print("Addition: ", mo.add(num1, num2))
print("Subtraction: ", mo.subtract(num1, num2))
print("Multiplication: ", mo.multiply(num1, num2))
print("Division: ", mo.divide(num1, num2))
```

## Wrapping Up Day 3
Functions and Modules are the very core of software design. By breaking your AI pipelines down into modular functions (e.g., `load_data()`, `clean_data()`, `train_model()`), your projects will be infinitely easier to manage. 

Tomorrow is arguably the most important day of the week: **Day 4: Data Structures**. We will learn exactly how to store and manipulate complex datasets using Lists, Tuples, Dictionaries, and Sets!
