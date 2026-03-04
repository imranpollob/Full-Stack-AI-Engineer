# Day 2: Control Flow - Giving Your Code a Brain

Welcome back! On Day 1, we learned how to write basic Python code and store data safely in variables. But what if we want our program to react differently based on that data? What if we want it to repeat a task a thousand times?

That's where **Control Flow** comes in. Today, we'll learn how to give your code a brain using Conditional Statements and Loops.

## Conditional Statements (`if`, `elif`, `else`)

In AI, your models are constantly making decisions. Even in standard programming, you need your code to branch off depending on the input. We do this using `if` statements.

*   `if`: Executes code *only* if a condition is True.
*   `elif` (Else If): Adds secondary conditions to check if the first `if` fails.
*   `else`: The fallback. Executes if *none* of the above conditions are met.

Let's look at a practical example from today's code:

```python
num = -50
if num > 0:
    print("Positive Number")
elif num == 0:
    print("Zero")
else:
    print("Negative Number")
```
*Note the indentation! Python uses indentation (spaces) to define blocks of code. If you don't indent properly, Python will throw an error.*

## The Power of Loops (`for` and `while`)

Often, you don't just want to do something once; you want to do it repeatedly. To train a Machine Learning model, you might need to loop over your dataset thousands of times (we call these "epochs").

### 1. The `for` Loop
A `for` loop is used to iterate over a sequence (like a List or a Range).

```python
# Looping through a list of items
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Looping a specific number of times using range()
for i in range(5): # This will print 0, 1, 2, 3, 4
    print(i)
```

### 2. The `while` Loop
A `while` loop continues to execute *as long as* a specified condition remains True. Be careful! If the condition never turns False, you'll create an "infinite loop."

```python
count = 5
while count > 0:
    print(count)
    count -= 1 # We must decrease the count, or it loops forever!
```

## Controlling the Loops (`break` and `continue`)

Sometimes you need to interrupt a loop mid-execution:
*   `break`: Stops the loop completely and immediately jumps out.
*   `continue`: Skips the rest of the current iteration and jumps to the next cycle.

```python
# This loop skips the number 5, but continues to 9
for i in range(10):
    if i == 5:
        continue
    print(i)
```

## Hands-On: A Menu-Driven Calculator

Let's combine everything we've learned today into a fully functional application! This is the solution for Exercise 2: building a calculator that runs continuously using a `while True` loop and breaks when the user types "5".

```python
# day2_exercise2.py

# First, we define some basic math functions (more on this on Day 3!)
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): 
    if b != 0: return a / b
    else: return "Division by zero!"

# The main Control Flow
while True: # Infinite loop to keep the menu running
    print("\nMenu:")
    print("1. Addition | 2. Subtraction | 3. Multiplication | 4. Division | 5. Exit")
    
    choice = input("Enter your choice: ")
    
    # Using 'break' to escape the infinite loop
    if choice == "5":
        print("Exiting Program.")
        break
    
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    
    # Conditionals to determine which math function to run
    if choice == "1":
        print("Result: ", add(num1, num2))
    elif choice == "2":
        print("Result: ", subtract(num1, num2))
    elif choice == "3":
        print("Result: ", multiply(num1, num2))
    elif choice == "4":
        print("Result: ", divide(num1, num2))
    else:
        print("Invalid choice. Please try again.")
```

## Wrapping Up Day 2
You can now write programs that *think* and automate repetitive tasks. Tomorrow is extremely important: we're going to dive deep into **Functions and Modules**. Functions allow us to package our code into reusable blocks, which is how large-scale AI applications are actually built. Stay tuned!
