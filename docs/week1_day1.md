# Day 1: Introduction to Python and Development Setup

Welcome to Day 1! Today, we take our first actual steps into code. Python is the backbone of almost every major AI framework—from Scikit-Learn to PyTorch and TensorFlow. Before we can build intelligent systems, we need to understand how to speak the language.

## Why Python for AI?

You might wonder why Python, and not Java, C++, or Rust, dominates the AI landscape. The answer comes down to three things:
1. **Simplicity:** Python syntax is clean and readable, allowing researchers to focus on algorithms rather than boilerplate code.
2. **Ecosystem:** It has a massive, mature ecosystem of libraries. Need to do complex math? Import NumPy. Need to process massive tables? Import Pandas.
3. **Community:** Almost every new AI paper releases its code in Python first.

## Setting Up Your Environment
To write Python, you need two things:
1. **The Python Interpreter:** The program that reads and executes your Python code. You can download it from [python.org](https://www.python.org/downloads/).
2. **An IDE (Integrated Development Environment):** This is where you write your code. For this course, **Visual Studio Code (VS Code)** is highly recommended. It’s free, fast, and has excellent Python extensions. Later, we'll also use **Jupyter Notebooks**, which are fantastic for data exploration.

## Basic Syntax & Data Types
Let's get straight into it. Python code is executed line by line.

Here are the fundamental data types you will use constantly:
*   `Integers`: Whole numbers (e.g., `10`, `-5`)
*   `Floats`: Decimal numbers (e.g., `3.14`, `0.001`)
*   `Strings`: Text, always wrapped in quotes (e.g., `"AI"`, `'Machine Learning'`)
*   `Booleans`: True or False values (e.g., `True`, `False`)

Python also has powerful built-in ways to group data together:
*   `Lists`: Ordered, changeable arrays. Defined with `[]`.
*   `Tuples`: Ordered, *unchangeable* arrays. Defined with `()`.
*   `Dictionaries`: Key-value pairs, like a real dictionary. Defined with `{}`.

## Hands-On Let's Code!

Let's look at the actual code from today's exercises. 

### Exercise 1: Your First Script
It is a tradition to make your first program print out a greeting. In Python, this is incredibly simple using the built-in `print()` function.

```python
# day1_exercise1.py
print("Hello, AI World!")
```

### Exercise 2: Manipulating Variables
Here is how you define those different data types we talked about earlier, and how you can start interacting with them.

```python
# day1_exercise2.py

# Define variables of different data types
integer_var = 10
float_var = 3.14
string_var = "AI"
list_var = [1, 2, 3]
tuple_var = (4, 5, 6)
dict_var = {"name": "Alice", "role": "Engineer"}
bool_var = True

# Print and manipulate variables
print("Integer: ", integer_var)
print("Float: ", float_var)

# We can "concatenate" (join) strings together using the + operator
print("String: ", string_var + " Bootcamp")

# Lists are mutable, meaning we can add to them! Let's add the number 4
list_var.append(4)
print("List: ", list_var)

print("Tuple: ", tuple_var)

# We access data in a dictionary using its key
print("Dictionary Value:", dict_var["role"])
print("Boolean: ", bool_var)
```

**Your Turn:** Try changing the values in `dict_var` to reflect your own name and role, and run the script again!

## Wrapping Up Day 1
Congratulations, you've written your first Python code for AI! You now know how to install your tools, define basic variables, and manipulate primitive data structures. 

Tomorrow we level up. In **Day 2: Control Flow**, we'll learn how to make our programs smart enough to make decisions and repeat tasks automatically. See you then!
