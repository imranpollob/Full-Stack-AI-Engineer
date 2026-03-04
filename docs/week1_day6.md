# Day 6: File Handling - Making Data Persistent

Welcome to Day 6! Up until now, every time we ran a Python script, any data we created vanished the moment the script finished running. 

In AI, that is unacceptable. We need to save our cleaned data, save our trained models, and read in massive datasets (like CSVs or JSON files) from our hard drives. Today, we learn **File Handling**.

## Reading and Writing Text Files
Python makes interacting with files incredibly straightforward using the built-in `open()` function.

You need to tell Python two things: the name of the file, and what you want to do with it (the "mode").
*   `"r"`: Read mode (default).
*   `"w"`: Write mode (overwrites the file completely!).
*   `"a"`: Append mode (adds data to the end of the file).

### The Danger of Doing it Wrong
Normally, you would do this:
```python
file = open("data.txt", "r")
content = file.read()
file.close() # DO NOT FORGET THIS
```
If your program crashes before `file.close()` runs, that file might be corrupted or locked indefinitely.

## The Right Way: The `with` Statement
Python has a built-in "Context Manager" called the `with` statement. It guarantees that the file will be safely closed the moment you are done with it, *even if the program completely crashes*.

```python
# day6_samples.py
try:
    # Python automatically closes the file when this block finishes
    with open("sample.txt", "r") as file:
        content = file.read()
# We also use a try/except block just in case the file doesn't exist!
except FileNotFoundError:
    print("File Not Found!")
```

## Hands-On Let's Code!

Let's do some actual file manipulation.

### Exercise 1: Count Words and Lines in a File
Imagine you have a large text document and want some quick statistics. This script opens a file safely, reads all the lines, and calculates the word count using List comprehensions.

```python
# exercise1.py
def count_words_and_lines(filename):
    try:
        with open(filename, "r") as file:
            # .readlines() returns a List where each item is a line in the text file
            lines = file.readlines()
            
            line_count = len(lines)
            
            # We split each line by spaces, count the length of that list, and add them all together
            word_count = sum(len(line.split()) for line in lines)
            
            print(f"Number of lines: {line_count}")
            print(f"Number of words: {word_count}")
    except FileNotFoundError:
        print(f"File {filename} not found!")
        
count_words_and_lines("sample.txt")
```

### Exercise 2: Writing and Reading a List
Here is a complete pipeline demonstrating how to take data from our code (a Python List), save it permanently to the hard drive, and then prove it worked by reading it back into Python.

```python
# exercise2.py

# Function 1: Writing Data TO the hard drive
def write_item_to_file(filename, items):
    # Using "w" mode means we will create the file if it doesn't exist
    with open(filename, "w") as file:
        for item in items:
            # We must manually add the newline character "\n"
            file.write(item + "\n")
            
# Function 2: Reading Data FROM the hard drive
def read_items_from_file(filename):
    try:
        with open(filename, "r") as file:
            items = file.readlines()
            print("Items in the file:")
            for item in items:
                print(item.strip()) # .strip() removes that "\n" we added earlier
    except FileNotFoundError:
        print(f"File {filename} not found!")

# Let's test it out!     
fruits = ["Apple", "Banana", "Cherry", "Dates"]

# Save the list to disk
write_item_to_file("fruits.txt", fruits)

# Read it back
read_items_from_file("fruits.txt")
```

## Wrapping Up Day 6
You can now safely persist data to your hard drive and read it back! You have officially learned all the foundational mechanics of Python.

Tomorrow is the final day of Week 1: **Day 7: Pythonic Code and Project Work**. We will learn advanced tricks to make our code faster and cleaner, and build our very first command-line application from scratch!
