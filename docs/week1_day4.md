# Day 4: Data Structures - The Backbone of Data Science

Welcome to Day 4! In AI and Machine Learning, you are constantly managing massive amounts of data. Individual variables like `x = 10` or `name = "Alice"` are not enough anymore.

We need ways to group data together, search through it, and modify it efficiently. Python provides four built-in Data Structures to do exactly this: **Lists, Tuples, Dictionaries, and Sets**.

## 1. Lists
A List is exactly what it sounds like: an ordered collection of items. Lists are *mutable*, meaning you can add, remove, and change items after the list is created.

```python
# Creating different types of lists
fruits = ["apple", "banana", "cherry"]
mixed_list = [1, "apple", True]

# Accessing an element (Lists are 0-indexed!)
print(fruits[0]) # Output: "apple"

# Modifying the list
fruits.append("orange") # Adds an item to the end
fruits.insert(1, "grape") # Inserts 'grape' at index 1
fruits.remove("banana") # Removes a specific item

# Slicing a list to get a sub-list
sliced_fruits = fruits[2:4] 
```

## 2. Tuples
Tuples are almost identical to Lists, but with one massive difference: they are **immutable**. Once created, you *cannot* change them. 

Why use them? Because they are faster than lists and guarantee that your data won't be accidentally modified by another part of your code.

```python
# Creating Tuples uses parenthesis ()
colors = ("red", "green", "blue")

# Creating a single item tuple requires a trailing comma
single_item = ("glass",)

print(colors[0]) # Output: red
# colors.append("yellow") # ERROR! Tuples cannot be modified!
```

## 3. Dictionaries
If Lists are arrays, Dictionaries are Key-Value mappings (similar to JSON objects). You look up data using completely custom keys instead of numerical indexes.

```python
# Creating a dictionary uses curly braces {} with key:value pairs
student = {"name": "Alice", "age": 25, "grade": "A"}

# Modifying data
student["subject"] = "Math" # Adds a new key-value pair
student["age"] = 32 # Updates an existing value

# Removing data
del student["grade"] 
```

## 4. Sets
Sets are unordered collections of *unique* elements. They are incredibly useful for removing duplicates from data or performing mathematical operations like Intersections and Unions.

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Set Difference: What is in set1 but NOT in set2?
print(set1 - set2) # Output: {1, 2}
```

## Hands-On Let's Code!

Let's apply these structures to real problems!

### Exercise 1: Manipulate Data in a Dictionary
Here, we take a basic dictionary, add new features to it, update existing data, and clean out data we no longer need (simulating data cleaning).

```python
# exercise1.py
person = {"name": "Alice", "age": 25, "grade": "A"}

# Adding a new Address Key
person["address"] = "123 Main St"

# Updating the Age Key
person["age"] = 32

# Safely cleaning out the Grade Key
if "grade" in person:
    del person["grade"]
    
print(person)
# Output: {'name': 'Alice', 'age': 32, 'address': '123 Main St'}
```

### Exercise 2: Word Frequency Counter (NLP Basics)
This is an incredibly common task in Natural Language Processing (NLP)—counting how many times specific words appear in a sentence. We use a combination of string `.split()` and a Dictionary to store the counts.

```python
# exercise2.py
sentence = input("Enter a Sentence: ")

# Split the string into a List of words
words = sentence.split()

# Initialize an empty Dictionary to hold our counts
word_count = {}

# Iterate over our List of words
for word in words:
    word = word.lower() # Normalize case so 'The' and 'the' match
    
    # Check if the word is already a Key in our Dictionary
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1 # Create the Key if it doesn't exist
        
print(word_count)
```

## Wrapping Up Day 4
You have now unlocked the heavy machinery of Python. Lists and Dictionaries will be used in almost every single AI script you write.

Tomorrow on **Day 5: Working with Strings**, we take a deep dive into manipulating text data, a crucial skill for prepping data for Large Language Models!
