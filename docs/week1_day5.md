# Day 5: Working with Strings - Prepping Data for NLP

Welcome to Day 5! Today we tackle text. 

Before an AI model (like ChatGPT) can understand human language, that language must be processed, cleaned, and organized. This entire process is called Natural Language Processing (NLP). Today, we learn the essential Python tools used to clean and parse text data.

## Fundamental String Manipulation
Strings in Python are actually arrays of characters, meaning we can do some pretty interesting things to them using built in string methods.

*   `.split()`: Breaks a string into a List of words.
*   `.join()`: Takes a List of words and stitches them back into a single String.
*   `.replace()`: Swaps one piece of text for another.
*   `.strip()`: Cleans up messy whitespace at the beginning or end of your text.

```python
# day5_strings.py

sentence = "Python is fun"
words = sentence.split() # Splits the sentence into a list: ['Python', 'is', 'fun']

# Join the list back together, separated by a pipe
new_sentence = "|".join(words) # Output: Python|is|fun

# Replace text
text = "I love Java"
updated_text = text.replace("Java", "Python")

# Strip whitespace
messy = "     Hello, World     "
cleaned_text = messy.strip() # Output: "Hello, World"
```

## The Heavy Lifter: Regular Expressions (Regex)

Basic string methods are great, but what if you need to find *all* phone numbers in a massive document, but you don't know what the numbers are?

You need **Regular Expressions**. Regex is a specialized mini-language used purely for pattern matching. Python handles this with the built-in `re` module.

*   `re.search(pattern, string)`: Looks for the first occurrence of a pattern.
*   `re.findall(pattern, string)`: Finds *all* matches and returns them as a list.
*   `re.sub(pattern, replacement, string)`: Finds a pattern and replaces it.

```python
# day5_regex.py
import re

text = "Contact me at 123-456-7890"

# Find all 1 or more digits
digits = re.findall(r"\d+", text) 
print(digits) # Output: ['123', '456', '7890']

# Replace all single digits with the letter 'X' (Redaction!)
updated_text = re.sub(r"\d", "X", text)
print(updated_text) # Output: Contact me at XXX-XXX-XXXX
```

## Hands-On Let's Code!

Let's look at today's exercises!

### Exercise 1: The NLP Text Cleaner
Real world text is messy. It's full of punctuation, weird capitalization, and double spaces. Here is an actual script you could use to clean a dataset before feeding it to an AI model.

```python
# exercise1.py
import re

def clean_text(text):
    # The Regex pattern `[^\w\s]` targets anything that is NOT a word or a space
    # We replace those matches with nothing ("") to remove punctuation.
    text = re.sub(r"[^\w\s]", "", text)
    
    # Split text into a list (removing extra spaces) and join it back with single spaces
    text = " ".join(text.split())
    
    # Convert everything to lowercase so the AI treats 'Apple' and 'apple' as the same word
    return text.lower()

input_text = "   Hello, World.!!! Welcome to Python, Programming....    "
cleaned_text = clean_text(input_text)
print("Cleaned Text: ", cleaned_text)
# Output: Cleaned Text:  hello world welcome to python programming
```

### Exercise 2: The Palindrome Checker
A palindrome is a word that is spelled the same backward and forward (like "Racecar"). We can use Python string manipulation to easily strip out spaces and check this!

```python
# exercise2.py
def is_palindrome(text):
    # This complex line uses a "list comprehension" (we learn this on Day 7!)
    # It removes anything that isn't a letter or number, and forces it to lowercase
    text = "".join(char.lower() for char in text if char.isalnum())
    
    # Python lets us reverse a string using this slice syntax: [::-1]
    return text == text[::-1]

input_text = input("Enter a string: ")
if is_palindrome(input_text):
    print(f'"{input_text}" is a palindrome.')
```

## Wrapping Up Day 5
You now know how to extract data (like phone numbers) from raw messy paragraphs, and how to sanitize text for NLP pipelines. 

Tomorrow on **Day 6: File Handling**, we learn how to actually get that text out of a file on your hard drive and into your Python script!
