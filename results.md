Our three selected functions: get_increment, get_index, extract.

# Zubair's num.py

  * We can't initially understand what is going on with get_increment. 
   Perhaps you can provide some comments with your code. 

  * Your get_index looks good. It is very clean.
  
  * Once again, the code is hard to understand at first. Aim for better
   code clarity.

# Jinnie's num.py

  * The code for get_increment is very clean. It is easy to follow. 
   It is short, sweet, & simple.

  * For get_index, be sure to have an assert statement, and to have
   proper spacing between your operators. 

  * For extract, there is a clear understanding of what's going on 
   with the code. The comments are very helpful.

# Mike's num.py

  * Your get_increment could be cleaner if you had used the reduce function.

  * Clever use of the xrange function.

  * Your extract function is huge. Also, you have four for loops, two of which
   loop over the same iterator. Perhaps you can make this more concise. 

# Edith's num.py

  * Your get_increment is literally one long line past the 80 character limit.
   Try to space your code out a little more.

  * Your get_index function is very clear, and easy to follow. Your
   combo of a for loop and while loop wasn't too hard to read. 

  * Your extract function has a lot of for loops. Try to reduce these, and 
   maybe your code will be quicker. 

# Time Battles:

  * After running %timeit for each person's functions', Mike had the quickest
   functions from everyone, for all three functions. Here are the results:

| Name          | get_increment | get_index | extract |
| ------------- |:-------------:|:---------:| -------:|
| Zubair        | 6.35          |      7.09 |     418 |
| Jinnie        | 6.18          |      7.03 |     275 |
| Mike          | 2.12          |      2.60 |     249 |
| Edith         | 4.2           |      4.73 |     492 |