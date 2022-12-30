import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')   
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

# My sorting (e.g. bubble sort)
# ADD HERE YOUR CODE
def bubble_sort(my_numbers):        
    for i in range(len(my_numbers)):        # running through the list  X  elemement amount
        for j in range(len(my_numbers) - 1):        # setting up so the elements next to each other are correct. last two n-2 and n-1
            if my_numbers[j] > my_numbers[j+1]:         # exchanging if is true 
                my_numbers[j], my_numbers[j+1] = my_numbers[j+1], my_numbers[j]

bubble_sort(my_numbers)

# Print
print(f'After sorting {my_numbers}')