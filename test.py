import csv

# Example arrays
array1 = [1, 2, 3, 4, 5]
array2 = ['a', 'b', 'c', 'd', 'e']

# Open a file in write mode
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the arrays to the file
    writer.writerow(array1)
    writer.writerow(array2)

# Open the file in read mode
with open('output.csv', mode='r') as file:
    reader = csv.reader(file)

    # Read the arrays from the file
    array1 = next(reader)
    array2 = next(reader)

# Convert the read strings to appropriate types if necessary
array1 = [int(i) for i in array1]
array2 = [str(i) for i in array2]

print("Array 1:", array1)
print("Array 2:", array2)