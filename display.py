import tkinter as tk
from tkinter import filedialog
from tkinter import *
import csv

root = tk.Tk()
root.geometry("500x250")
root.title("CSV File Browser")

def browse_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if filename:
        print("Selected CSV file:", filename)
        fname=Label(root, text = filename)
        fname.grid(row = 1,column= 2)
        # Process the CSV file
        '''
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)  # Print each row to the console

        '''
        

# Create a button to trigger file browsing
browse_button = tk.Button(root, text="Browse CSV", command=browse_file)
browse_button.grid(row=0,column=4)

username=Label(root,text = "filename")
username.grid(row = 1, column = 0)
c1 = tk.Checkbutton(root, text='downsampling')#, command=print_selection)
c1.grid(row=2,column=2)

browse_button = tk.Button(root, text="check")#, command=check)
browse_button.grid(row=2,column=4)

# Create the main application window



# Run the Tkinter event loop
root.mainloop()
